import numpy as np
import pytest
import shutil

from robocasa.scripts.backfill_sonic_wrist_cameras import (
    AlignmentError,
    AlignmentThresholds,
    RAW_ROBOT_STATE_COLUMNS,
    _dataset_identity_manifest,
    _prepare_work_manifest,
    align_source_states_to_raw,
    extract_raw_robot_state,
    validate_partial_baseline,
)


def test_extract_raw_robot_state_reorders_hand_joints():
    raw = np.arange(2 * 60, dtype=np.float64).reshape(2, 60)
    result = extract_raw_robot_state(raw)
    assert result.shape == (2, 43)
    np.testing.assert_array_equal(result[0], raw[0, RAW_ROBOT_STATE_COLUMNS])
    np.testing.assert_array_equal(
        result[0, 22:29], raw[0, [33, 34, 35, 36, 30, 31, 32]]
    )
    np.testing.assert_array_equal(
        result[0, 36:43], raw[0, [47, 48, 49, 50, 44, 45, 46]]
    )


def test_alignment_keeps_duplicates_and_fills_off_grid_frame():
    raw = np.zeros((45, 3), dtype=np.float64)
    raw[:, 0] = np.arange(45)
    raw[:, 1] = np.arange(45) * 0.1
    source_indices = np.array([2, 6, 10, 14, 18, 22, 22, 30, 34, 38, 42])
    source = raw[source_indices].copy()
    source[7] = (raw[26] + raw[27]) / 2
    thresholds = AlignmentThresholds(
        minimum_anchor_fraction=0.80,
        minimum_slope=3.0,
        maximum_slope=5.0,
        maximum_p99_l2=1.0,
        maximum_l2=1.0,
    )

    result, metrics = align_source_states_to_raw(source, raw, thresholds)

    np.testing.assert_array_equal(result[:7], source_indices[:7])
    assert result[6] == result[5]
    assert result[7] in (26, 27)
    np.testing.assert_array_equal(result[8:], source_indices[8:])
    assert np.all(np.diff(result) >= 0)
    assert metrics["duplicate_mapping_steps"] == 1
    assert metrics["anchor_frames"] == 10


def test_alignment_spreads_stationary_exact_state_over_raw_time():
    raw = np.zeros((41, 3), dtype=np.float64)
    raw[:11, 0] = np.arange(11)
    raw[:11, 1] = np.arange(11) * 0.1
    raw[10:31] = raw[10]
    raw[31:, 0] = np.arange(11, 21)
    raw[31:, 1] = np.arange(11, 21) * 0.1
    source_indices = np.array([2, 6, 10, 14, 18, 22, 26, 30, 34, 38])
    source = raw[source_indices]

    result, metrics = align_source_states_to_raw(source, raw)

    np.testing.assert_array_equal(result, source_indices)
    assert metrics["longest_duplicate_mapping_run"] == 0
    assert metrics["duplicate_mapping_steps"] == 0
    assert metrics["backward_steps"] == 0


def test_alignment_resolves_revisited_pose_with_temporal_neighbors():
    raw = np.zeros((41, 3), dtype=np.float64)
    raw[:, 0] = np.arange(41)
    raw[:, 1] = np.arange(41) * 0.1
    raw[30] = raw[10]
    source_indices = np.array([2, 6, 10, 14, 18, 22, 26, 30, 34, 38])

    result, metrics = align_source_states_to_raw(raw[source_indices], raw)

    np.testing.assert_array_equal(result, source_indices)
    assert metrics["backward_steps"] == 0


@pytest.mark.parametrize("stationary_slice", [slice(0, 17), slice(24, 41)])
def test_alignment_extrapolates_stationary_prefix_and_suffix(stationary_slice):
    raw = np.zeros((41, 3), dtype=np.float64)
    raw[:, 0] = np.arange(41)
    raw[:, 1] = np.arange(41) * 0.1
    boundary = stationary_slice.stop - 1 if stationary_slice.start == 0 else 24
    raw[stationary_slice] = raw[boundary]
    source_indices = np.arange(0, 41, 4)

    result, metrics = align_source_states_to_raw(raw[source_indices], raw)

    np.testing.assert_array_equal(result, source_indices)
    assert metrics["longest_duplicate_mapping_run"] == 0


def test_alignment_uses_anchor_cadence_for_padded_stationary_prefix():
    raw = np.zeros((49, 3), dtype=np.float64)
    raw[:, 0] = np.arange(49)
    raw[:, 1] = np.arange(49) * 0.1
    raw[5:18] = raw[17]
    source_indices = np.arange(5, 46, 4)

    result, _metrics = align_source_states_to_raw(raw[source_indices], raw)

    np.testing.assert_array_equal(result, source_indices)


def test_alignment_uses_anchor_cadence_for_padded_stationary_suffix():
    raw = np.zeros((51, 3), dtype=np.float64)
    raw[:, 0] = np.arange(51)
    raw[:, 1] = np.arange(51) * 0.1
    raw[26:43] = raw[26]
    source_indices = np.arange(2, 43, 4)

    result, _metrics = align_source_states_to_raw(raw[source_indices], raw)

    np.testing.assert_array_equal(result, source_indices)


def test_alignment_rejects_fully_stationary_ambiguous_episode():
    raw = np.zeros((41, 3), dtype=np.float64)
    source = np.zeros((11, 3), dtype=np.float64)

    with pytest.raises(AlignmentError, match="unambiguous exact anchors"):
        align_source_states_to_raw(source, raw)


def test_alignment_rejects_wrong_pair():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(400, 43))
    source = rng.normal(size=(100, 43))
    with pytest.raises(AlignmentError, match="anchor_fraction"):
        align_source_states_to_raw(source, raw)


def test_alignment_rejects_large_forward_time_jump():
    rng = np.random.default_rng(11)
    raw = rng.normal(size=(500, 43))
    source = rng.normal(size=(100, 43))
    mapped = np.r_[
        np.arange(50) * 4,
        np.rint(220 + np.arange(50) * (176 / 49)).astype(np.int64),
    ]
    raw[mapped] = source

    with pytest.raises(AlignmentError, match="maximum raw step=24"):
        align_source_states_to_raw(source, raw)


def test_partial_resume_requires_exact_baseline(tmp_path):
    baseline = tmp_path / "baseline"
    partial = tmp_path / "output.partial"
    (baseline / "meta").mkdir(parents=True)
    (baseline / "data").mkdir()
    (baseline / "meta" / "info.json").write_text("{}\n")
    (baseline / "data" / "episode.parquet").write_bytes(b"payload")
    shutil.copytree(baseline, partial)
    wrist = (
        partial
        / "videos/chunk-000/observation.images.left_wrist/episode_000000.mp4"
    )
    wrist.parent.mkdir(parents=True)
    wrist.write_bytes(b"generated")
    manifest = _dataset_identity_manifest(baseline)

    validate_partial_baseline(baseline, partial, manifest)
    (partial / "meta" / "info.json").write_text("{\"changed\": true}\n")
    with pytest.raises(ValueError, match="partial baseline file differs"):
        validate_partial_baseline(baseline, partial, manifest)


def test_work_manifest_refuses_different_spec(tmp_path):
    partial = tmp_path / "partial"
    (partial / "meta").mkdir(parents=True)
    first = _prepare_work_manifest(partial, {"identity": "one"})
    assert _prepare_work_manifest(partial, {"identity": "one"}) == first
    with pytest.raises(RuntimeError, match="does not match"):
        _prepare_work_manifest(partial, {"identity": "two"})
