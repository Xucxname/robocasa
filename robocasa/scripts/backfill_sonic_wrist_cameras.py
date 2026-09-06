#!/usr/bin/env python3
"""Backfill synchronized SonicG1 wrist videos into a cleaned LeRobot dataset.

The original dataset remains untouched.  A sibling ``.partial`` directory is
built from it, checked against the unfiltered LeRobot rows and raw MuJoCo
states, rendered, validated, and atomically promoted only after every episode
passes.  Existing videos in the partial directory make the operation resumable.

This tool is intentionally fail-closed: an incorrect raw/source pairing, an
ambiguous time alignment, or a video with the wrong frame count aborts the
operation before the final dataset is published.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from fractions import Fraction
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Sequence
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/robocasa_numba_cache")

import av
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from robocasa.scripts.clean_sonic_dataset import (
    DEFAULT_POSE_MODES,
    SMPL_POSE_COLUMN,
    STREAM_MODE_COLUMN,
    build_mode_aware_stale_mask,
    get_parquet_path,
    get_video_path,
    inspect_video,
    load_info,
    read_jsonlines,
    verify_dataset,
)
from robocasa.scripts.dataset_scripts.convert_sonic_hdf5_lerobot import (
    VIRTUAL_WRIST_CAMERA_SPECS,
    _make_env_from_hdf5,
    _prepare_model_xml,
)


CAMERAS = (
    ("robot0_left_wrist_camera", "observation.images.left_wrist"),
    ("robot0_right_wrist_camera", "observation.images.right_wrist"),
)

ENCODING_CONFIG = {
    "codec": "libx264",
    "pixel_format": "yuv420p",
    "crf": 23,
    "preset": "fast",
    "threads_per_stream": 1,
}

# Raw flattened MuJoCo qpos order differs from the 43D LeRobot state order.
# The first seven qpos after mjData.time are the floating base.
RAW_ROBOT_STATE_COLUMNS = np.r_[
    np.arange(8, 30),
    np.arange(33, 37),
    np.arange(30, 33),
    np.arange(37, 44),
    np.arange(47, 51),
    np.arange(44, 47),
]


class AlignmentError(RuntimeError):
    """Raised when a raw/source state alignment fails a safety gate."""


@dataclass(frozen=True)
class AlignmentThresholds:
    exact_l2: float = 1e-6
    minimum_anchor_fraction: float = 0.80
    warning_anchor_fraction: float = 0.95
    maximum_p99_l2: float = 0.02
    maximum_l2: float = 0.05
    minimum_slope: float = 2.0
    maximum_slope: float = 6.0
    maximum_slope_to_frame_ratio_error: float = 0.25
    maximum_unanchored_run: int = 100
    maximum_raw_step: int = 16
    maximum_consecutive_duplicate_steps: int = 10
    minimum_unambiguous_anchor_frames: int = 2


@dataclass(frozen=True)
class EpisodeJob:
    output_episode_index: int
    source_episode_index: int
    raw_episode: str
    raw_hdf5: str
    source_parquet: str
    cleaned_parquet: str
    output_root: str
    source_frames: int
    output_frames: int
    raw_render_indices: tuple[int, ...]
    raw_hdf5_bytes: int
    raw_hdf5_sha256: str
    source_parquet_bytes: int
    source_parquet_sha256: str
    cleaned_parquet_bytes: int
    cleaned_parquet_sha256: str
    trusted_video_sha256: tuple[tuple[str, str], ...]
    width: int
    height: int
    fps: int


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_write(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _stack_column(frame: pd.DataFrame, column: str, dtype) -> np.ndarray:
    return np.vstack(
        [np.asarray(value, dtype=dtype) for value in frame[column].to_numpy()]
    )


def extract_raw_robot_state(raw_states: np.ndarray) -> np.ndarray:
    """Extract raw robot qpos and reorder the two seven-joint hands."""

    states = np.asarray(raw_states)
    if states.ndim != 2:
        raise ValueError(f"raw states must be 2-D, got {states.shape}")
    if states.shape[1] <= int(RAW_ROBOT_STATE_COLUMNS.max()):
        raise ValueError(
            f"raw states have only {states.shape[1]} columns; need at least "
            f"{int(RAW_ROBOT_STATE_COLUMNS.max()) + 1}"
        )
    return np.asarray(states[:, RAW_ROBOT_STATE_COLUMNS], dtype=np.float64)


def _longest_false_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _longest_true_run(mask: np.ndarray) -> int:
    return _longest_false_run(~np.asarray(mask, dtype=bool))


def _refine_exact_anchor_indices(
    source: np.ndarray,
    raw: np.ndarray,
    tree: cKDTree,
    anchors: np.ndarray,
    nearest_index: np.ndarray,
    exact_l2: float,
    frame_ratio: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Resolve spatially exact anchor ties with temporal neighbors.

    ``cKDTree.query(..., k=1)`` may choose any one of several identical raw
    states.  That is unsafe for stationary robot stretches because scene
    objects can still move while the 43D robot state remains unchanged.  Keep
    unique exact matches as hard temporal landmarks, then choose each tied
    candidate nearest the piecewise-linear time prediction between landmarks.
    """

    result = np.asarray(nearest_index, dtype=np.int64).copy()
    anchor_source = np.flatnonzero(anchors)
    if not len(anchor_source):
        return result, {
            "unambiguous_anchor_frames": 0,
            "ambiguous_anchor_frames": 0,
            "temporally_refined_anchor_frames": 0,
            "longest_ambiguous_anchor_run": 0,
        }

    candidate_lists: list[np.ndarray] = []
    ambiguous_mask = np.zeros(len(source), dtype=bool)
    unique_source: list[int] = []
    unique_raw: list[int] = []
    for source_index in anchor_source:
        candidates = np.sort(
            np.asarray(
                tree.query_ball_point(source[int(source_index)], r=exact_l2),
                dtype=np.int64,
            )
        )
        if not len(candidates):
            raise AlignmentError(
                f"anchor {int(source_index)} has no exact raw candidate"
            )
        candidate_lists.append(candidates)
        if len(candidates) == 1:
            unique_source.append(int(source_index))
            unique_raw.append(int(candidates[0]))
        else:
            ambiguous_mask[int(source_index)] = True

    unique_source_array = np.asarray(unique_source, dtype=np.int64)
    unique_raw_array = np.asarray(unique_raw, dtype=np.int64)
    if len(unique_raw_array) and np.any(np.diff(unique_raw_array) < 0):
        backwards = int(np.sum(np.diff(unique_raw_array) < 0))
        raise AlignmentError(
            f"unique exact anchors have {backwards} backward steps"
        )

    unique_source_steps = np.diff(unique_source_array)
    unique_raw_steps = np.diff(unique_raw_array)
    positive = (unique_source_steps > 0) & (unique_raw_steps > 0)
    exact_step_slopes = unique_raw_steps[positive] / unique_source_steps[positive]
    if len(exact_step_slopes):
        edge_count = min(32, len(exact_step_slopes))
        prefix_slope = float(np.median(exact_step_slopes[:edge_count]))
        suffix_slope = float(np.median(exact_step_slopes[-edge_count:]))
    else:
        prefix_slope = suffix_slope = frame_ratio

    previous = 0
    for source_index, candidates in zip(anchor_source, candidate_lists):
        source_index = int(source_index)
        position = int(np.searchsorted(unique_source_array, source_index))
        has_left = position > 0
        has_right = position < len(unique_source_array)

        lower = int(unique_raw_array[position - 1]) if has_left else 0
        upper = int(unique_raw_array[position]) if has_right else len(raw) - 1
        lower = max(lower, previous)
        allowed = candidates[(candidates >= lower) & (candidates <= upper)]
        if not len(allowed):
            raise AlignmentError(
                f"anchor {source_index} has no monotonic exact candidate in "
                f"raw interval [{lower}, {upper}]"
            )

        if has_left and has_right:
            left_source = int(unique_source_array[position - 1])
            right_source = int(unique_source_array[position])
            left_raw = int(unique_raw_array[position - 1])
            right_raw = int(unique_raw_array[position])
            if right_source == left_source:
                predicted = float(left_raw)
            else:
                fraction = (source_index - left_source) / (
                    right_source - left_source
                )
                predicted = left_raw + fraction * (right_raw - left_raw)
        elif has_left:
            left_source = int(unique_source_array[position - 1])
            left_raw = int(unique_raw_array[position - 1])
            predicted = left_raw + suffix_slope * (source_index - left_source)
        elif has_right:
            right_source = int(unique_source_array[position])
            right_raw = int(unique_raw_array[position])
            predicted = right_raw - prefix_slope * (right_source - source_index)
        else:
            predicted = frame_ratio * source_index

        chosen = int(allowed[np.argmin(np.abs(allowed - predicted))])
        result[source_index] = chosen
        previous = chosen
    return result, {
        "unambiguous_anchor_frames": len(unique_source),
        "ambiguous_anchor_frames": int(ambiguous_mask.sum()),
        "temporally_refined_anchor_frames": int(
            np.sum(result[anchor_source] != nearest_index[anchor_source])
        ),
        "longest_ambiguous_anchor_run": _longest_true_run(ambiguous_mask),
    }


def _spread_duplicate_runs_over_exact_candidates(
    source: np.ndarray,
    raw: np.ndarray,
    mapping: np.ndarray,
    tree: cKDTree,
    exact_l2: float,
    frame_ratio: float,
) -> np.ndarray:
    """Resolve exact-state ties using neighboring temporal anchors.

    A stationary robot can have the same 43D state at many raw timesteps.  A
    spatial nearest-neighbor query may collapse all corresponding source rows
    onto one arbitrary timestep even while scene objects are still moving.
    This pass spreads only such duplicate runs across exact candidates bounded
    by their neighboring mapped frames.  If only one exact raw state exists,
    the duplicate is retained and reported for the fail-closed run-length gate.
    """

    result = np.asarray(mapping, dtype=np.int64).copy()
    equal_steps = np.diff(result) == 0
    cursor = 0
    while cursor < len(equal_steps):
        if not equal_steps[cursor]:
            cursor += 1
            continue
        step_start = cursor
        while cursor + 1 < len(equal_steps) and equal_steps[cursor + 1]:
            cursor += 1
        step_end = cursor
        frame_start = step_start
        frame_end = step_end + 1
        left_raw = (
            int(result[frame_start - 1])
            if frame_start > 0
            else max(0, int(round(result[frame_start] - frame_ratio)))
        )
        right_raw = (
            int(result[frame_end + 1])
            if frame_end + 1 < len(result)
            else min(len(raw) - 1, int(round(result[frame_end] + frame_ratio)))
        )
        previous = left_raw
        denominator = frame_end - frame_start + 2
        for frame_index in range(frame_start, frame_end + 1):
            fraction = (frame_index - frame_start + 1) / denominator
            predicted = left_raw + fraction * (right_raw - left_raw)
            exact_candidates = np.asarray(
                tree.query_ball_point(source[frame_index], r=exact_l2),
                dtype=np.int64,
            )
            allowed = exact_candidates[
                (exact_candidates >= previous) & (exact_candidates <= right_raw)
            ]
            if len(allowed):
                chosen = int(allowed[np.argmin(np.abs(allowed - predicted))])
                result[frame_index] = chosen
                previous = chosen
        cursor += 1
    return result


def _nearest_with_time_tiebreak(
    target: np.ndarray,
    raw_states: np.ndarray,
    lower: int,
    upper: int,
    predicted: float,
) -> tuple[int, float]:
    if lower > upper:
        raise AlignmentError(f"empty raw search interval [{lower}, {upper}]")
    candidates = raw_states[lower : upper + 1]
    distances = np.linalg.norm(candidates - target, axis=1)
    minimum = float(distances.min())
    # Numerical ties occur during genuine stationary/duplicate stretches.  Use
    # the nominal capture time only as a tie-breaker, never as the main cost.
    tied = np.flatnonzero(np.isclose(distances, minimum, rtol=0.0, atol=1e-12))
    candidate_indices = tied + lower
    chosen = int(candidate_indices[np.argmin(np.abs(candidate_indices - predicted))])
    return chosen, float(distances[chosen - lower])


def align_source_states_to_raw(
    source_states: np.ndarray,
    raw_robot_states: np.ndarray,
    thresholds: AlignmentThresholds = AlignmentThresholds(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map each 50 Hz source row to a 200 Hz raw state.

    Exact/near-exact robot-state matches form temporal anchors.  Isolated
    off-grid rows are filled by state nearest-neighbor search inside adjacent
    anchor intervals.  Duplicate raw indices are deliberately retained.
    """

    source = np.asarray(source_states, dtype=np.float64)
    raw = np.asarray(raw_robot_states, dtype=np.float64)
    if source.ndim != 2 or raw.ndim != 2 or source.shape[1] != raw.shape[1]:
        raise ValueError(
            f"state shape mismatch: source={source.shape}, raw={raw.shape}"
        )
    if len(source) < 2 or len(raw) < 2:
        raise AlignmentError("need at least two source and raw states")

    tree = cKDTree(raw)
    nearest_l2, nearest_index = tree.query(source, k=1, workers=1)
    nearest_l2 = np.asarray(nearest_l2, dtype=np.float64)
    nearest_index = np.asarray(nearest_index, dtype=np.int64)
    anchors = nearest_l2 <= thresholds.exact_l2
    frame_ratio = (len(raw) - 1) / (len(source) - 1)
    nearest_index, anchor_refinement = _refine_exact_anchor_indices(
        source,
        raw,
        tree,
        anchors,
        nearest_index,
        thresholds.exact_l2,
        frame_ratio,
    )
    anchor_source = np.flatnonzero(anchors)
    anchor_raw = nearest_index[anchors]
    anchor_fraction = float(anchors.mean())

    failures: list[str] = []
    if anchor_fraction < thresholds.minimum_anchor_fraction:
        failures.append(
            f"anchor_fraction={anchor_fraction:.6f} < "
            f"{thresholds.minimum_anchor_fraction:.6f}"
        )
    if len(anchor_source) < 2:
        failures.append(f"only {len(anchor_source)} exact anchors")
    if (
        anchor_refinement["unambiguous_anchor_frames"]
        < thresholds.minimum_unambiguous_anchor_frames
    ):
        failures.append(
            "only "
            f"{anchor_refinement['unambiguous_anchor_frames']} "
            "unambiguous exact anchors < "
            f"{thresholds.minimum_unambiguous_anchor_frames}"
        )
    anchor_backwards = int(np.sum(np.diff(anchor_raw) < 0)) if len(anchor_raw) else 0
    if anchor_backwards:
        failures.append(f"anchor mapping has {anchor_backwards} backward steps")

    if len(anchor_source) >= 2:
        slope, intercept = np.polyfit(anchor_source, anchor_raw, 1)
    else:
        slope, intercept = np.nan, np.nan
    if not np.isfinite(slope) or not (
        thresholds.minimum_slope <= slope <= thresholds.maximum_slope
    ):
        failures.append(
            f"alignment slope={slope!r} outside "
            f"[{thresholds.minimum_slope}, {thresholds.maximum_slope}]"
        )
    slope_to_frame_ratio_error = (
        abs(float(slope) / frame_ratio - 1.0) if np.isfinite(slope) else np.inf
    )
    if slope_to_frame_ratio_error > thresholds.maximum_slope_to_frame_ratio_error:
        failures.append(
            f"slope/frame-ratio relative error={slope_to_frame_ratio_error:.6f} > "
            f"{thresholds.maximum_slope_to_frame_ratio_error:.6f}"
        )

    longest_unanchored = _longest_false_run(anchors)
    if longest_unanchored > thresholds.maximum_unanchored_run:
        failures.append(
            f"longest unanchored run={longest_unanchored} > "
            f"{thresholds.maximum_unanchored_run}"
        )
    if failures:
        raise AlignmentError("; ".join(failures))

    result = nearest_index.copy()
    final_l2 = nearest_l2.copy()
    non_anchor = np.flatnonzero(~anchors)
    for source_index in non_anchor:
        left_position = int(np.searchsorted(anchor_source, source_index) - 1)
        right_position = left_position + 1
        if left_position >= 0:
            left_source = int(anchor_source[left_position])
            lower = int(anchor_raw[left_position])
        else:
            left_source = None
            lower = 0
        if right_position < len(anchor_source):
            right_source = int(anchor_source[right_position])
            upper = int(anchor_raw[right_position])
        else:
            right_source = None
            upper = len(raw) - 1

        if left_source is not None and right_source is not None:
            fraction = (source_index - left_source) / (right_source - left_source)
            predicted = lower + fraction * (upper - lower)
        else:
            predicted = slope * source_index + intercept
        chosen, distance = _nearest_with_time_tiebreak(
            source[source_index], raw, lower, upper, predicted
        )
        result[source_index] = chosen
        final_l2[source_index] = distance

    result = _spread_duplicate_runs_over_exact_candidates(
        source,
        raw,
        result,
        tree,
        thresholds.exact_l2,
        frame_ratio,
    )
    final_l2 = np.linalg.norm(source - raw[result], axis=1)

    backward_steps = int(np.sum(np.diff(result) < 0))
    raw_steps = np.diff(result)
    maximum_raw_step = int(raw_steps.max()) if len(raw_steps) else 0
    longest_duplicate_run = _longest_true_run(raw_steps == 0)
    p99_l2 = float(np.quantile(final_l2, 0.99))
    maximum_l2 = float(final_l2.max())
    maximum_linf = float(np.max(np.abs(source - raw[result])))
    if backward_steps:
        failures.append(f"final mapping has {backward_steps} backward steps")
    if maximum_raw_step > thresholds.maximum_raw_step:
        failures.append(
            f"maximum raw step={maximum_raw_step} > "
            f"{thresholds.maximum_raw_step}"
        )
    if longest_duplicate_run > thresholds.maximum_consecutive_duplicate_steps:
        failures.append(
            f"longest duplicate mapping run={longest_duplicate_run} > "
            f"{thresholds.maximum_consecutive_duplicate_steps}"
        )
    if p99_l2 > thresholds.maximum_p99_l2:
        failures.append(
            f"state L2 p99={p99_l2:.6g} > {thresholds.maximum_p99_l2:.6g}"
        )
    if maximum_l2 > thresholds.maximum_l2:
        failures.append(
            f"state L2 max={maximum_l2:.6g} > {thresholds.maximum_l2:.6g}"
        )

    metrics = {
        "source_frames": int(len(source)),
        "raw_frames": int(len(raw)),
        "exact_l2_threshold": thresholds.exact_l2,
        "anchor_frames": int(anchors.sum()),
        **anchor_refinement,
        "anchor_fraction": anchor_fraction,
        "anchor_fraction_warning": (
            anchor_fraction < thresholds.warning_anchor_fraction
        ),
        "longest_unanchored_run": int(longest_unanchored),
        "ols_raw_ticks_per_source_frame": float(slope),
        "ols_intercept": float(intercept),
        "raw_to_source_frame_ratio": float(frame_ratio),
        "slope_to_frame_ratio_relative_error": float(slope_to_frame_ratio_error),
        "backward_steps": backward_steps,
        "maximum_raw_step": maximum_raw_step,
        "longest_duplicate_mapping_run": int(longest_duplicate_run),
        "duplicate_mapping_steps": int(np.sum(np.diff(result) == 0)),
        "unique_raw_frames": int(len(np.unique(result))),
        "first_raw_index": int(result[0]),
        "last_raw_index": int(result[-1]),
        "state_l2_median": float(np.median(final_l2)),
        "state_l2_p99": p99_l2,
        "state_l2_max": maximum_l2,
        "state_linf_max": maximum_linf,
    }
    if failures:
        raise AlignmentError("; ".join(failures))
    return result, metrics


def _sha256_bytes(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _dataset_identity_manifest(dataset: Path) -> dict[str, dict[str, Any]]:
    """Hash every regular baseline file, including all metadata."""

    result: dict[str, dict[str, Any]] = {}
    for path in sorted(dataset.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"dataset must not contain symlinks: {path}")
        if path.is_file():
            result[str(path.relative_to(dataset))] = _file_fingerprint(path)
    return result


def _allowed_partial_extra(relative_path: str) -> bool:
    if relative_path in {
        "meta/camera_backfill_work.json",
        "meta/camera_backfill_work.json.tmp",
        "meta/camera_backfill_stream_work.json",
        "meta/camera_backfill_stream_work.json.tmp",
    }:
        return True
    path = Path(relative_path)
    if path.suffix != ".mp4":
        return False
    return any(
        part in {
            "observation.images.left_wrist",
            "observation.images.right_wrist",
        }
        for part in path.parts
    )


def validate_partial_baseline(
    baseline: Path,
    partial: Path,
    baseline_manifest: dict[str, dict[str, Any]],
) -> None:
    """Require partial to contain an exact baseline plus allowed work files."""

    if partial.is_symlink():
        raise ValueError(f"partial dataset must not be a symlink: {partial}")
    actual_paths: set[str] = set()
    for path in partial.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"partial dataset must not contain symlinks: {path}")
        if path.is_file():
            actual_paths.add(str(path.relative_to(partial)))
    missing = sorted(set(baseline_manifest) - actual_paths)
    unexpected = sorted(
        path
        for path in actual_paths - set(baseline_manifest)
        if not _allowed_partial_extra(path)
    )
    if missing or unexpected:
        raise ValueError(
            f"partial is not an allowed copy of {baseline}: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    for relative_path, expected in baseline_manifest.items():
        actual = _file_fingerprint(partial / relative_path)
        if actual != expected:
            raise ValueError(
                f"partial baseline file differs: {relative_path}: "
                f"{actual} != {expected}"
            )


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _raw_hdf5_index(raw_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(raw_root.rglob("ep_demo.hdf5")):
        name = path.parent.name
        if name in result:
            raise ValueError(
                f"duplicate raw episode {name}: {result[name]} and {path}"
            )
        result[name] = path
    return result


def _single_demo(raw_file: h5py.File):
    if "data" not in raw_file:
        raise ValueError(f"raw HDF5 has no /data group: {raw_file.filename}")
    keys = sorted(key for key in raw_file["data"] if key.startswith("demo_"))
    if len(keys) != 1:
        raise ValueError(
            f"expected one demo in {raw_file.filename}, found {len(keys)}"
        )
    return raw_file["data"][keys[0]]


def build_episode_jobs(
    cleaned_dataset: Path,
    source_dataset: Path,
    raw_root: Path,
    output_root: Path,
    thresholds: AlignmentThresholds,
    output_episode_indices: set[int] | None = None,
) -> tuple[list[EpisodeJob], list[dict[str, Any]]]:
    """Join provenance records and preflight every source/raw alignment."""

    cleaned_info = load_info(cleaned_dataset)
    source_info = load_info(source_dataset)
    cleaned_episodes = read_jsonlines(cleaned_dataset / "meta" / "episodes.jsonl")
    cleanup = _json_load(cleaned_dataset / "meta" / "cleanup_report.json")
    replay = _json_load(cleaned_dataset / "meta" / "raw_replay_alignment.json")
    raw_index = _raw_hdf5_index(raw_root)

    mappings = [
        entry
        for entry in cleanup["episode_mapping"]
        if int(entry["output_episode_index"]) >= 0
    ]
    selected_entries = [
        entry
        for entry in replay["matches"]
        if entry.get("selection") == "keep_replay_passed"
    ]
    if len(mappings) != len(cleaned_episodes):
        raise ValueError(
            f"cleanup mapping has {len(mappings)} kept episodes, metadata has "
            f"{len(cleaned_episodes)}"
        )
    if len(selected_entries) != len(cleaned_episodes):
        raise ValueError(
            f"replay alignment has {len(selected_entries)} selected episodes, metadata has "
            f"{len(cleaned_episodes)}"
        )
    output_indices = [int(entry["output_episode_index"]) for entry in mappings]
    source_indices = [int(entry["source_episode_index"]) for entry in mappings]
    selected_source_indices = [
        int(entry["lerobot_episode_index"]) for entry in selected_entries
    ]
    selected_raw_episodes = [str(entry["raw_episode"]) for entry in selected_entries]
    expected_outputs = list(range(len(cleaned_episodes)))
    uniqueness_checks = {
        "cleanup output_episode_index": output_indices,
        "cleanup source_episode_index": source_indices,
        "replay lerobot_episode_index": selected_source_indices,
        "replay raw_episode": selected_raw_episodes,
    }
    for label, values in uniqueness_checks.items():
        if len(values) != len(set(values)):
            raise ValueError(f"{label} values are not unique")
    if sorted(output_indices) != expected_outputs:
        raise ValueError(
            f"cleanup output indices are not contiguous: {sorted(output_indices)}"
        )
    if set(source_indices) != set(selected_source_indices):
        raise ValueError(
            "cleanup kept source indices differ from replay-passed source indices"
        )
    metadata_indices = [int(entry["episode_index"]) for entry in cleaned_episodes]
    if metadata_indices != expected_outputs:
        raise ValueError("cleaned episodes.jsonl indices are not contiguous")
    if output_episode_indices is not None:
        unknown = set(output_episode_indices) - set(expected_outputs)
        if unknown:
            raise ValueError(f"unknown requested output episodes: {sorted(unknown)}")
        mappings_to_build = [
            entry
            for entry in mappings
            if int(entry["output_episode_index"]) in output_episode_indices
        ]
    else:
        mappings_to_build = mappings
    selected = {
        int(entry["lerobot_episode_index"]): entry for entry in selected_entries
    }

    jobs: list[EpisodeJob] = []
    reports: list[dict[str, Any]] = []
    for mapping in sorted(
        mappings_to_build, key=lambda item: item["output_episode_index"]
    ):
        output_index = int(mapping["output_episode_index"])
        source_index = int(mapping["source_episode_index"])
        if source_index not in selected:
            raise ValueError(f"source episode {source_index} has no passed raw match")
        match = selected[source_index]
        raw_episode = str(match["raw_episode"])
        if raw_episode not in raw_index:
            raise FileNotFoundError(
                f"raw HDF5 for {raw_episode} is absent under {raw_root}"
            )
        raw_hdf5 = raw_index[raw_episode]
        source_parquet = get_parquet_path(source_dataset, source_info, source_index)
        cleaned_parquet = get_parquet_path(cleaned_dataset, cleaned_info, output_index)
        raw_fingerprint = _file_fingerprint(raw_hdf5)
        source_fingerprint = _file_fingerprint(source_parquet)
        cleaned_fingerprint = _file_fingerprint(cleaned_parquet)
        source_frame = pd.read_parquet(
            source_parquet,
            columns=["observation.state", SMPL_POSE_COLUMN, STREAM_MODE_COLUMN],
        )
        cleaned_frame = pd.read_parquet(
            cleaned_parquet,
            columns=["observation.state"],
        )
        source_state = _stack_column(source_frame, "observation.state", np.float64)
        cleaned_state = _stack_column(cleaned_frame, "observation.state", np.float64)
        smpl_pose = _stack_column(source_frame, SMPL_POSE_COLUMN, np.float32)
        stream_mode = source_frame[STREAM_MODE_COLUMN].to_numpy()
        stale = build_mode_aware_stale_mask(
            smpl_pose, stream_mode, DEFAULT_POSE_MODES
        )
        valid_indices = np.flatnonzero(~stale)

        expected_source = int(mapping["source_frames"])
        expected_output = int(mapping["output_frames"])
        metadata_output = int(cleaned_episodes[output_index]["length"])
        if not (
            len(source_frame) == expected_source
            and len(valid_indices) == expected_output
            and len(cleaned_frame) == expected_output
            and metadata_output == expected_output
        ):
            raise ValueError(
                f"episode {output_index} length mismatch: source={len(source_frame)}/"
                f"{expected_source}, retained={len(valid_indices)}/{expected_output}, "
                f"cleaned={len(cleaned_frame)}, metadata={metadata_output}"
            )
        if not np.array_equal(source_state[valid_indices], cleaned_state):
            maximum = float(np.max(np.abs(source_state[valid_indices] - cleaned_state)))
            raise ValueError(
                f"episode {output_index} cleaned/source state mismatch, max={maximum}"
            )

        with h5py.File(raw_hdf5, "r") as raw_file:
            demo = _single_demo(raw_file)
            raw_states = np.asarray(demo["states"][:], dtype=np.float64)
        raw_robot_state = extract_raw_robot_state(raw_states)
        source_to_raw, metrics = align_source_states_to_raw(
            source_state, raw_robot_state, thresholds
        )
        render_indices = source_to_raw[valid_indices]
        if np.any(np.diff(render_indices) < 0):
            raise AlignmentError(
                f"episode {output_index} retained mapping moves backward"
            )

        job = EpisodeJob(
            output_episode_index=output_index,
            source_episode_index=source_index,
            raw_episode=raw_episode,
            raw_hdf5=str(raw_hdf5),
            source_parquet=str(source_parquet),
            cleaned_parquet=str(cleaned_parquet),
            output_root=str(output_root),
            source_frames=len(source_frame),
            output_frames=len(valid_indices),
            raw_render_indices=tuple(int(value) for value in render_indices),
            raw_hdf5_bytes=int(raw_fingerprint["bytes"]),
            raw_hdf5_sha256=str(raw_fingerprint["sha256"]),
            source_parquet_bytes=int(source_fingerprint["bytes"]),
            source_parquet_sha256=str(source_fingerprint["sha256"]),
            cleaned_parquet_bytes=int(cleaned_fingerprint["bytes"]),
            cleaned_parquet_sha256=str(cleaned_fingerprint["sha256"]),
            trusted_video_sha256=(),
            width=int(cleaned_info["features"]["observation.images.ego_view"]["shape"][1]),
            height=int(cleaned_info["features"]["observation.images.ego_view"]["shape"][0]),
            fps=int(cleaned_info["fps"]),
        )
        jobs.append(job)
        reports.append(
            {
                "output_episode_index": output_index,
                "source_episode_index": source_index,
                "raw_episode": raw_episode,
                "raw_hdf5": raw_fingerprint,
                "source_parquet": source_fingerprint,
                "cleaned_parquet": cleaned_fingerprint,
                "source_frames": len(source_frame),
                "output_frames": len(valid_indices),
                "retained_source_indices_sha256": _sha256_bytes(
                    valid_indices.astype("<i8")
                ),
                "raw_render_indices_sha256": _sha256_bytes(
                    render_indices.astype("<i8")
                ),
                "retained_unique_raw_frames": int(len(np.unique(render_indices))),
                "retained_duplicate_mapping_steps": int(
                    np.sum(np.diff(render_indices) == 0)
                ),
                "alignment": metrics,
            }
        )
        print(
            f"[align output={output_index:02d}] "
            f"source={source_index} raw={raw_episode} "
            f"anchors={metrics['anchor_fraction']:.3%} "
            f"unique={metrics['unambiguous_anchor_frames']} "
            f"ambiguous={metrics['ambiguous_anchor_frames']} "
            f"ambiguous_run={metrics['longest_ambiguous_anchor_run']} "
            f"slope={metrics['ols_raw_ticks_per_source_frame']:.4f} "
            f"max_l2={metrics['state_l2_max']:.6g}",
            flush=True,
        )
    return jobs, reports


def _video_details(path: Path) -> dict[str, Any]:
    container = av.open(str(path))
    try:
        if len(container.streams.video) != 1:
            raise ValueError(f"expected one video stream in {path}")
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(stream))
        return {
            "frames": frames,
            "width": int(stream.codec_context.width),
            "height": int(stream.codec_context.height),
            "fps": float(stream.average_rate) if stream.average_rate else 0.0,
            "codec": str(stream.codec_context.name),
            "pixel_format": str(stream.codec_context.pix_fmt),
            "audio_streams": len(container.streams.audio),
        }
    finally:
        container.close()


def _validate_video(path: Path, frames: int, width: int, height: int, fps: int) -> dict:
    details = _video_details(path)
    expected = {
        "frames": frames,
        "width": width,
        "height": height,
        "fps": float(fps),
        "codec": "h264",
        "pixel_format": "yuv420p",
        "audio_streams": 0,
    }
    mismatches = {
        key: (details[key], value)
        for key, value in expected.items()
        if not (
            np.isclose(details[key], value)
            if key == "fps"
            else details[key] == value
        )
    }
    if mismatches:
        raise ValueError(f"video validation failed for {path}: {mismatches}")
    details["bytes"] = path.stat().st_size
    details["sha256"] = _sha256_file(path)
    return details


def _open_video_writer(path: Path, width: int, height: int, fps: int):
    container = av.open(str(path), mode="w")
    stream = container.add_stream(ENCODING_CONFIG["codec"], rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = ENCODING_CONFIG["pixel_format"]
    stream.time_base = Fraction(1, fps)
    stream.codec_context.thread_count = ENCODING_CONFIG["threads_per_stream"]
    stream.options = {
        "crf": str(ENCODING_CONFIG["crf"]),
        "preset": ENCODING_CONFIG["preset"],
        "threads": str(ENCODING_CONFIG["threads_per_stream"]),
    }
    return container, stream


def _validate_canonical_wrist_cameras(model_xml: str, camera_names: Sequence[str]) -> None:
    root = ET.fromstring(model_xml)
    located: dict[str, tuple[str, ET.Element]] = {}
    for body in root.iter("body"):
        for camera in body.findall("camera"):
            name = camera.attrib.get("name")
            if name in camera_names:
                if name in located:
                    raise ValueError(f"camera {name} appears more than once in model XML")
                located[name] = (str(body.attrib.get("name")), camera)
    for camera_name in camera_names:
        if camera_name not in located:
            raise ValueError(f"camera {camera_name} is absent after injection")
        body_name, camera = located[camera_name]
        expected = VIRTUAL_WRIST_CAMERA_SPECS[camera_name]
        if body_name != expected["body"]:
            raise ValueError(
                f"camera {camera_name} parent {body_name!r} != {expected['body']!r}"
            )
        if camera.attrib.get("mode", "fixed") != "fixed":
            raise ValueError(f"camera {camera_name} is not fixed")
        for attribute, expected_text in expected["attributes"].items():
            actual_text = camera.attrib.get(attribute)
            if actual_text is None:
                raise ValueError(f"camera {camera_name} lacks {attribute}")
            actual_values = np.fromstring(actual_text, sep=" ")
            expected_values = np.fromstring(str(expected_text), sep=" ")
            if actual_values.shape != expected_values.shape or not np.allclose(
                actual_values, expected_values, rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    f"camera {camera_name} {attribute} differs from canonical config: "
                    f"{actual_text!r} != {expected_text!r}"
                )


def _render_episode(job_dict: dict[str, Any]) -> dict[str, Any]:
    """Worker entry point: render any missing wrist video for one episode."""

    job = EpisodeJob(**job_dict)
    root = Path(job.output_root)
    raw_hdf5 = Path(job.raw_hdf5)
    source_parquet = Path(job.source_parquet)
    cleaned_parquet = Path(job.cleaned_parquet)
    dependency_checks = (
        (raw_hdf5, job.raw_hdf5_bytes, job.raw_hdf5_sha256),
        (source_parquet, job.source_parquet_bytes, job.source_parquet_sha256),
        (cleaned_parquet, job.cleaned_parquet_bytes, job.cleaned_parquet_sha256),
    )
    for path, expected_bytes, expected_sha256 in dependency_checks:
        actual = _file_fingerprint(path)
        if actual != {"bytes": expected_bytes, "sha256": expected_sha256}:
            raise RuntimeError(f"render dependency changed after preflight: {path}")

    info = load_info(root)
    trusted_video_sha256 = dict(job.trusted_video_sha256)
    final_paths = {
        camera_name: get_video_path(
            root, info, job.output_episode_index, video_key
        )
        for camera_name, video_key in CAMERAS
    }
    missing: list[str] = []
    details: dict[str, dict[str, Any]] = {}
    for camera_name, path in final_paths.items():
        if path.is_file():
            try:
                existing = _validate_video(
                    path, job.output_frames, job.width, job.height, job.fps
                )
                relative = str(path.relative_to(root))
                if trusted_video_sha256.get(relative) == existing["sha256"]:
                    details[camera_name] = existing
                    continue
            except Exception:
                pass
            path.unlink()
        missing.append(camera_name)

    if missing:
        temporary_paths = {
            name: final_paths[name].with_name(final_paths[name].stem + ".tmp.mp4")
            for name in missing
        }
        for path in temporary_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                path.unlink()

        env_args = SimpleNamespace(
            camera_names=list(missing),
            camera_height=job.height,
            camera_width=job.width,
            infer_rewards=False,
            inject_virtual_wrist_cameras=True,
        )
        env = _make_env_from_hdf5(raw_hdf5, env_args)
        writers: dict[str, tuple[Any, Any]] = {}
        try:
            with h5py.File(raw_hdf5, "r") as raw_file:
                demo = _single_demo(raw_file)
                states = np.asarray(demo["states"][:], dtype=np.float64)
                model_xml = _prepare_model_xml(demo.attrs["model_file"], env_args)
                _validate_canonical_wrist_cameras(model_xml, missing)
                first_index = int(job.raw_render_indices[0])
                env.reset_to(
                    {
                        "states": states[first_index],
                        "model": model_xml,
                        "ep_meta": demo.attrs.get("ep_meta"),
                    }
                )
                base_env = getattr(env, "env", env)
                for camera_name in missing:
                    writers[camera_name] = _open_video_writer(
                        temporary_paths[camera_name], job.width, job.height, job.fps
                    )

                for frame_number, raw_index in enumerate(job.raw_render_indices):
                    base_env.sim.set_state_from_flattened(states[int(raw_index)])
                    base_env.sim.forward()
                    for camera_name in missing:
                        container, stream = writers[camera_name]
                        del container
                        rgb = base_env.sim.render(
                            width=job.width,
                            height=job.height,
                            camera_name=camera_name,
                        )[::-1]
                        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                        frame.pts = frame_number
                        frame.time_base = Fraction(1, job.fps)
                        for packet in stream.encode(frame):
                            writers[camera_name][0].mux(packet)

                for camera_name in missing:
                    container, stream = writers[camera_name]
                    for packet in stream.encode():
                        container.mux(packet)
                    container.close()
                writers.clear()

            for camera_name in missing:
                temporary = temporary_paths[camera_name]
                _validate_video(
                    temporary, job.output_frames, job.width, job.height, job.fps
                )
                temporary.replace(final_paths[camera_name])
        finally:
            for container, _stream in writers.values():
                container.close()
            base_env = getattr(env, "env", None)
            if base_env is not None and hasattr(base_env, "close"):
                base_env.close()

    for camera_name, video_key in CAMERAS:
        path = final_paths[camera_name]
        item = _validate_video(
            path, job.output_frames, job.width, job.height, job.fps
        )
        item["path"] = str(path.relative_to(root))
        details[video_key] = item
        details.pop(camera_name, None)
    return {
        "output_episode_index": job.output_episode_index,
        "rendered_camera_count": len(missing),
        "videos": details,
    }


def _camera_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {
        "ego_view": {
            "image_key": "ego_view",
            "camera_name": "robot0_head_camera",
            "feature_key": "observation.images.ego_view",
            "provenance": "original_teleoperation_capture",
        }
    }
    for camera_name, video_key in CAMERAS:
        spec = VIRTUAL_WRIST_CAMERA_SPECS[camera_name]
        image_key = video_key.rsplit(".", 1)[-1]
        result[image_key] = {
            "image_key": image_key,
            "camera_name": camera_name,
            "feature_key": video_key,
            "provenance": "raw_mujoco_state_replay_render",
            "injected_when_missing": True,
            "mujoco_body": spec["body"],
            **spec["attributes"],
        }
    return result


def _build_work_spec(
    baseline: Path,
    source: Path,
    raw_root: Path,
    baseline_manifest: dict[str, dict[str, Any]],
    thresholds: AlignmentThresholds,
    jobs: Sequence[EpisodeJob],
    alignment_reports: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    reports = {
        int(entry["output_episode_index"]): entry for entry in alignment_reports
    }
    episodes = []
    for job in jobs:
        report = reports[job.output_episode_index]
        episodes.append(
            {
                "output_episode_index": job.output_episode_index,
                "source_episode_index": job.source_episode_index,
                "raw_episode": job.raw_episode,
                "source_frames": job.source_frames,
                "output_frames": job.output_frames,
                "raw_hdf5": {
                    "bytes": job.raw_hdf5_bytes,
                    "sha256": job.raw_hdf5_sha256,
                },
                "source_parquet": {
                    "bytes": job.source_parquet_bytes,
                    "sha256": job.source_parquet_sha256,
                },
                "cleaned_parquet": {
                    "bytes": job.cleaned_parquet_bytes,
                    "sha256": job.cleaned_parquet_sha256,
                },
                "retained_source_indices_sha256": report[
                    "retained_source_indices_sha256"
                ],
                "raw_render_indices_sha256": report[
                    "raw_render_indices_sha256"
                ],
            }
        )
    return {
        "implementation_version": 1,
        "implementation_sha256": _sha256_file(Path(__file__).resolve()),
        "baseline_dataset": str(baseline),
        "source_dataset": str(source),
        "raw_root": str(raw_root),
        "baseline_manifest": baseline_manifest,
        "source_info": _file_fingerprint(source / "meta" / "info.json"),
        "thresholds": asdict(thresholds),
        "cameras": _camera_metadata(),
        "encoding": ENCODING_CONFIG,
        "episodes": episodes,
    }


def _prepare_work_manifest(partial: Path, spec: dict[str, Any]) -> dict[str, Any]:
    path = partial / "meta" / "camera_backfill_work.json"
    if path.is_file():
        work = _json_load(path)
        if work.get("spec") != spec:
            raise RuntimeError(
                "partial work manifest does not match the current baseline/raw/alignment"
            )
        if not isinstance(work.get("completed_videos"), dict):
            raise RuntimeError("partial work manifest has invalid completed_videos")
        return work
    work = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "spec": spec,
        "completed_videos": {},
    }
    _json_write(path, work)
    return work


def _add_trusted_video_hashes(
    jobs: Sequence[EpisodeJob],
    partial: Path,
    work: dict[str, Any],
) -> list[EpisodeJob]:
    info = load_info(partial)
    completed = work["completed_videos"]
    result = []
    for job in jobs:
        trusted = []
        for _camera_name, video_key in CAMERAS:
            path = get_video_path(partial, info, job.output_episode_index, video_key)
            relative = str(path.relative_to(partial))
            record = completed.get(relative)
            if isinstance(record, dict) and isinstance(record.get("sha256"), str):
                trusted.append((relative, record["sha256"]))
        result.append(replace(job, trusted_video_sha256=tuple(trusted)))
    return result


def _record_completed_videos(
    partial: Path,
    work: dict[str, Any],
    render_report: dict[str, Any],
) -> None:
    for details in render_report["videos"].values():
        work["completed_videos"][details["path"]] = {
            "bytes": int(details["bytes"]),
            "sha256": details["sha256"],
        }
    work["updated_at"] = datetime.now().astimezone().isoformat()
    _json_write(partial / "meta" / "camera_backfill_work.json", work)


def _validate_job_dependencies(jobs: Sequence[EpisodeJob]) -> None:
    for job in jobs:
        checks = (
            (Path(job.raw_hdf5), job.raw_hdf5_bytes, job.raw_hdf5_sha256),
            (
                Path(job.source_parquet),
                job.source_parquet_bytes,
                job.source_parquet_sha256,
            ),
            (
                Path(job.cleaned_parquet),
                job.cleaned_parquet_bytes,
                job.cleaned_parquet_sha256,
            ),
        )
        for path, expected_bytes, expected_sha256 in checks:
            if _file_fingerprint(path) != {
                "bytes": expected_bytes,
                "sha256": expected_sha256,
            }:
                raise RuntimeError(f"dependency changed during rendering: {path}")


def _validate_preserved_baseline(
    partial: Path,
    baseline_manifest: dict[str, dict[str, Any]],
) -> None:
    mutable = {"meta/info.json", "meta/modality.json"}
    for relative_path, expected in baseline_manifest.items():
        if relative_path in mutable:
            continue
        actual = _file_fingerprint(partial / relative_path)
        if actual != expected:
            raise RuntimeError(f"baseline payload was modified: {relative_path}")


def _update_metadata(
    dataset: Path,
    baseline: Path,
    source: Path,
    raw_root: Path,
    thresholds: AlignmentThresholds,
    alignment_reports: list[dict[str, Any]],
    render_reports: list[dict[str, Any]],
    immutable_manifest: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    info_path = dataset / "meta" / "info.json"
    info = _json_load(info_path)
    original_record_wrist = info.get("script_config", {}).get(
        "record_wrist_cameras"
    )
    ego_feature = deepcopy(info["features"]["observation.images.ego_view"])
    for _camera_name, video_key in CAMERAS:
        info["features"][video_key] = deepcopy(ego_feature)
    info["total_videos"] = int(info["total_episodes"]) * 3
    info.setdefault("script_config", {})["record_wrist_cameras"] = True
    _json_write(info_path, info)

    modality_path = dataset / "meta" / "modality.json"
    modality = _json_load(modality_path)
    video = modality.setdefault("video", {})
    for _camera_name, video_key in CAMERAS:
        image_key = video_key.rsplit(".", 1)[-1]
        video[image_key] = {"original_key": video_key}
    _json_write(modality_path, modality)
    _json_write(dataset / "meta" / "cameras.json", _camera_metadata())

    render_by_episode = {
        int(entry["output_episode_index"]): entry for entry in render_reports
    }
    for entry in alignment_reports:
        entry["videos"] = render_by_episode[entry["output_episode_index"]]["videos"]
    report = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "operation": "backfill_synchronized_sonic_wrist_cameras",
        "provenance": {
            "ego_view": "pixels captured during original teleoperation",
            "left_wrist": "rendered from recorded raw MuJoCo states",
            "right_wrist": "rendered from recorded raw MuJoCo states",
            "original_record_wrist_cameras": original_record_wrist,
            "final_record_wrist_cameras": True,
            "parquet_rewritten": False,
            "existing_ego_videos_rewritten": False,
        },
        "alignment_method": (
            "43D observation.state exact anchors plus monotonic interval-constrained "
            "nearest-state interpolation; retained cleaned rows selected by the "
            "mode-aware stale-pose mask"
        ),
        "thresholds": asdict(thresholds),
        "source": {
            "baseline_dataset": baseline.name,
            "unfiltered_lerobot_dataset": source.name,
            "raw_hdf5_dataset": raw_root.name,
            "immutable_file_manifest": immutable_manifest,
        },
        "result": {
            "total_episodes": int(info["total_episodes"]),
            "total_frames": int(info["total_frames"]),
            "total_videos": int(info["total_videos"]),
            "camera_count": 3,
            "wrist_videos_added": int(info["total_episodes"]) * 2,
        },
        "episodes": alignment_reports,
    }
    _json_write(dataset / "meta" / "camera_backfill_report.json", report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-dataset", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cleaned = args.cleaned_dataset.expanduser().resolve()
    source = args.source_dataset.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    output = args.output_dataset.expanduser().resolve()
    partial = output.with_name(output.name + ".partial")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    for required in (cleaned, source, raw_root):
        if not required.is_dir():
            raise FileNotFoundError(required)
    for source_root in (cleaned, source, raw_root):
        if _paths_overlap(output, source_root) or _paths_overlap(partial, source_root):
            raise ValueError(
                f"output paths must not overlap an input: {source_root}"
            )
    if output.exists():
        raise FileExistsError(f"final output already exists: {output}")

    baseline_manifest = _dataset_identity_manifest(cleaned)
    if not partial.exists():
        partial.parent.mkdir(parents=True, exist_ok=True)
        print(f"Copying baseline dataset to {partial}", flush=True)
        shutil.copytree(cleaned, partial, copy_function=shutil.copy2)
    else:
        print(f"Resuming partial dataset at {partial}", flush=True)
    validate_partial_baseline(cleaned, partial, baseline_manifest)

    thresholds = AlignmentThresholds()
    jobs, alignment_reports = build_episode_jobs(
        cleaned, source, raw_root, partial, thresholds
    )
    work_spec = _build_work_spec(
        cleaned,
        source,
        raw_root,
        baseline_manifest,
        thresholds,
        jobs,
        alignment_reports,
    )
    work = _prepare_work_manifest(partial, work_spec)
    jobs = _add_trusted_video_hashes(jobs, partial, work)
    print(
        f"All {len(jobs)} episode alignments passed; starting wrist rendering "
        f"with {args.workers} workers.",
        flush=True,
    )

    render_reports: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=context)
    futures = {}
    try:
        futures = {
            pool.submit(_render_episode, asdict(job)): job for job in jobs
        }
        completed = 0
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            render_reports.append(result)
            _record_completed_videos(partial, work, result)
            completed += 1
            print(
                f"[render {completed:02d}/{len(jobs):02d}] "
                f"output_episode={job.output_episode_index} "
                f"new_cameras={result['rendered_camera_count']}",
                flush=True,
            )
    except BaseException:
        for future in futures:
            future.cancel()
        for process in list(pool._processes.values()):
            if process.is_alive():
                process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)

    _validate_job_dependencies(jobs)
    if _file_fingerprint(source / "meta" / "info.json") != work_spec["source_info"]:
        raise RuntimeError("source info.json changed while rendering")
    if _dataset_identity_manifest(cleaned) != baseline_manifest:
        raise RuntimeError("baseline dataset changed while backfill was running")
    validate_partial_baseline(cleaned, partial, baseline_manifest)

    generated_metadata = (
        Path("meta/cameras.json"),
        Path("meta/camera_backfill_report.json"),
    )
    try:
        report = _update_metadata(
            partial,
            cleaned,
            source,
            raw_root,
            thresholds,
            alignment_reports,
            render_reports,
            baseline_manifest,
        )
        _validate_preserved_baseline(partial, baseline_manifest)

        print("Running full LeRobot/parquet/video/statistics validation.", flush=True)
        validation = verify_dataset(
            partial,
            expected_episodes=int(report["result"]["total_episodes"]),
            expected_frames=int(report["result"]["total_frames"]),
            pose_modes=DEFAULT_POSE_MODES,
        )
        report["validation"] = validation
        report["validated_at"] = datetime.now().astimezone().isoformat()
        _json_write(partial / "meta" / "camera_backfill_report.json", report)
    except BaseException:
        # Preserve a resumable baseline-shaped partial if final validation
        # fails after the metadata has been expanded to three cameras.
        for relative in (Path("meta/info.json"), Path("meta/modality.json")):
            shutil.copy2(cleaned / relative, partial / relative)
        for relative in generated_metadata:
            baseline_path = cleaned / relative
            partial_path = partial / relative
            if baseline_path.is_file():
                shutil.copy2(baseline_path, partial_path)
            else:
                partial_path.unlink(missing_ok=True)
        raise
    (partial / "meta" / "camera_backfill_work.json").unlink()
    partial.replace(output)
    print(
        f"COMPLETE output={output} episodes={report['result']['total_episodes']} "
        f"frames={report['result']['total_frames']} "
        f"videos={report['result']['total_videos']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
