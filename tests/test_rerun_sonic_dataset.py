from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import cv2
import numpy as np
import pandas as pd
import pytest


FPS = 10.0
FRAME_COUNT = 4
FRAME_WIDTH = 64
FRAME_HEIGHT = 48
CAMERA_KEY = "observation.images.ego_view"
SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "robocasa"
    / "scripts"
    / "rerun_sonic_dataset.py"
)
RIGHT_ARM_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _load_rerun_script() -> ModuleType:
    """Load the standalone CLI without importing robocasa and robosuite."""

    module_name = "rerun_sonic_dataset_under_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    script_directory = str(SCRIPT_PATH.parent)
    sys.path.insert(0, script_directory)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(script_directory)
    return module


rerun_script = _load_rerun_script()


def _state_names() -> list[str]:
    names = [f"joint_{index}" for index in range(43)]
    names[29:36] = RIGHT_ARM_NAMES
    return names


def _write_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )
    assert writer.isOpened()
    try:
        for frame_index in range(FRAME_COUNT):
            frame = np.full(
                (FRAME_HEIGHT, FRAME_WIDTH, 3),
                (20, 30 + frame_index * 20, 200),
                dtype=np.uint8,
            )
            writer.write(frame)
    finally:
        writer.release()


def _metadata() -> tuple[dict, dict]:
    info = {
        "codebase_version": "v2.1",
        "total_episodes": 1,
        "total_frames": FRAME_COUNT,
        "total_videos": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "data_path": (
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            CAMERA_KEY: {
                "dtype": "video",
                "shape": [FRAME_HEIGHT, FRAME_WIDTH, 3],
                "info": {
                    "video.height": FRAME_HEIGHT,
                    "video.width": FRAME_WIDTH,
                    "video.fps": FPS,
                },
            },
            "observation.state": {
                "dtype": "float64",
                "shape": [43],
                "names": _state_names(),
            },
            "action.wbc": {
                "dtype": "float64",
                "shape": [43],
                "names": _state_names(),
            },
            "observation.eef_state": {
                "dtype": "float64",
                "shape": [14],
                "names": [
                    "left_wrist_pos",
                    "left_wrist_abs_quat",
                    "right_wrist_pos",
                    "right_wrist_abs_quat",
                ],
            },
            "action.motion_token": {
                "dtype": "float64",
                "shape": [64],
                "names": "motion_token",
            },
        },
    }
    modality = {
        "state": {
            "right_arm": {"start": 29, "end": 36},
            "right_wrist_pos": {
                "start": 7,
                "end": 10,
                "original_key": "observation.eef_state",
            },
        },
        "action": {
            "motion_token": {
                "start": 0,
                "end": 64,
                "original_key": "action.motion_token",
            }
        },
    }
    return info, modality


def _write_dataset(tmp_path: Path) -> tuple[Path, np.ndarray]:
    dataset_path = tmp_path / "sonic_g1"
    meta_path = dataset_path / "meta"
    meta_path.mkdir(parents=True)
    info, modality = _metadata()
    (meta_path / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta_path / "modality.json").write_text(
        json.dumps(modality), encoding="utf-8"
    )
    episode = {
        "episode_index": 0,
        "tasks": ["Test the Rerun visualization."],
        "length": FRAME_COUNT,
    }
    (meta_path / "episodes.jsonl").write_text(
        json.dumps(episode) + "\n", encoding="utf-8"
    )

    state_rows = []
    command_rows = []
    eef_rows = []
    motion_token_rows = []
    for frame_index in range(FRAME_COUNT):
        state = np.zeros(43, dtype=np.float64)
        state[29:36] = frame_index + np.arange(7) / 10.0
        command = state.copy()
        command[29:36] += 0.05
        eef = np.zeros(14, dtype=np.float64)
        eef[7:10] = [frame_index / 10.0, 0.2, 0.3]
        motion_token = (
            np.arange(64, dtype=np.float64) / 16.0 + frame_index / 16.0
        )
        state_rows.append(state)
        command_rows.append(command)
        eef_rows.append(eef)
        motion_token_rows.append(motion_token)

    parquet_path = dataset_path / "data/chunk-000/episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "observation.state": state_rows,
            "action.wbc": command_rows,
            "observation.eef_state": eef_rows,
            "action.motion_token": motion_token_rows,
            "timestamp": np.arange(FRAME_COUNT, dtype=np.float32) / FPS,
            "frame_index": np.arange(FRAME_COUNT, dtype=np.int64),
            "episode_index": np.zeros(FRAME_COUNT, dtype=np.int64),
            "index": np.arange(FRAME_COUNT, dtype=np.int64),
        }
    ).to_parquet(parquet_path, index=False)
    video_path = (
        dataset_path
        / "videos/chunk-000"
        / CAMERA_KEY
        / "episode_000000.mp4"
    )
    _write_video(video_path)
    return dataset_path, np.stack(motion_token_rows)


def test_resolves_all_motion_token_and_named_g1_indices() -> None:
    info, modality = _metadata()
    names = info["features"]["observation.state"]["names"]
    names[29], names[32] = names[32], names[29]

    spec = rerun_script.resolve_signal_spec(info, modality, [0, 2, 63])

    assert spec.motion_token_key == "action.motion_token"
    assert spec.motion_token_indices == tuple(range(64))
    assert spec.visible_motion_token_indices == (0, 2, 63)
    assert spec.state_indices == (32, 30, 31, 29, 33, 34, 35)
    assert spec.command_indices == tuple(range(29, 36))
    assert spec.right_eef_indices == (7, 8, 9)


def test_rejects_invalid_motion_token_or_eef_schema() -> None:
    info, modality = _metadata()
    with pytest.raises(ValueError, match="latent indices.*64"):
        rerun_script.resolve_signal_spec(info, modality, [64])

    missing_motion_token = json.loads(json.dumps(info))
    del missing_motion_token["features"]["action.motion_token"]
    with pytest.raises(ValueError, match="missing feature 'action.motion_token'"):
        rerun_script.resolve_signal_spec(missing_motion_token, modality, [0])

    invalid_eef = json.loads(json.dumps(modality))
    invalid_eef["state"]["right_wrist_pos"]["end"] = 9
    with pytest.raises(ValueError, match="must contain xyz"):
        rerun_script.resolve_signal_spec(info, invalid_eef, [0])


def test_time_crop_is_inclusive_and_preserves_source_indices() -> None:
    timestamps = np.arange(4, dtype=np.float32) / np.float32(10.0)

    assert rerun_script.select_frame_indices(timestamps, 0.1, 0.2, 1).tolist() == [
        1,
        2,
    ]
    assert rerun_script.select_frame_indices(timestamps, None, None, 2).tolist() == [
        0,
        2,
    ]
    with pytest.raises(ValueError, match="must not exceed"):
        rerun_script.select_frame_indices(timestamps, 0.3, 0.2, 1)
    with pytest.raises(ValueError, match="selects no frames"):
        rerun_script.select_frame_indices(timestamps, 1.0, None, 1)


def test_camera_stream_validation_covers_compact_asset_mode() -> None:
    info, _ = _metadata()
    stream = rerun_script.CameraStreamInfo(
        frames=FRAME_COUNT,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        fps=FPS,
    )

    rerun_script.validate_camera_stream(
        stream,
        info["features"][CAMERA_KEY],
        expected_frames=FRAME_COUNT,
        expected_fps=FPS,
    )
    with pytest.raises(ValueError, match="FPS"):
        rerun_script.validate_camera_stream(
            rerun_script.CameraStreamInfo(
                frames=FRAME_COUNT,
                width=FRAME_WIDTH,
                height=FRAME_HEIGHT,
                fps=FPS / 2,
            ),
            info["features"][CAMERA_KEY],
            expected_frames=FRAME_COUNT,
            expected_fps=FPS,
        )


def test_blueprint_has_requested_four_rows() -> None:
    _, rrb = pytest.importorskip("rerun"), pytest.importorskip("rerun.blueprint")

    blueprint = rerun_script.build_blueprint(rrb, [0, 1, 2])
    views = blueprint.root_container.contents

    assert blueprint.root_container.row_shares == [4.0, 1.2, 2.0, 1.2]
    assert [view.class_identifier for view in views] == [
        "2D",
        "TimeSeries",
        "TimeSeries",
        "TimeSeries",
    ]
    assert [view.origin for view in views] == [
        "/camera",
        "/sonic/motion_token",
        "/g1/right_arm",
        "/g1/eef/right/position",
    ]
    assert "/sonic/motion_token/z02" in views[1].contents
    assert "/sonic/motion_token/z03" not in views[1].contents


def test_default_blueprint_displays_all_motion_token_dimensions() -> None:
    rrb = pytest.importorskip("rerun.blueprint")
    args = rerun_script.parse_args(["/tmp/example_dataset"])

    blueprint = rerun_script.build_blueprint(rrb)
    motion_token_view = blueprint.root_container.contents[1]

    assert args.latent_dims == list(range(64))
    assert motion_token_view.contents == [
        f"/sonic/motion_token/z{index:02d}" for index in range(64)
    ]
    assert "z0-z63" in motion_token_view.name


def test_saves_headless_rrd_with_all_64_motion_token_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rr = pytest.importorskip("rerun")
    dataset_path, motion_token = _write_dataset(tmp_path)
    output_path = tmp_path / "episode_000000.rrd"
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(
        rr,
        "spawn",
        lambda *args, **kwargs: pytest.fail("save-only mode spawned a viewer"),
    )

    status = rerun_script.main(
        [
            str(dataset_path),
            "--episode",
            "0",
            "--start-seconds",
            "0.1",
            "--end-seconds",
            "0.2",
            "--save",
            str(output_path),
            "--log-level",
            "ERROR",
        ]
    )

    assert status == 0
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    manifest = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert manifest["dataset"]["selected_frames"] == 2
    assert manifest["dataset"]["selected_source_indices"] == [1, 2]
    token_metrics = manifest["validation"]["rerun_metrics"]["motion_token"]
    assert token_metrics["shape"] == [2, 64]
    assert len(token_metrics["dimensions"]) == 64
    assert token_metrics["nearest_1_over_16_max_residual"] == 0.0

    recording = rr.dataframe.load_recording(output_path)
    index_names = {str(column) for column in recording.schema().index_columns()}
    assert index_names == {"Index(timeline:timestamp)"}
    component_names = {
        str(column) for column in recording.schema().component_columns()
    }
    assert "Component(/camera/ego_view:Blob)" in component_names
    assert "Component(/sonic/motion_token/z63:Scalar)" in component_names
    assert "Component(/g1/right_arm/elbow/target:Scalar)" in component_names
    assert "Component(/g1/right_arm/elbow/measured:Scalar)" in component_names
    assert "Component(/g1/eef/right/position/z:Scalar)" in component_names

    camera_table = (
        recording.view(index="timestamp", contents="/camera/ego_view")
        .select()
        .read_all()
        .to_pydict()
    )
    assert len(camera_table["timestamp"]) == 2

    frame_table = (
        recording.view(index="timestamp", contents="/metadata/source_frame_index")
        .select()
        .read_all()
        .to_pydict()
    )
    assert frame_table["/metadata/source_frame_index:Scalar"] == [[1.0], [2.0]]

    table = (
        recording.view(index="timestamp", contents="/sonic/motion_token/z63")
        .select()
        .read_all()
        .to_pydict()
    )
    assert len(table["timestamp"]) == 2
    scalar_key = "/sonic/motion_token/z63:Scalar"
    assert table[scalar_key] == [
        [pytest.approx(motion_token[1, 63])],
        [pytest.approx(motion_token[2, 63])],
    ]
