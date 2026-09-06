#!/usr/bin/env python3
"""Visualize one collected SONIC G1 episode in Rerun.

The recording uses the LeRobot ``timestamp`` column as a shared time axis for
the encoded camera stream, all 64 SONIC motion-token dimensions, measured/target G1
right-arm joints, and the right end-effector position.  The default operation
is headless and saves an ``.rrd`` recording; pass ``--spawn`` to open the
native Rerun viewer.

This is an observation replay, not a MuJoCo state replay.  The SONIC exporter
stores images, joint values, actions, and end-effector poses, but not the full
simulator state required to restore contacts and object state exactly.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import metadata
import json
import logging
from pathlib import Path
import subprocess
from types import ModuleType
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd

if __package__:
    from .replay_sonic_dataset import (
        get_parquet_path,
        get_video_keys,
        get_video_path,
        load_dataset_metadata,
        resolve_camera_keys,
        validate_parquet,
    )
else:
    from replay_sonic_dataset import (  # type: ignore[no-redef]
        get_parquet_path,
        get_video_keys,
        get_video_path,
        load_dataset_metadata,
        resolve_camera_keys,
        validate_parquet,
    )


LOGGER = logging.getLogger("rerun_sonic_dataset")
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_CAMERA_KEY = "ego_view"
MOTION_TOKEN_WIDTH = 64
DEFAULT_MOTION_TOKEN_DIMS = tuple(range(MOTION_TOKEN_WIDTH))
DEFAULT_JPEG_QUALITY = 85
RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
AXIS_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class SignalSpec:
    """Resolved feature names and vector indices for the four Rerun rows."""

    motion_token_key: str
    motion_token_indices: tuple[int, ...]
    visible_motion_token_indices: tuple[int, ...]
    state_key: str
    state_indices: tuple[int, ...]
    command_key: str
    command_indices: tuple[int, ...]
    right_arm_names: tuple[str, ...]
    eef_key: str
    right_eef_indices: tuple[int, int, int]


@dataclass(frozen=True)
class PreparedEpisode:
    """Validated, selected episode data ready to send to Rerun."""

    frame: pd.DataFrame
    source_indices: np.ndarray
    frame_indices: np.ndarray
    timestamps_seconds: np.ndarray
    timestamps_ns: np.ndarray
    video_pts_ns: np.ndarray | None
    motion_token: np.ndarray
    right_arm_state: np.ndarray
    right_arm_command: np.ndarray
    right_eef_xyz: np.ndarray


@dataclass(frozen=True)
class EncodedCameraFrames:
    """JPEG camera frames plus decoded-stream validation metadata."""

    blobs: tuple[bytes, ...]
    decoded_frames: int
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class CameraStreamInfo:
    """Container-level metadata reported by OpenCV."""

    frames: int
    width: int
    height: int
    fps: float


def load_rerun() -> tuple[ModuleType, ModuleType]:
    """Import Rerun lazily so metadata-only helpers remain usable without it."""

    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ModuleNotFoundError as error:
        if error.name == "rerun" or str(error.name).startswith("rerun."):
            raise RuntimeError(
                "Rerun is not installed; install rerun-sdk==0.22.1 "
                "or reinstall this project"
            ) from error
        raise
    return rr, rrb


def _feature_width(info: dict[str, Any], key: str) -> int:
    feature = info.get("features", {}).get(key)
    if feature is None:
        raise ValueError(f"dataset metadata is missing feature {key!r}")
    shape = feature.get("shape")
    if not isinstance(shape, list) or len(shape) != 1:
        raise ValueError(f"feature {key!r} must be a one-dimensional vector")
    return int(shape[0])


def _named_indices(
    info: dict[str, Any],
    feature_key: str,
    required_names: Sequence[str],
) -> tuple[int, ...]:
    feature = info.get("features", {}).get(feature_key)
    if feature is None:
        raise ValueError(f"dataset metadata is missing feature {feature_key!r}")
    names = feature.get("names")
    if not isinstance(names, list):
        raise ValueError(f"feature {feature_key!r} has no per-dimension names")
    feature_width = _feature_width(info, feature_key)
    if len(names) != feature_width:
        raise ValueError(
            f"feature {feature_key!r} has {len(names)} names for width "
            f"{feature_width}"
        )
    missing = [name for name in required_names if name not in names]
    if missing:
        raise ValueError(
            f"feature {feature_key!r} is missing right-arm joints: {missing}"
        )
    if len(names) != len(set(names)):
        raise ValueError(f"feature {feature_key!r} contains duplicate names")
    return tuple(names.index(name) for name in required_names)


def resolve_signal_spec(
    info: dict[str, Any],
    modality: dict[str, Any],
    latent_dims: Sequence[int],
) -> SignalSpec:
    """Resolve all signal indices from LeRobot metadata and modality config."""

    if not latent_dims:
        raise ValueError("at least one SONIC latent dimension is required")
    if len(latent_dims) != len(set(latent_dims)):
        raise ValueError("SONIC latent dimensions must be unique")

    action_modality = modality.get("action", {})
    state_modality = modality.get("state", {})
    latent_modality = action_modality.get("motion_token")
    if not isinstance(latent_modality, dict):
        raise ValueError("meta/modality.json is missing action.motion_token")
    latent_key = str(latent_modality.get("original_key", "action.motion_token"))
    latent_start = int(latent_modality.get("start", 0))
    latent_end = int(latent_modality.get("end", _feature_width(info, latent_key)))
    latent_width = _feature_width(info, latent_key)
    if not 0 <= latent_start < latent_end <= latent_width:
        raise ValueError(
            f"invalid {latent_key!r} modality slice [{latent_start}:{latent_end}] "
            f"for width {latent_width}"
        )
    if latent_end - latent_start != MOTION_TOKEN_WIDTH:
        raise ValueError(
            f"{latent_key!r} must contain {MOTION_TOKEN_WIDTH} dimensions, got "
            f"slice [{latent_start}:{latent_end}]"
        )
    visible_motion_token_indices = tuple(
        latent_start + int(dim) for dim in latent_dims
    )
    invalid_dims = [
        int(dim)
        for dim, index in zip(latent_dims, visible_motion_token_indices)
        if dim < 0 or index >= latent_end
    ]
    if invalid_dims:
        raise ValueError(
            f"SONIC latent indices {invalid_dims} are outside "
            f"{latent_key!r}[0:{latent_end - latent_start}]"
        )

    state_key = "observation.state"
    command_key = "action.wbc"
    state_indices = _named_indices(info, state_key, RIGHT_ARM_JOINT_NAMES)
    command_indices = _named_indices(info, command_key, RIGHT_ARM_JOINT_NAMES)

    right_arm_modality = state_modality.get("right_arm")
    if not isinstance(right_arm_modality, dict):
        raise ValueError("meta/modality.json is missing state.right_arm")
    right_arm_range = tuple(
        range(
            int(right_arm_modality.get("start", -1)),
            int(right_arm_modality.get("end", -1)),
        )
    )
    if set(right_arm_range) != set(state_indices):
        raise ValueError(
            "state.right_arm modality slice disagrees with "
            "observation.state joint names"
        )

    right_eef = state_modality.get("right_wrist_pos")
    if not isinstance(right_eef, dict):
        raise ValueError("meta/modality.json is missing state.right_wrist_pos")
    eef_key = str(right_eef.get("original_key", "observation.eef_state"))
    eef_start = int(right_eef.get("start", -1))
    eef_end = int(right_eef.get("end", -1))
    if eef_end - eef_start != 3:
        raise ValueError(
            f"{eef_key!r} right_wrist_pos must contain xyz, got "
            f"slice [{eef_start}:{eef_end}]"
        )
    eef_width = _feature_width(info, eef_key)
    if not 0 <= eef_start < eef_end <= eef_width:
        raise ValueError(
            f"invalid {eef_key!r} right_wrist_pos slice "
            f"[{eef_start}:{eef_end}] for width {eef_width}"
        )

    return SignalSpec(
        motion_token_key=latent_key,
        motion_token_indices=tuple(range(latent_start, latent_end)),
        visible_motion_token_indices=visible_motion_token_indices,
        state_key=state_key,
        state_indices=state_indices,
        command_key=command_key,
        command_indices=command_indices,
        right_arm_names=RIGHT_ARM_JOINT_NAMES,
        eef_key=eef_key,
        right_eef_indices=(eef_start, eef_start + 1, eef_start + 2),
    )


def select_frame_indices(
    timestamps: np.ndarray,
    start_seconds: float | None,
    end_seconds: float | None,
    stride: int,
) -> np.ndarray:
    """Select a closed time interval while preserving source row indices."""

    values = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("timestamp must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("timestamp contains NaN or Inf")
    if values.size > 1 and not np.all(np.diff(values) > 0):
        raise ValueError("timestamp must be strictly increasing")
    if stride < 1:
        raise ValueError("--stride must be at least 1")
    if start_seconds is not None and not np.isfinite(start_seconds):
        raise ValueError("--start-seconds must be finite")
    if end_seconds is not None and not np.isfinite(end_seconds):
        raise ValueError("--end-seconds must be finite")
    if (
        start_seconds is not None
        and end_seconds is not None
        and start_seconds > end_seconds
    ):
        raise ValueError("--start-seconds must not exceed --end-seconds")

    # Dataset timestamps are stored as float32.  Expand user-entered decimal
    # boundaries by one float32 ULP so e.g. 0.2 includes a stored
    # 0.20000000298 sample without including a neighboring 50 Hz frame.
    def boundary_tolerance(boundary: float) -> float:
        return max(
            float(np.spacing(np.float32(abs(boundary)))) * 2.0,
            float(np.finfo(np.float32).eps),
        )

    mask = np.ones(values.shape, dtype=bool)
    if start_seconds is not None:
        mask &= values >= start_seconds - boundary_tolerance(start_seconds)
    if end_seconds is not None:
        mask &= values <= end_seconds + boundary_tolerance(end_seconds)
    selected = np.flatnonzero(mask)[::stride]
    if selected.size == 0:
        raise ValueError("the requested time range selects no frames")
    return selected


def _stack_feature(frame: pd.DataFrame, key: str) -> np.ndarray:
    if key not in frame:
        raise ValueError(f"episode parquet is missing feature {key!r}")
    try:
        values = np.stack(frame[key].to_numpy()).astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"feature {key!r} does not contain uniform vectors") from error
    if values.ndim != 2:
        raise ValueError(f"feature {key!r} must contain vectors")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"feature {key!r} contains NaN or Inf")
    return values


def prepare_episode(
    frame: pd.DataFrame,
    signal_spec: SignalSpec,
    video_pts_ns: Sequence[int] | None,
    start_seconds: float | None,
    end_seconds: float | None,
    stride: int,
) -> PreparedEpisode:
    """Extract synchronized arrays from a validated episode."""

    timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
    source_indices = select_frame_indices(
        timestamps,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        stride=stride,
    )
    selected_video_pts: np.ndarray | None = None
    if video_pts_ns is not None:
        pts = np.asarray(video_pts_ns, dtype=np.int64)
        if pts.ndim != 1 or len(pts) != len(frame):
            raise ValueError(
                f"camera video has {len(pts)} frame timestamps; "
                f"episode parquet has {len(frame)} rows"
            )
        if len(pts) > 1 and not np.all(np.diff(pts) > 0):
            raise ValueError(
                "camera video frame timestamps are not strictly increasing"
            )
        selected_video_pts = pts[source_indices]

    motion_token = _stack_feature(frame, signal_spec.motion_token_key)
    state = _stack_feature(frame, signal_spec.state_key)
    command = _stack_feature(frame, signal_spec.command_key)
    eef = _stack_feature(frame, signal_spec.eef_key)
    frame_indices = frame["frame_index"].to_numpy(dtype=np.int64)[source_indices]
    selected_timestamps = timestamps[source_indices]

    return PreparedEpisode(
        frame=frame.iloc[source_indices].reset_index(drop=True),
        source_indices=source_indices,
        frame_indices=frame_indices,
        timestamps_seconds=selected_timestamps,
        timestamps_ns=np.rint(selected_timestamps * 1_000_000_000).astype(np.int64),
        video_pts_ns=selected_video_pts,
        motion_token=motion_token[
            np.ix_(source_indices, signal_spec.motion_token_indices)
        ],
        right_arm_state=state[np.ix_(source_indices, signal_spec.state_indices)],
        right_arm_command=command[
            np.ix_(source_indices, signal_spec.command_indices)
        ],
        right_eef_xyz=eef[np.ix_(source_indices, signal_spec.right_eef_indices)],
    )


def resolve_camera_mode(requested_mode: str, codec: str | None) -> str:
    """Choose a viewer-compatible camera representation."""

    if requested_mode != "auto":
        return requested_mode
    # Rerun 0.22 only guarantees MP4/AV1 playback.  Existing G1 exports are
    # H.264 and can fail with the host FFmpeg, so encode timestamped JPEGs for
    # every other/unknown codec.  This costs more disk but is portable.
    return "asset-video" if (codec or "").lower() == "av1" else "jpeg"


def encode_camera_frames(
    video_path: Path,
    source_indices: np.ndarray,
    expected_frames: int,
    jpeg_quality: int,
) -> EncodedCameraFrames:
    """Decode a complete stream and JPEG-encode only selected source frames."""

    if not 1 <= jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    selected = {int(index) for index in source_indices}
    encoded: dict[int, bytes] = {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"unable to decode camera video: {video_path}")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded_frames = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if decoded_frames in selected:
                encode_ok, buffer = cv2.imencode(
                    ".jpg",
                    image,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                if not encode_ok:
                    raise RuntimeError(
                        f"unable to JPEG-encode camera frame {decoded_frames}"
                    )
                encoded[decoded_frames] = buffer.tobytes()
            decoded_frames += 1
    finally:
        capture.release()

    if decoded_frames != expected_frames:
        raise ValueError(
            f"camera video has {decoded_frames} decoded frames; "
            f"episode parquet has {expected_frames} rows"
        )
    missing = [int(index) for index in source_indices if int(index) not in encoded]
    if missing:
        raise ValueError(f"camera video is missing selected frames: {missing[:10]}")
    return EncodedCameraFrames(
        blobs=tuple(encoded[int(index)] for index in source_indices),
        decoded_frames=decoded_frames,
        width=width,
        height=height,
        fps=fps,
    )


def probe_camera_stream(video_path: Path) -> CameraStreamInfo:
    """Read camera container metadata without changing the encoded stream."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"unable to open camera video: {video_path}")
    try:
        return CameraStreamInfo(
            frames=int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
            width=int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            height=int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
        )
    finally:
        capture.release()


def validate_camera_stream(
    stream: CameraStreamInfo,
    camera_feature: dict[str, Any],
    expected_frames: int,
    expected_fps: float,
) -> None:
    """Compare video container properties with episode metadata."""

    shape = camera_feature.get("shape")
    if not isinstance(shape, list) or len(shape) < 2:
        raise ValueError("camera feature metadata must declare height and width")
    expected_height, expected_width = [int(value) for value in shape[:2]]
    if stream.frames != expected_frames:
        raise ValueError(
            f"camera video declares {stream.frames} frames; "
            f"episode parquet has {expected_frames} rows"
        )
    if (stream.width, stream.height) != (expected_width, expected_height):
        raise ValueError(
            f"camera video resolution is {stream.width}x{stream.height}; "
            f"metadata declares {expected_width}x{expected_height}"
        )
    if not np.isclose(stream.fps, expected_fps, rtol=0.0, atol=0.01):
        raise ValueError(
            f"camera video FPS is {stream.fps:g}; metadata declares "
            f"{expected_fps:g}"
        )


def _short_joint_name(name: str) -> str:
    result = name
    if result.startswith("right_"):
        result = result[len("right_") :]
    if result.endswith("_joint"):
        result = result[: -len("_joint")]
    return result


def build_blueprint(
    rrb: ModuleType,
    visible_motion_token_indices: Sequence[int] = DEFAULT_MOTION_TOKEN_DIMS,
) -> Any:
    """Build the requested four-row synchronized viewer layout."""

    visible_token_paths = [
        f"/sonic/motion_token/z{index:02d}"
        for index in visible_motion_token_indices
    ]
    if tuple(visible_motion_token_indices) == DEFAULT_MOTION_TOKEN_DIMS:
        visible_token_label = f"z0-z{MOTION_TOKEN_WIDTH - 1}"
    else:
        visible_token_label = ", ".join(
            f"z{index}" for index in visible_motion_token_indices
        )
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial2DView(
                origin="/camera",
                contents="/camera/**",
                name="Camera Video | Unitree G1",
            ),
            rrb.TimeSeriesView(
                origin="/sonic/motion_token",
                contents=visible_token_paths,
                name=(
                    f"SONIC motion_token | {visible_token_label} "
                    "(all 64D stored)"
                ),
            ),
            rrb.TimeSeriesView(
                origin="/g1/right_arm",
                contents="/g1/right_arm/**",
                name="Right Arm Joint | target vs measured",
            ),
            rrb.TimeSeriesView(
                origin="/g1/eef/right/position",
                contents="/g1/eef/right/position/**",
                name="ee xyz | SONIC exporter FK frame",
            ),
            row_shares=[4.0, 1.2, 2.0, 1.2],
            name="SONIC G1 synchronized replay",
        ),
        rrb.TimePanel(expanded=True),
        collapse_panels=True,
    )


def _send_scalar(
    rr: ModuleType,
    entity_path: str,
    prepared: PreparedEpisode,
    values: np.ndarray,
) -> None:
    rr.send_columns(
        entity_path,
        indexes=[rr.TimeNanosColumn("timestamp", prepared.timestamps_ns)],
        columns=rr.Scalar.columns(scalar=values),
    )


def log_episode(
    rr: ModuleType,
    prepared: PreparedEpisode,
    signal_spec: SignalSpec,
    camera_key: str,
    camera_mode: str,
    video: Any | None,
    encoded_camera: EncodedCameraFrames | None,
    manifest: dict[str, Any],
) -> None:
    """Batch-log synchronized video references and scalar series."""

    camera_name = camera_key.rsplit(".", maxsplit=1)[-1]
    camera_entity = f"/camera/{camera_name}"
    camera_indexes = [rr.TimeNanosColumn("timestamp", prepared.timestamps_ns)]
    if camera_mode == "asset-video":
        if video is None or prepared.video_pts_ns is None:
            raise RuntimeError("asset-video camera payload was not prepared")
        rr.log(camera_entity, video, static=True)
        rr.send_columns(
            camera_entity,
            indexes=camera_indexes,
            # Rerun 0.22.1's columns_seconds truncates fractional seconds
            # before conversion. Nanoseconds preserve every 20 ms frame.
            columns=rr.VideoFrameReference.columns_nanoseconds(
                prepared.video_pts_ns
            ),
        )
    else:
        if encoded_camera is None:
            raise RuntimeError("JPEG camera payload was not prepared")
        rr.send_columns(
            camera_entity,
            indexes=camera_indexes,
            columns=rr.EncodedImage.columns(
                blob=encoded_camera.blobs,
                media_type=["image/jpeg"] * len(encoded_camera.blobs),
            ),
        )

    for column, latent_index in enumerate(signal_spec.motion_token_indices):
        _send_scalar(
            rr,
            f"/sonic/motion_token/z{latent_index:02d}",
            prepared,
            prepared.motion_token[:, column],
        )
    _send_scalar(
        rr,
        "/sonic/motion_token/summary/l2_norm",
        prepared,
        np.linalg.norm(prepared.motion_token, axis=1),
    )
    changed_dimensions = np.zeros(len(prepared.motion_token), dtype=np.float64)
    if len(prepared.motion_token) > 1:
        changed_dimensions[1:] = np.count_nonzero(
            np.diff(prepared.motion_token, axis=0), axis=1
        )
    _send_scalar(
        rr,
        "/sonic/motion_token/summary/changed_dimensions",
        prepared,
        changed_dimensions,
    )

    for column, joint_name in enumerate(signal_spec.right_arm_names):
        short_name = _short_joint_name(joint_name)
        joint_root = f"/g1/right_arm/{short_name}"
        _send_scalar(
            rr,
            f"{joint_root}/target",
            prepared,
            prepared.right_arm_command[:, column],
        )
        _send_scalar(
            rr,
            f"{joint_root}/measured",
            prepared,
            prepared.right_arm_state[:, column],
        )

    for column, axis in enumerate(AXIS_NAMES):
        _send_scalar(
            rr,
            f"/g1/eef/right/position/{axis}",
            prepared,
            prepared.right_eef_xyz[:, column],
        )

    _send_scalar(
        rr,
        "/metadata/source_frame_index",
        prepared,
        prepared.frame_indices.astype(np.float64),
    )

    rr.log(
        "/metadata/replay_manifest",
        rr.TextDocument(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            media_type="application/json",
        ),
        static=True,
    )


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _build_metrics(
    prepared: PreparedEpisode,
    signal_spec: SignalSpec,
    fps: float,
) -> dict[str, Any]:
    tracking_error = np.abs(
        prepared.right_arm_command - prepared.right_arm_state
    )
    motion_token_deltas = np.diff(prepared.motion_token, axis=0)
    period = np.diff(prepared.timestamps_seconds)
    dimension_stats = {}
    for column, index in enumerate(signal_spec.motion_token_indices):
        values = prepared.motion_token[:, column]
        deltas = motion_token_deltas[:, column]
        dimension_stats[f"z{index:02d}"] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "nonzero_fraction": float(np.count_nonzero(values) / values.size),
            "changed_transition_fraction": (
                float(np.mean(deltas != 0.0)) if deltas.size else 0.0
            ),
        }
    return {
        "timestamp": {
            "timeline": "timestamp",
            "source_frame_index_entity": "/metadata/source_frame_index",
            "semantics": "exporter logical time; not camera hardware capture time",
            "expected_period_seconds": 1.0 / fps,
            "max_period_error_seconds": (
                float(
                    np.max(
                        np.abs(
                            period
                            - (
                                prepared.frame_indices[1:]
                                - prepared.frame_indices[:-1]
                            )
                            / fps
                        )
                    )
                )
                if period.size
                else 0.0
            ),
        },
        "motion_token": {
            "shape": list(prepared.motion_token.shape),
            "all_zero_rows": int(
                np.count_nonzero(np.all(prepared.motion_token == 0.0, axis=1))
            ),
            "nonzero_fraction": float(
                np.count_nonzero(prepared.motion_token) / prepared.motion_token.size
            ),
            "changed_transition_fraction": (
                float(np.mean(np.any(motion_token_deltas != 0.0, axis=1)))
                if motion_token_deltas.size
                else 0.0
            ),
            "nearest_1_over_16_max_residual": float(
                np.max(
                    np.abs(
                        prepared.motion_token
                        - np.round(prepared.motion_token * 16.0) / 16.0
                    )
                )
            ),
            "dimensions": dimension_stats,
        },
        "right_arm_tracking": {
            _short_joint_name(name): {
                "mean_absolute_error_rad": float(np.mean(tracking_error[:, column])),
                "p95_absolute_error_rad": float(
                    np.quantile(tracking_error[:, column], 0.95)
                ),
                "max_absolute_error_rad": float(np.max(tracking_error[:, column])),
            }
            for column, name in enumerate(signal_spec.right_arm_names)
        },
        "right_eef_xyz": {
            "coordinate_frame": "SONIC exporter Pinocchio FK reference frame",
            "axes": {
                axis: {
                    "min": float(np.min(prepared.right_eef_xyz[:, column])),
                    "max": float(np.max(prepared.right_eef_xyz[:, column])),
                }
                for column, axis in enumerate(AXIS_NAMES)
            },
        },
    }


def _default_rrd_path(dataset_path: Path, episode_index: int) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return (
        Path.cwd()
        / "artifacts"
        / "rerun"
        / dataset_path.name
        / f"episode_{episode_index:06d}_{timestamp}.rrd"
    )


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path | None]:
    """Validate one episode, configure the sink, and write the Rerun recording."""

    dataset_path = args.dataset.expanduser().resolve()
    info, episode_records = load_dataset_metadata(dataset_path)
    episodes_by_index = {
        int(record["episode_index"]): record for record in episode_records
    }
    if args.episode not in episodes_by_index:
        raise ValueError(f"episode {args.episode} is not present in the dataset")
    discarded_indices = {
        int(index) for index in info.get("discarded_episode_indices", [])
    }
    if args.episode in discarded_indices and not args.include_discarded:
        raise ValueError(
            f"episode {args.episode} is marked discarded; "
            "pass --include-discarded to inspect it"
        )

    fps = float(info.get("fps", 0.0))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"dataset FPS must be positive, got {fps!r}")
    episode_record = episodes_by_index[args.episode]
    expected_frames = int(episode_record["length"])
    parquet_path = get_parquet_path(dataset_path, info, args.episode)
    frame, feature_checks, telemetry, parquet_errors = validate_parquet(
        parquet_path=parquet_path,
        info=info,
        episode_index=args.episode,
        expected_frames=expected_frames,
        fps=fps,
    )
    if frame is None or parquet_errors:
        detail = "; ".join(parquet_errors) or "unknown parquet validation error"
        raise ValueError(f"episode parquet is not reliable: {detail}")

    modality_path = dataset_path / "meta" / "modality.json"
    if not modality_path.is_file():
        raise FileNotFoundError(f"missing modality metadata: {modality_path}")
    modality = json.loads(modality_path.read_text(encoding="utf-8"))
    signal_spec = resolve_signal_spec(info, modality, args.latent_dims)

    camera_keys = resolve_camera_keys([args.camera_key], get_video_keys(info))
    camera_key = camera_keys[0]
    video_path = get_video_path(dataset_path, info, args.episode, camera_key)
    if not video_path.is_file():
        raise FileNotFoundError(f"missing camera video: {video_path}")

    rr, rrb = load_rerun()
    camera_feature = info["features"][camera_key]
    camera_info = camera_feature.get("info", {})
    codec_value = camera_info.get("video.codec")
    codec = str(codec_value) if codec_value is not None else None
    camera_mode = resolve_camera_mode(args.camera_mode, codec)
    video: Any | None = None
    video_pts_ns: Sequence[int] | None = None
    if camera_mode == "asset-video":
        video = rr.AssetVideo(path=video_path)
        video_pts_ns = video.read_frame_timestamps_ns()
    camera_stream = probe_camera_stream(video_path)
    validate_camera_stream(
        stream=camera_stream,
        camera_feature=camera_feature,
        expected_frames=len(frame),
        expected_fps=fps,
    )
    prepared = prepare_episode(
        frame=frame,
        signal_spec=signal_spec,
        video_pts_ns=video_pts_ns,
        start_seconds=args.start_seconds,
        end_seconds=args.end_seconds,
        stride=args.stride,
    )
    encoded_camera: EncodedCameraFrames | None = None
    camera_validation: dict[str, Any]
    if camera_mode == "jpeg":
        encoded_camera = encode_camera_frames(
            video_path=video_path,
            source_indices=prepared.source_indices,
            expected_frames=len(frame),
            jpeg_quality=args.jpeg_quality,
        )
        camera_validation = {
            "decoded_frames": encoded_camera.decoded_frames,
            "selected_encoded_frames": len(encoded_camera.blobs),
            "width": encoded_camera.width,
            "height": encoded_camera.height,
            "fps": encoded_camera.fps,
        }
    else:
        assert video_pts_ns is not None
        camera_validation = {
            "video_pts_count": len(video_pts_ns),
            "video_pts_strictly_increasing": True,
            "container_frames": camera_stream.frames,
            "width": camera_stream.width,
            "height": camera_stream.height,
            "fps": camera_stream.fps,
        }

    save_path: Path | None = None
    if not args.spawn and args.connect_tcp is None:
        save_path = (
            args.save.expanduser().resolve()
            if args.save is not None
            else _default_rrd_path(dataset_path, args.episode).resolve()
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = save_path.with_suffix(".json")
        existing_outputs = [
            path for path in (save_path, manifest_path) if path.exists()
        ]
        if existing_outputs and not args.overwrite:
            existing = ", ".join(str(path) for path in existing_outputs)
            raise FileExistsError(
                f"output already exists: {existing}; pass --overwrite to replace it"
            )

    recording_id = args.recording_id or f"{dataset_path.name}_ep{args.episode:06d}"
    blueprint = build_blueprint(
        rrb,
        visible_motion_token_indices=signal_spec.visible_motion_token_indices,
    )
    rr.init(args.application_id, recording_id=recording_id)
    if args.spawn:
        rr.spawn(hide_welcome_screen=True)
    elif args.connect_tcp is not None:
        rr.connect_tcp(args.connect_tcp)
    else:
        assert save_path is not None
        rr.save(save_path)
    rr.send_blueprint(blueprint)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset": {
            "path": str(dataset_path),
            "codebase_version": info.get("codebase_version"),
            "episode_index": args.episode,
            "tasks": list(episode_record.get("tasks", [])),
            "discarded": args.episode in discarded_indices,
            "fps": fps,
            "source_frames": len(frame),
            "selected_frames": len(prepared.frame),
            "selected_source_indices": [
                int(prepared.source_indices[0]),
                int(prepared.source_indices[-1]),
            ],
            "selected_time_seconds": [
                float(prepared.timestamps_seconds[0]),
                float(prepared.timestamps_seconds[-1]),
            ],
            "stride": args.stride,
        },
        "camera": {
            "feature": camera_key,
            "video_path": str(video_path),
            "source_codec": codec,
            "requested_mode": args.camera_mode,
            "resolved_mode": camera_mode,
            "jpeg_quality": args.jpeg_quality if camera_mode == "jpeg" else None,
            "validation": camera_validation,
        },
        "signals": asdict(signal_spec),
        "validation": {
            "feature_checks": {
                key: asdict(check) for key, check in feature_checks.items()
            },
            "source_telemetry": telemetry,
            "rerun_metrics": _build_metrics(prepared, signal_spec, fps),
            "semantic_task_success": "not_available; inspect the replay",
        },
        "reproducibility": {
            "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
            "rerun_sdk": _package_version("rerun-sdk"),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "application_id": args.application_id,
            "recording_id": recording_id,
        },
        "output_rrd": str(save_path) if save_path is not None else None,
    }

    log_episode(
        rr=rr,
        prepared=prepared,
        signal_spec=signal_spec,
        camera_key=camera_key,
        camera_mode=camera_mode,
        video=video,
        encoded_camera=encoded_camera,
        manifest=manifest,
    )
    recording = rr.get_global_data_recording()
    if recording is not None:
        recording.flush(blocking=True)

    if save_path is not None:
        manifest_path = save_path.with_suffix(".json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        LOGGER.info("saved Rerun recording: %s", save_path)
        LOGGER.info("saved replay manifest: %s", manifest_path)
    return manifest, save_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize synchronized G1 camera, all SONIC motion_token dimensions, "
            "right-arm joints, "
            "and right end-effector xyz in Rerun."
        )
    )
    parser.add_argument("dataset", type=Path, help="LeRobot dataset directory")
    parser.add_argument("--episode", type=int, default=0, help="episode index")
    parser.add_argument(
        "--camera-key",
        default=DEFAULT_CAMERA_KEY,
        help="video feature key or short name (default: ego_view)",
    )
    parser.add_argument(
        "--camera-mode",
        choices=("auto", "jpeg", "asset-video"),
        default="auto",
        help=(
            "camera storage: auto uses portable JPEG frames except for AV1; "
            "asset-video is smaller but codec support depends on the viewer"
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help="JPEG quality for jpeg/auto camera mode (default: 85)",
    )
    parser.add_argument(
        "--motion-token-dims",
        "--latent-dims",
        dest="latent_dims",
        type=int,
        nargs="+",
        default=list(DEFAULT_MOTION_TOKEN_DIMS),
        metavar="INDEX",
        help=(
            "motion_token dimensions shown by default (all 64 are validated, "
            "saved, and visible by default)"
        ),
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        help="inclusive start time on the episode timestamp timeline",
    )
    parser.add_argument(
        "--end-seconds",
        type=float,
        help="inclusive end time on the episode timestamp timeline",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="log every Nth selected frame without changing timestamps",
    )
    parser.add_argument(
        "--include-discarded",
        action="store_true",
        help="allow viewing an episode marked discarded",
    )

    sink = parser.add_mutually_exclusive_group()
    sink.add_argument(
        "--save",
        type=Path,
        help="save an .rrd file (default: timestamped path under artifacts/rerun)",
    )
    sink.add_argument(
        "--spawn",
        action="store_true",
        help="open the native Rerun viewer instead of saving an .rrd file",
    )
    sink.add_argument(
        "--connect-tcp",
        metavar="HOST:PORT",
        help="stream to an already-running Rerun viewer",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing --save .rrd and its JSON manifest",
    )
    parser.add_argument(
        "--application-id",
        default="robocasa_sonic_g1_replay",
        help="Rerun application id",
    )
    parser.add_argument("--recording-id", help="override the Rerun recording id")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )
    try:
        run(args)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        LOGGER.error("%s", error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
