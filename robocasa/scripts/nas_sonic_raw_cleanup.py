#!/usr/bin/env python3
"""Clean a backed-up SONIC raw dataset task-by-task through state replay.

The source ``sonic_raw`` tree and its selected backup are read-only inputs. For
each task this worker creates a canonical, reflinked NAS staging directory from
the backup, downloads exactly one task, verifies the transfer, runs
``evaluate_sonic_raw.py`` in report-only mode, moves failed episodes to a
separate NAS quarantine, and atomically publishes the successful episodes.

The job state is fsynced after every phase and guarded by a local file lock, so
an interrupted background run can be resumed with the same immutable config.
This is task-state replay: recorded MuJoCo states are restored and the task's
``_check_success()`` predicate is evaluated. Recorded actions are not stepped
through controller dynamics.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import traceback
from typing import Any


STATE_SCHEMA_VERSION = 1
TASK_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
EPISODE_PATTERN = re.compile(r"^ep_[A-Za-z0-9_.-]+$")
RUN_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_"
    r"(?P<task>[A-Za-z][A-Za-z0-9_]*)_sonic$"
)
HOST_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
REMOTE_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
PHASE_ORDER = {
    "pending": 0,
    "remote_partial_ready": 1,
    "downloaded": 2,
    "transfer_verified": 3,
    "evaluated": 4,
    "placement_planned": 5,
    "placement_complete": 6,
    "published": 7,
    "complete": 8,
}


class WorkflowError(RuntimeError):
    """Raised when a safety or reproducibility invariant is violated."""


@dataclasses.dataclass(frozen=True)
class TaskInventory:
    """Source inventory for one RoboCasa task."""

    name: str
    episode_hdf5: int
    valid_episode_bytes: int
    runs: tuple[str, ...]
    episode_dirs: int
    missing_hdf5: int
    batch_hdf5: int


@dataclasses.dataclass(frozen=True)
class Inventory:
    """Validated source inventory."""

    root: str
    tasks: dict[str, TaskInventory]
    schema_version: int


def validate_task_name(task: str) -> str:
    """Return a safe task identifier or fail closed."""

    if not isinstance(task, str) or TASK_PATTERN.fullmatch(task) is None:
        raise WorkflowError(f"unsafe task name: {task!r}")
    return task


def choose_task_placement(
    task: str,
    free_bytes: int,
    required_bytes: int,
    reserve_bytes: int,
) -> str:
    """Choose one destination for all successful episodes of a task."""

    validate_task_name(task)
    values = {
        "free_bytes": free_bytes,
        "required_bytes": required_bytes,
        "reserve_bytes": reserve_bytes,
    }
    for label, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise WorkflowError(f"invalid capacity {label}: {value!r}")
    return "local" if free_bytes >= required_bytes + reserve_bytes else "nas"


def _safe_file_manifest(directory: Path) -> list[dict[str, Any]]:
    """Return a complete content manifest for one regular episode tree."""

    if not directory.is_dir() or directory.is_symlink():
        raise WorkflowError(f"episode is not a regular directory: {directory}")
    records: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*"), key=lambda item: str(item)):
        if path.is_symlink():
            raise WorkflowError(f"episode contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise WorkflowError(f"episode contains a special file: {path}")
        relative = path.relative_to(directory).as_posix()
        records.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not records or not (directory / "ep_demo.hdf5").is_file():
        raise WorkflowError(f"episode has no regular ep_demo.hdf5: {directory}")
    return records


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_local_merge_root(target_root: Path) -> Path:
    """Validate or create a local dataset root without crossing symlinks."""

    target = target_root.expanduser()
    if not target.is_absolute():
        raise WorkflowError(f"local target root must be absolute: {target}")
    existing = target
    missing: list[Path] = []
    while not existing.exists() and not existing.is_symlink():
        missing.append(existing)
        if existing.parent == existing:
            break
        existing = existing.parent
    if existing.is_symlink() or not existing.is_dir() or existing.resolve() != existing:
        raise WorkflowError(
            f"local target root ancestor is unsafe or a symlink: {existing}"
        )
    for path in reversed(missing):
        path.mkdir(mode=0o755)
    if target.is_symlink() or not target.is_dir() or target.resolve() != target:
        raise WorkflowError(f"local target root is unsafe or a symlink: {target}")
    return target


def _fsync_tree(directory: Path) -> None:
    for path in sorted(directory.rglob("*"), key=lambda item: str(item)):
        if path.is_file() and not path.is_symlink():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for path in sorted(
        (item for item in directory.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def merge_successful_episodes(
    source_task: Path | str,
    target_root: Path | str,
    task: str,
    expected_episodes: Sequence[str],
) -> dict[str, Any]:
    """Atomically merge one task's successful episodes into local sonic_raw.

    All conflicts are detected before the first copy. Existing episodes are
    reused only when their complete file trees and SHA256 digests match.
    Source episodes remain intact until the worker durably records the merge.
    """

    task_name = validate_task_name(task)
    source = Path(source_task).expanduser()
    root_input = Path(target_root).expanduser()
    if source.name != f"{task_name}_sonic":
        raise WorkflowError(
            f"source task name {source.name!r} does not match {task_name!r}"
        )
    if not source.is_absolute():
        source = source.resolve()
    if not source.is_dir() or source.is_symlink() or source.resolve() != source:
        raise WorkflowError(f"source task is unsafe or contains a symlink: {source}")

    names = sorted(set(expected_episodes))
    if len(names) != len(expected_episodes):
        raise WorkflowError("expected episode names must be unique")
    for name in names:
        if EPISODE_PATTERN.fullmatch(name) is None:
            raise WorkflowError(f"unsafe expected episode name: {name!r}")
        if not (source / name).is_dir() or (source / name).is_symlink():
            raise WorkflowError(f"expected episode is missing from source: {name}")

    # Reject overlap before target creation so an unsafe call is mutation-free.
    root_candidate = root_input if root_input.is_absolute() else root_input.resolve()
    source_resolved = source.resolve()
    root_resolved = root_candidate.resolve(strict=False)
    if _path_is_relative_to(source_resolved, root_resolved) or _path_is_relative_to(
        root_resolved, source_resolved
    ):
        raise WorkflowError("source and local target paths overlap")

    root = _validate_local_merge_root(root_candidate)
    target_task = root / f"{task_name}_sonic"
    if target_task.is_symlink():
        raise WorkflowError(f"local target task is a symlink: {target_task}")
    if target_task.exists() and not target_task.is_dir():
        raise WorkflowError(f"local target task is not a directory: {target_task}")

    source_manifests = {
        name: _safe_file_manifest(source / name) for name in names
    }
    reused: list[str] = []
    copied: list[str] = []
    recoverable: list[str] = []
    for name in names:
        destination = target_task / name
        incoming = target_task / f".copying-{name}"
        if destination.is_symlink():
            raise WorkflowError(f"local target episode is a symlink: {destination}")
        if incoming.is_symlink():
            raise WorkflowError(f"local incoming episode is a symlink: {incoming}")
        if destination.exists():
            if incoming.exists():
                raise WorkflowError(
                    f"both local target and incoming episode exist: {name}"
                )
            if not destination.is_dir():
                raise WorkflowError(
                    f"local target episode is not a directory: {destination}"
                )
            if _safe_file_manifest(destination) != source_manifests[name]:
                raise WorkflowError(
                    f"episode checksum conflict for {task_name}/{name}"
                )
            reused.append(name)
        elif incoming.exists():
            if not incoming.is_dir():
                raise WorkflowError(
                    f"local incoming episode is not a directory: {incoming}"
                )
            if _safe_file_manifest(incoming) != source_manifests[name]:
                raise WorkflowError(
                    f"stale local incoming checksum conflict for {name}"
                )
            recoverable.append(name)
        else:
            copied.append(name)

    if target_task.exists():
        allowed = set(names) | {
            f".copying-{name}" for name in recoverable
        }
        unexpected_incoming = sorted(
            item.name
            for item in target_task.iterdir()
            if item.name.startswith(".copying-") and item.name not in allowed
        )
        if unexpected_incoming:
            raise WorkflowError(
                "unexpected local incoming episode(s): "
                + ", ".join(unexpected_incoming)
            )

    if not names:
        return {
            "task": task_name,
            "target_task": str(target_task),
            "expected": 0,
            "copied": [],
            "reused": [],
        }

    target_task.mkdir(mode=0o755, exist_ok=True)
    for name in recoverable + copied:
        source_episode = source / name
        destination = target_task / name
        incoming = target_task / f".copying-{name}"
        if name in copied:
            shutil.copytree(source_episode, incoming, copy_function=shutil.copy2)
        try:
            if _safe_file_manifest(incoming) != source_manifests[name]:
                raise WorkflowError(f"local copy checksum failed for {name}")
            _fsync_tree(incoming)
            os.rename(incoming, destination)
            directory_fd = os.open(target_task, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            if incoming.exists() and not incoming.is_symlink():
                shutil.rmtree(incoming)
            raise

    return {
        "task": task_name,
        "target_task": str(target_task),
        "expected": len(names),
        "copied": sorted(copied + recoverable),
        "reused": reused,
        "episode_manifests": source_manifests,
    }


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkflowError(f"{label} must be a non-negative integer")
    return value


def load_inventory(path: Path | str) -> Inventory:
    """Load and validate a NAS inventory created before backup."""

    inventory_path = Path(path).expanduser()
    try:
        payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"cannot read inventory JSON {inventory_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise WorkflowError("unsupported inventory schema")
    root = payload.get("root")
    if not isinstance(root, str) or not PurePosixPath(root).is_absolute():
        raise WorkflowError("inventory root must be an absolute NAS path")
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, dict) or not raw_tasks:
        raise WorkflowError("inventory tasks must be a non-empty mapping")

    tasks: dict[str, TaskInventory] = {}
    all_runs: set[str] = set()
    for raw_name, raw_task in raw_tasks.items():
        name = validate_task_name(raw_name)
        if not isinstance(raw_task, dict):
            raise WorkflowError(f"inventory task {name} must be an object")
        episode_hdf5 = _nonnegative_int(
            raw_task.get("episode_hdf5"), label=f"{name}.episode_hdf5"
        )
        episode_dirs = _nonnegative_int(
            raw_task.get("episode_dirs"), label=f"{name}.episode_dirs"
        )
        missing_hdf5 = _nonnegative_int(
            raw_task.get("missing_hdf5"), label=f"{name}.missing_hdf5"
        )
        batch_hdf5 = _nonnegative_int(
            raw_task.get("batch_hdf5"), label=f"{name}.batch_hdf5"
        )
        valid_bytes = _nonnegative_int(
            raw_task.get("valid_episode_bytes"),
            label=f"{name}.valid_episode_bytes",
        )
        if episode_dirs != episode_hdf5 + missing_hdf5:
            raise WorkflowError(
                f"{name}: episode_dirs != episode_hdf5 + missing_hdf5"
            )
        if (episode_hdf5 == 0) != (valid_bytes == 0):
            raise WorkflowError(
                f"{name}: episode count and valid byte accounting disagree"
            )
        raw_runs = raw_task.get("runs")
        if not isinstance(raw_runs, list) or not raw_runs:
            raise WorkflowError(f"{name}: runs must be a non-empty list")
        runs: list[str] = []
        for run in raw_runs:
            if not isinstance(run, str):
                raise WorkflowError(f"{name}: run name must be a string")
            match = RUN_PATTERN.fullmatch(run)
            if match is None or match.group("task") != name:
                raise WorkflowError(f"{name}: unsafe or mismatched run {run!r}")
            if run in all_runs:
                raise WorkflowError(f"duplicate run in inventory: {run}")
            all_runs.add(run)
            runs.append(run)
        tasks[name] = TaskInventory(
            name=name,
            episode_hdf5=episode_hdf5,
            valid_episode_bytes=valid_bytes,
            runs=tuple(runs),
            episode_dirs=episode_dirs,
            missing_hdf5=missing_hdf5,
            batch_hdf5=batch_hdf5,
        )

    total = payload.get("total")
    if isinstance(total, dict):
        checks = {
            "episode_dirs": sum(task.episode_dirs for task in tasks.values()),
            "episode_hdf5": sum(task.episode_hdf5 for task in tasks.values()),
            "missing_hdf5": sum(task.missing_hdf5 for task in tasks.values()),
            "batch_hdf5": sum(task.batch_hdf5 for task in tasks.values()),
            "runs": sum(len(task.runs) for task in tasks.values()),
        }
        for key, expected in checks.items():
            if total.get(key) != expected:
                raise WorkflowError(
                    f"inventory total.{key}={total.get(key)!r}, expected {expected}"
                )
    return Inventory(root=root, tasks=tasks, schema_version=1)


def ordered_tasks(
    inventory: Inventory,
    requested_tasks: Sequence[str] = (),
) -> list[TaskInventory]:
    """Return selected tasks in stable smallest-first order."""

    requested = [validate_task_name(task) for task in requested_tasks]
    unknown = sorted(set(requested) - set(inventory.tasks))
    if unknown:
        raise WorkflowError(f"unknown task(s): {', '.join(unknown)}")
    selected = (
        [inventory.tasks[name] for name in set(requested)]
        if requested
        else list(inventory.tasks.values())
    )
    return sorted(
        selected,
        key=lambda task: (task.valid_episode_bytes, task.name),
    )


def _strict_remote_path(value: Any, *, field: str) -> PurePosixPath:
    if not isinstance(value, (str, PurePosixPath)):
        raise WorkflowError(f"{field} must be an absolute path")
    raw = str(value)
    path = PurePosixPath(raw)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or str(path) != raw.rstrip("/")
        or path == PurePosixPath("/")
        or any(
            REMOTE_COMPONENT_PATTERN.fullmatch(part) is None
            for part in path.parts[1:]
        )
    ):
        raise WorkflowError(f"unsafe {field} path: {raw!r}")
    return path


def _is_posix_relative_to(path: PurePosixPath, parent: PurePosixPath) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_remote_config(config: Mapping[str, Any]) -> dict[str, PurePosixPath]:
    """Separate read-only source paths from the only mutable NAS trees."""

    required = ("raw_root", "backup_root", "clean_root", "rejected_root")
    parsed = {
        field: _strict_remote_path(config.get(field), field=field)
        for field in required
    }
    raw = parsed["raw_root"]
    backup = parsed["backup_root"]
    clean = parsed["clean_root"]
    rejected = parsed["rejected_root"]
    if backup.parent != raw.parent or not backup.name.startswith(raw.name + ".backup-"):
        raise WorkflowError(
            "backup_root must be a versioned sibling of raw_root"
        )
    if clean in (raw, backup, rejected):
        raise WorkflowError("clean_root collides with another remote path")
    if rejected in (raw, backup, clean):
        raise WorkflowError("rejected_root collides with another remote path")
    if raw == backup:
        raise WorkflowError("backup_root collides with raw_root")
    for field, destination in (("clean_root", clean), ("rejected_root", rejected)):
        for source in (raw, backup):
            if _is_posix_relative_to(destination, source):
                raise WorkflowError(f"{field} must not be inside {source}")
        if destination.parent != raw.parent:
            raise WorkflowError(f"{field} must be a sibling of raw_root")
    if _is_posix_relative_to(clean, rejected) or _is_posix_relative_to(rejected, clean):
        raise WorkflowError("clean_root and rejected_root must be disjoint")
    return parsed


def validate_remote_mutation_target(
    target: PurePosixPath | str,
    config: Mapping[str, Any],
) -> PurePosixPath:
    """Allow writes only at or beneath the clean and rejected roots."""

    parsed = validate_remote_config(config)
    path = _strict_remote_path(target, field="mutation target")
    if not any(
        _is_posix_relative_to(path, parsed[field])
        for field in ("clean_root", "rejected_root")
    ):
        raise WorkflowError(f"remote mutation target is outside output trees: {path}")
    return path


def validate_remote_rsync_path(value: str) -> str:
    """Validate the fixed unprivileged NAS rsync runtime command."""

    try:
        tokens = shlex.split(value)
    except ValueError as exc:
        raise WorkflowError(f"invalid remote rsync path: {exc}") from exc
    if len(tokens) != 4 or tokens[1] != "--library-path":
        raise WorkflowError(
            "remote rsync path must be: loader --library-path DIR rsync"
        )
    loader = _strict_remote_path(tokens[0], field="remote rsync loader")
    library_dir = _strict_remote_path(
        tokens[2], field="remote rsync library directory"
    )
    binary = _strict_remote_path(tokens[3], field="remote rsync binary")
    if (
        loader.parent != library_dir
        or binary.parent != library_dir
        or not loader.name.startswith("ld-linux-")
        or binary.name != "rsync"
    ):
        raise WorkflowError("remote rsync runtime paths are inconsistent")
    return shlex.join([str(loader), "--library-path", str(library_dir), str(binary)])


def atomic_write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Durably replace a JSON file without damaging a previous version."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def build_initial_state(
    config: Mapping[str, Any], inventory: Inventory
) -> dict[str, Any]:
    """Create the initial auditable state, including zero-valid tasks."""

    now = dt.datetime.now().astimezone().isoformat()
    minimum_episodes = config.get("minimum_episodes", 0)
    if (
        isinstance(minimum_episodes, bool)
        or not isinstance(minimum_episodes, int)
        or minimum_episodes < 0
    ):
        raise WorkflowError("minimum_episodes must be a non-negative integer")
    tasks: dict[str, dict[str, Any]] = {}
    for task in ordered_tasks(inventory):
        if task.episode_hdf5 == 0:
            tasks[task.name] = {
                "phase": "complete",
                "status": "zero_valid",
                "source_episode_count": 0,
                "source_valid_bytes": 0,
                "updated_at": now,
            }
        elif task.episode_hdf5 < minimum_episodes:
            tasks[task.name] = {
                "phase": "complete",
                "status": "below_minimum",
                "source_episode_count": task.episode_hdf5,
                "source_valid_bytes": task.valid_episode_bytes,
                "minimum_episodes": minimum_episodes,
                "updated_at": now,
            }
        else:
            tasks[task.name] = {
                "phase": "pending",
                "status": "pending",
                "source_episode_count": task.episode_hdf5,
                "source_valid_bytes": task.valid_episode_bytes,
                "updated_at": now,
            }
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "job_id": config.get("job_id"),
        "created_at": now,
        "updated_at": now,
        "status": "pending",
        "config": dict(config),
        "tasks": tasks,
    }


def load_job_state(
    path: Path | str, expected_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Load an existing state only if its immutable config still matches."""

    state_path = Path(path)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"cannot load job state JSON {state_path}: {exc}") from exc
    if not isinstance(state, dict) or state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise WorkflowError("job state schema is invalid")
    if state.get("config") != dict(expected_config):
        raise WorkflowError("job state config differs from the requested config")
    if not isinstance(state.get("tasks"), dict):
        raise WorkflowError("job state tasks are invalid")
    return state


def validate_evaluation_report(
    report: Mapping[str, Any],
    *,
    expected_task: str,
    expected_total: int,
    expected_episodes: Sequence[str] = (),
) -> dict[str, int]:
    """Validate a complete report before any remote episode is classified."""

    task = validate_task_name(expected_task)
    if report.get("status") != "complete":
        raise WorkflowError("evaluation report is not complete")
    cleanup = report.get("cleanup")
    if not isinstance(cleanup, Mapping) or cleanup.get("requested") is not False:
        raise WorkflowError("evaluation cleanup must be report-only")
    if cleanup.get("status") != "not_started":
        raise WorkflowError("evaluation cleanup status must be not_started")
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise WorkflowError("evaluation summary is missing")
    required = ("total", "passed", "failed", "error", "missing_hdf5")
    normalized: dict[str, int] = {}
    for key in required:
        normalized[key] = _nonnegative_int(
            summary.get(key), label=f"evaluation summary.{key}"
        )
    normalized["invalid"] = _nonnegative_int(
        summary.get("invalid", 0), label="evaluation summary.invalid"
    )
    if normalized["total"] != expected_total:
        raise WorkflowError(
            f"evaluation total {normalized['total']} != expected {expected_total}"
        )
    if normalized["error"]:
        raise WorkflowError("evaluation report contains error results")
    if normalized["missing_hdf5"]:
        raise WorkflowError("evaluation report contains missing_hdf5 results")
    if (
        normalized["passed"]
        + normalized["failed"]
        + normalized["invalid"]
        != normalized["total"]
    ):
        raise WorkflowError(
            "evaluation summary passed/failed/invalid arithmetic is invalid"
        )

    results = report.get("results")
    if not isinstance(results, list) or len(results) != expected_total:
        raise WorkflowError("evaluation results length does not match total")
    observed: list[str] = []
    counted = {"passed": 0, "failed": 0, "invalid": 0}
    for result in results:
        if not isinstance(result, Mapping):
            raise WorkflowError("evaluation result must be an object")
        if result.get("task") != task:
            raise WorkflowError("evaluation result task does not match expected task")
        episode = result.get("episode")
        if not isinstance(episode, str) or EPISODE_PATTERN.fullmatch(episode) is None:
            raise WorkflowError(f"unsafe evaluation episode: {episode!r}")
        observed.append(episode)
        status = result.get("status")
        if status not in counted:
            raise WorkflowError(f"unsupported evaluation status: {status!r}")
        counted[status] += 1
        if status == "invalid":
            error = result.get("error")
            if not isinstance(error, str) or not error.startswith(
                "EvaluationError:"
            ):
                raise WorkflowError(
                    f"invalid evaluation result is not deterministic for {episode}"
                )
        elif result.get("state_source") != "raw_npz/true_terminal_state":
            raise WorkflowError(
                f"evaluation state_source is unsafe for {episode}: "
                f"{result.get('state_source')!r}"
            )
    if len(set(observed)) != len(observed):
        raise WorkflowError("evaluation episode names are not unique")
    if expected_episodes and set(observed) != set(expected_episodes):
        raise WorkflowError("evaluation episode set differs from remote manifest")
    if any(counted[key] != normalized[key] for key in counted):
        raise WorkflowError("evaluation summary does not match result statuses")
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return dt.datetime.now().astimezone().isoformat()


def _phase_at_least(task_state: Mapping[str, Any], phase: str) -> bool:
    current = task_state.get("phase")
    if current not in PHASE_ORDER:
        raise WorkflowError(f"invalid task phase: {current!r}")
    return PHASE_ORDER[current] >= PHASE_ORDER[phase]


REMOTE_PREPARE_SCRIPT = r'''
import json
import hashlib
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess

p = PAYLOAD
task = p["task"]
ep_re = re.compile(r"^ep_[A-Za-z0-9_.-]+$")
backup = Path(p["backup_root"])
clean_root = Path(p["clean_root"])
partial = clean_root / ("." + task + "_sonic.partial-" + p["job_id"])
final = clean_root / (task + "_sonic")

def fail(message):
    raise RuntimeError(message)

def regular(path):
    return path.is_file() and not path.is_symlink() and stat.S_ISREG(os.lstat(path).st_mode)

if not backup.is_dir() or backup.is_symlink():
    fail("backup is not a regular directory")
if backup.resolve() != backup:
    fail("backup path contains a symlink")
if not clean_root.parent.is_dir() or clean_root.parent.resolve() != clean_root.parent:
    fail("clean root parent is missing or contains a symlink")
backup_inventory = backup / "source-inventory.json"
if not regular(backup_inventory):
    fail("backup does not contain its immutable source inventory")
inventory_digest = hashlib.sha256(backup_inventory.read_bytes()).hexdigest()
if inventory_digest != p["inventory_sha256"]:
    fail("backup source inventory SHA256 differs from the local inventory")
if final.exists() or final.is_symlink():
    fail("final task already exists while state is before publish: " + str(final))
clean_root.mkdir(mode=0o755, parents=True, exist_ok=True)
if clean_root.is_symlink() or clean_root.resolve() != clean_root:
    fail("clean root contains a symlink")

records = []
seen = set()
for run_name in p["runs"]:
    run = backup / run_name
    if run.parent != backup or not run.is_dir() or run.is_symlink():
        fail("invalid source run: " + str(run))
    episodes = run / "episodes"
    if not episodes.is_dir() or episodes.is_symlink():
        continue
    for ep in sorted(episodes.iterdir(), key=lambda item: item.name):
        if not ep_re.fullmatch(ep.name) or not ep.is_dir() or ep.is_symlink():
            continue
        hdf5 = ep / "ep_demo.hdf5"
        if not regular(hdf5):
            continue
        if ep.name in seen:
            fail("episode name collision: " + ep.name)
        seen.add(ep.name)
        children = sorted(ep.iterdir(), key=lambda item: item.name)
        if any(not regular(child) for child in children):
            fail("valid episode contains symlink, special file, or directory: " + str(ep))
        names = {child.name for child in children}
        state_names = sorted(name for name in names if name.startswith("state_") and name.endswith(".npz"))
        if not {"ep_demo.hdf5", "ep_meta.json", "model.xml"}.issubset(names) or len(state_names) != 1:
            fail("valid episode does not have the required raw files: " + str(ep))
        files = [{"name": child.name, "size": child.stat().st_size} for child in children]
        records.append({
            "episode": ep.name,
            "source": str(ep.relative_to(backup)),
            "bytes": sum(item["size"] for item in files),
            "files": files,
        })
records.sort(key=lambda item: item["episode"])
if len(records) != p["expected_count"]:
    fail("source episode count mismatch: %d != %d" % (len(records), p["expected_count"]))
total_bytes = sum(item["bytes"] for item in records)
if total_bytes != p["expected_bytes"]:
    fail("source byte count mismatch: %d != %d" % (total_bytes, p["expected_bytes"]))

partial.mkdir(mode=0o755, exist_ok=True)
if partial.is_symlink():
    fail("partial task is a symlink")

def actual_record(directory):
    if not directory.is_dir() or directory.is_symlink():
        fail("target episode is not a regular directory: " + str(directory))
    children = sorted(directory.iterdir(), key=lambda item: item.name)
    if any(not regular(child) for child in children):
        fail("target episode contains unexpected entry: " + str(directory))
    return [{"name": child.name, "size": child.stat().st_size} for child in children]

expected_names = {record["episode"] for record in records}
for record in records:
    source = backup / record["source"]
    target = partial / record["episode"]
    temporary = partial / (".copying-" + record["episode"] + "-" + p["job_id"])
    if temporary.exists() or temporary.is_symlink():
        if temporary.is_symlink() or not temporary.is_dir():
            fail("unsafe interrupted copy path: " + str(temporary))
        cleanup_entries = []
        for root, dirs, files in os.walk(temporary, followlinks=False):
            root_path = Path(root)
            if root_path.is_symlink():
                fail("interrupted copy contains a symlink: " + str(root_path))
            cleanup_entries.append((root_path, True))
            for name in dirs:
                path = root_path / name
                mode = os.lstat(path).st_mode
                if path.is_symlink() or not stat.S_ISDIR(mode):
                    fail("interrupted copy contains an unsafe directory: " + str(path))
            for name in files:
                path = root_path / name
                mode = os.lstat(path).st_mode
                if path.is_symlink() or not stat.S_ISREG(mode):
                    fail("interrupted copy contains an unsafe file: " + str(path))
                cleanup_entries.append((path, False))
        for path, is_directory in cleanup_entries:
            os.chmod(path, 0o755 if is_directory else 0o644)
        shutil.rmtree(temporary)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or actual_record(target) != record["files"]:
            fail("existing partial episode differs from source: " + str(target))
        continue
    subprocess.run(
        ["/usr/bin/cp", "-a", "--reflink=always", "--no-target-directory", str(source), str(temporary)],
        check=True,
    )
    if actual_record(temporary) != record["files"]:
        fail("reflink copy verification failed: " + str(temporary))
    os.rename(temporary, target)

actual_names = {item.name for item in partial.iterdir() if item.is_dir() and not item.is_symlink()}
unexpected = [item.name for item in partial.iterdir() if item.name not in expected_names]
if actual_names != expected_names or unexpected:
    fail("partial task contains missing or unexpected entries")
print("RESULT_JSON=" + json.dumps({
    "task": task,
    "partial": str(partial),
    "final": str(final),
    "episode_count": len(records),
    "total_bytes": total_bytes,
    "episodes": records,
}, sort_keys=True))
'''


REMOTE_CLASSIFY_SCRIPT = r'''
import json
import os
from pathlib import Path
import re
import shutil
import stat

p = PAYLOAD
task = p["task"]
ep_re = re.compile(r"^ep_[A-Za-z0-9_.-]+$")
clean_root = Path(p["clean_root"])
rejected_root = Path(p["rejected_root"])
partial = clean_root / ("." + task + "_sonic.partial-" + p["job_id"])
final = clean_root / (task + "_sonic")
rejected_partial = rejected_root / ("." + task + "_sonic.partial-" + p["job_id"])
rejected_final = rejected_root / (task + "_sonic")
localized = clean_root / (".localized-" + task + "_sonic-" + p["job_id"])
marker = clean_root / ("." + task + "_sonic.classification-" + p["job_id"] + ".json")
nas_passed = set(p.get("nas_passed", p.get("passed", [])))
local_passed = set(p.get("local_passed", []))
passed = nas_passed | local_passed
failed = set(p["failed"])
expected = passed | failed

def fail(message):
    raise RuntimeError(message)

def episode_names(root):
    if not root.exists():
        return set()
    if not root.is_dir() or root.is_symlink():
        fail("task path is not a regular directory: " + str(root))
    entries = list(root.iterdir())
    names = {item.name for item in entries if item.is_dir() and not item.is_symlink() and ep_re.fullmatch(item.name)}
    if len(names) != len(entries):
        fail("task path has an unexpected entry: " + str(root))
    return names

def make_read_only(root):
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        if current_path.is_symlink():
            fail("published task contains a symlink: " + str(current_path))
        for name in files:
            path = Path(current) / name
            if path.is_symlink() or not stat.S_ISREG(os.lstat(path).st_mode):
                fail("published task contains an unsafe file: " + str(path))
            os.chmod(path, path.stat().st_mode & ~0o222)
        for name in dirs:
            path = Path(current) / name
            if path.is_symlink() or not stat.S_ISDIR(os.lstat(path).st_mode):
                fail("published task contains an unsafe directory: " + str(path))
            os.chmod(path, path.stat().st_mode & ~0o222)
    os.chmod(root, root.stat().st_mode & ~0o222)

def safely_remove_tree(root):
    if not root.exists():
        return
    if not root.is_dir() or root.is_symlink():
        fail("localized task is not a regular directory: " + str(root))
    directories = []
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        if current_path.is_symlink():
            fail("localized task contains a symlink: " + str(current_path))
        directories.append(current_path)
        for name in dirs:
            path = current_path / name
            if path.is_symlink() or not stat.S_ISDIR(os.lstat(path).st_mode):
                fail("localized task contains an unsafe directory: " + str(path))
        for name in files:
            path = current_path / name
            if path.is_symlink() or not stat.S_ISREG(os.lstat(path).st_mode):
                fail("localized task contains an unsafe file: " + str(path))
    for directory in directories:
        os.chmod(directory, directory.stat().st_mode | 0o200)
    shutil.rmtree(root)

def write_marker(localized_pruned):
    payload = {
        "task": task,
        "job_id": p["job_id"],
        "nas_passed": sorted(nas_passed),
        "local_passed": sorted(local_passed),
        "failed": sorted(failed),
        "localized_pruned": bool(localized_pruned),
    }
    temporary = marker.with_name("." + marker.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        if temporary.is_symlink() or not temporary.is_file():
            fail("unsafe classification marker temporary")
        temporary.unlink()
    with temporary.open("x") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(temporary, marker)
    descriptor = os.open(clean_root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def verify_marker():
    if marker.is_symlink() or not marker.is_file():
        fail("classification marker is unsafe")
    try:
        value = json.loads(marker.read_text())
    except Exception as exc:
        fail("classification marker is invalid: " + str(exc))
    expected_value = {
        "task": task,
        "job_id": p["job_id"],
        "nas_passed": sorted(nas_passed),
        "local_passed": sorted(local_passed),
        "failed": sorted(failed),
    }
    for key, expected_item in expected_value.items():
        if value.get(key) != expected_item:
            fail("classification marker differs at " + key)

if (
    passed & failed
    or nas_passed & local_passed
    or any(not ep_re.fullmatch(name) for name in expected)
):
    fail("invalid passed/failed episode sets")
if (
    not clean_root.is_dir()
    or clean_root.is_symlink()
    or clean_root.resolve() != clean_root
):
    fail("clean root is missing or contains a symlink")
if (
    not rejected_root.parent.is_dir()
    or rejected_root.parent.resolve() != rejected_root.parent
):
    fail("rejected root parent is missing or contains a symlink")
if rejected_root.exists() and (
    not rejected_root.is_dir()
    or rejected_root.is_symlink()
    or rejected_root.resolve() != rejected_root
):
    fail("rejected root contains a symlink or is not a directory")
if rejected_final.is_symlink() or rejected_partial.is_symlink():
    fail("rejected task path is a symlink")
if final.is_symlink() or partial.is_symlink():
    fail("clean final or partial task path is a symlink")
if localized.is_symlink() or marker.is_symlink():
    fail("localized task or marker is a symlink")
if rejected_final.exists() and rejected_partial.exists():
    fail("both rejected final and partial exist")

if marker.exists():
    verify_marker()
    if nas_passed:
        if partial.exists() or episode_names(final) != nas_passed:
            fail("published NAS clean task differs from marker")
        make_read_only(final)
    elif final.exists():
        fail("unexpected NAS clean final task for local placement")
    elif partial.exists():
        if episode_names(partial):
            fail("local-placement clean partial is not empty")
        partial.rmdir()
    if failed:
        if episode_names(rejected_final) != failed:
            fail("published rejected task differs from marker")
        make_read_only(rejected_final)
    elif rejected_final.exists() or rejected_partial.exists():
        fail("unexpected rejected task for zero failures")
    if localized.exists():
        if not episode_names(localized).issubset(local_passed):
            fail("localized episode set contains entries outside marker")
        safely_remove_tree(localized)
    write_marker(True)
    print("RESULT_JSON=" + json.dumps({
        "published": True,
        "passed": len(passed),
        "local_passed": len(local_passed),
        "nas_passed": len(nas_passed),
        "failed": len(failed),
    }, sort_keys=True))
    raise SystemExit(0)

if final.exists() and partial.exists():
    fail("both clean final and partial exist")
if final.exists() and episode_names(final) != nas_passed:
    fail("published clean task does not match NAS passed set")
if not final.exists() and not partial.exists():
    fail("neither clean partial nor clean final exists before marker")

partial_names = episode_names(partial)
final_names = episode_names(final)
rejected_names = episode_names(
    rejected_final if rejected_final.exists() else rejected_partial
)
localized_names = episode_names(localized)
locations = (partial_names, final_names, rejected_names, localized_names)
if set().union(*locations) != expected:
    fail("classification locations do not cover the evaluation set")
for index, left in enumerate(locations):
    for right in locations[index + 1:]:
        if left & right:
            fail("ambiguous episode exists in multiple classification locations")
if not final_names.issubset(nas_passed):
    fail("clean final contains a non-NAS episode")
if not rejected_names.issubset(failed):
    fail("rejected task contains an unexpected episode")
if not localized_names.issubset(local_passed):
    fail("localized task contains an unexpected episode")

if failed:
    rejected_root.mkdir(mode=0o755, parents=True, exist_ok=True)
    if rejected_root.is_symlink() or rejected_root.resolve() != rejected_root:
        fail("rejected root contains a symlink")
    destination_root = rejected_final if rejected_final.exists() else rejected_partial
    destination_root.mkdir(mode=0o755, exist_ok=True)
    for name in sorted(failed):
        source = partial / name
        target = destination_root / name
        source_exists = source.is_dir() and not source.is_symlink()
        target_exists = target.is_dir() and not target.is_symlink()
        if source_exists and not target_exists:
            # Synology's Btrfs permission layer requires the directory being
            # moved to be owner-writable, even though both parents are
            # writable. Backup-derived episode directories are intentionally
            # 0555, so grant only the output copy owner write permission for
            # the atomic rename; make_read_only() removes it after publish.
            os.chmod(source, source.stat().st_mode | 0o200)
            os.rename(source, target)
        elif not source_exists and target_exists:
            pass
        else:
            fail("ambiguous classification state for " + name)
    if episode_names(destination_root) != failed:
        fail("rejected episode set mismatch")
    if destination_root == rejected_partial:
        os.rename(rejected_partial, rejected_final)
    make_read_only(rejected_final)
elif rejected_final.exists() or rejected_partial.exists():
    fail("unexpected rejected task for zero failures")

if local_passed:
    localized.mkdir(mode=0o755, exist_ok=True)
    for name in sorted(local_passed):
        source = partial / name
        target = localized / name
        source_exists = source.is_dir() and not source.is_symlink()
        target_exists = target.is_dir() and not target.is_symlink()
        if source_exists and not target_exists:
            os.chmod(source, source.stat().st_mode | 0o200)
            os.rename(source, target)
        elif not source_exists and target_exists:
            pass
        else:
            fail("ambiguous localized classification state for " + name)
    if episode_names(localized) != local_passed:
        fail("localized episode set mismatch")
elif localized.exists():
    fail("unexpected localized task for NAS placement")

marker_written = False
if final.exists():
    if episode_names(final) != nas_passed:
        fail("clean final episode set mismatch")
elif nas_passed:
    if episode_names(partial) != nas_passed:
        fail("clean partial episode set mismatch")
    os.rename(partial, final)
    make_read_only(final)
else:
    if episode_names(partial):
        fail("clean partial is not empty for local placement")
    write_marker(False)
    marker_written = True
    partial.rmdir()

if not marker_written:
    write_marker(False)
if localized.exists():
    safely_remove_tree(localized)
write_marker(True)
print("RESULT_JSON=" + json.dumps({
    "published": True,
    "passed": len(passed),
    "local_passed": len(local_passed),
    "nas_passed": len(nas_passed),
    "failed": len(failed),
}, sort_keys=True))
'''


class NasCleanupWorker:
    """Resumable, serial task worker."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        inventory: Inventory,
        config: dict[str, Any],
        state_path: Path,
        state: dict[str, Any],
        logger: logging.Logger,
    ) -> None:
        self.args = args
        self.inventory = inventory
        self.config = config
        self.remote = validate_remote_config(config)
        self.state_path = state_path
        self.state = state
        self.logger = logger
        self.log_root = Path(config["log_root"])
        self.local_staging_root = Path(config["local_staging_root"])
        local_dataset_root = config.get("local_dataset_root")
        self.local_dataset_root = (
            Path(local_dataset_root) if local_dataset_root is not None else None
        )

    def save_state(self) -> None:
        self.state["updated_at"] = _utc_now()
        counts: dict[str, int] = {}
        for item in self.state["tasks"].values():
            status = str(item.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
        self.state["summary"] = counts
        atomic_write_json(self.state_path, self.state)

    def update_task(self, name: str, **updates: Any) -> None:
        task_state = self.state["tasks"][name]
        task_state.update(updates)
        task_state["updated_at"] = _utc_now()
        self.save_state()

    def remote_python(
        self, script: str, payload: Mapping[str, Any], *, log_path: Path
    ) -> dict[str, Any]:
        encoded = base64.b64encode(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).decode("ascii")
        source = (
            "import base64, json\n"
            f"PAYLOAD=json.loads(base64.b64decode({encoded!r}))\n"
            + script
        )
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            self.config["ssh_host"],
            "/usr/bin/python3 -",
        ]
        self.logger.info("remote operation started: %s", log_path.stem)
        result = subprocess.run(
            command,
            input=source,
            text=True,
            capture_output=True,
            check=False,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "$ " + shlex.join(command[:-1] + ["/usr/bin/python3 -"]) + "\n"
            + result.stdout
            + result.stderr,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise WorkflowError(
                f"remote operation failed ({result.returncode}); see {log_path}"
            )
        result_lines = [
            line.removeprefix("RESULT_JSON=")
            for line in result.stdout.splitlines()
            if line.startswith("RESULT_JSON=")
        ]
        if len(result_lines) != 1:
            raise WorkflowError(f"remote operation returned no unique result; see {log_path}")
        try:
            return json.loads(result_lines[0])
        except json.JSONDecodeError as exc:
            raise WorkflowError(f"invalid remote JSON result; see {log_path}") from exc

    def _rsync_command(self, remote_path: str, local_path: Path) -> list[str]:
        ssh_transport = shlex.join(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "PasswordAuthentication=no",
                "-o",
                "ConnectTimeout=20",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=3",
            ]
        )
        return [
            "rsync",
            "-a",
            "--no-owner",
            "--no-group",
            "--partial",
            "--append-verify",
            "--timeout=120",
            "--chmod=Du+rwx,Fu+rw",
            "--info=stats2,progress2",
            "-e",
            ssh_transport,
            f"--rsync-path={self.config['remote_rsync_path']}",
            f"{self.config['ssh_host']}:{remote_path.rstrip('/')}/",
            str(local_path) + "/",
        ]

    def run_logged_command(
        self, command: Sequence[str], *, log_path: Path, cwd: Path | None = None
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info("command started: %s", log_path.stem)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"\n[{_utc_now()}] $ {shlex.join(command)}\n")
            stream.flush()
            result = subprocess.run(
                list(command),
                cwd=cwd,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            raise WorkflowError(
                f"command failed ({result.returncode}); see {log_path}"
            )
        self.logger.info("command complete: %s", log_path.stem)

    def prepare_remote_task(
        self, task: TaskInventory, task_log: Path
    ) -> dict[str, Any]:
        partial = self.remote["clean_root"] / (
            f".{task.name}_sonic.partial-{self.config['job_id']}"
        )
        validate_remote_mutation_target(partial, self.remote)
        payload = {
            "task": task.name,
            "runs": list(task.runs),
            "expected_count": task.episode_hdf5,
            "expected_bytes": task.valid_episode_bytes,
            "backup_root": str(self.remote["backup_root"]),
            "clean_root": str(self.remote["clean_root"]),
            "job_id": self.config["job_id"],
            "inventory_sha256": self.config["inventory_sha256"],
        }
        return self.remote_python(
            REMOTE_PREPARE_SCRIPT,
            payload,
            log_path=task_log / "remote-prepare.log",
        )

    @staticmethod
    def load_manifest(
        path: Path,
        *,
        task: TaskInventory,
        expected_partial: str,
        expected_final: str,
    ) -> dict[str, Any]:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkflowError(f"cannot read task manifest {path}: {exc}") from exc
        if not isinstance(manifest, dict) or not isinstance(manifest.get("episodes"), list):
            raise WorkflowError(f"task manifest is invalid: {path}")
        if manifest.get("task") != task.name:
            raise WorkflowError(f"task manifest task is invalid: {path}")
        if manifest.get("partial") != expected_partial:
            raise WorkflowError(f"task manifest partial path is invalid: {path}")
        if manifest.get("final") != expected_final:
            raise WorkflowError(f"task manifest final path is invalid: {path}")
        if manifest.get("episode_count") != task.episode_hdf5:
            raise WorkflowError(f"task manifest episode count is invalid: {path}")
        if manifest.get("total_bytes") != task.valid_episode_bytes:
            raise WorkflowError(f"task manifest byte count is invalid: {path}")
        episodes = manifest["episodes"]
        names = [item.get("episode") for item in episodes if isinstance(item, Mapping)]
        if (
            len(names) != task.episode_hdf5
            or len(set(names)) != len(names)
            or any(
                not isinstance(name, str)
                or EPISODE_PATTERN.fullmatch(name) is None
                for name in names
            )
        ):
            raise WorkflowError(f"task manifest episodes are invalid: {path}")
        calculated_bytes = 0
        for record in episodes:
            if not isinstance(record, Mapping) or not isinstance(record.get("files"), list):
                raise WorkflowError(f"task manifest record is invalid: {path}")
            record_bytes = _nonnegative_int(
                record.get("bytes"), label="manifest episode bytes"
            )
            file_bytes = 0
            for file_record in record["files"]:
                if not isinstance(file_record, Mapping):
                    raise WorkflowError(f"task manifest file is invalid: {path}")
                file_name = file_record.get("name")
                if (
                    not isinstance(file_name, str)
                    or file_name in (".", "..")
                    or "/" in file_name
                    or "\\" in file_name
                ):
                    raise WorkflowError(f"task manifest filename is invalid: {path}")
                file_bytes += _nonnegative_int(
                    file_record.get("size"), label="manifest file size"
                )
            if file_bytes != record_bytes:
                raise WorkflowError(f"task manifest file bytes are invalid: {path}")
            calculated_bytes += record_bytes
        if calculated_bytes != task.valid_episode_bytes:
            raise WorkflowError(f"task manifest total bytes are invalid: {path}")
        return manifest

    @staticmethod
    def verify_local_manifest(local_task: Path, manifest: Mapping[str, Any]) -> None:
        if not local_task.is_dir() or local_task.is_symlink():
            raise WorkflowError(f"local task is not a regular directory: {local_task}")
        expected = {item["episode"]: item for item in manifest["episodes"]}
        entries = list(local_task.iterdir())
        actual_names = {
            item.name
            for item in entries
            if item.is_dir() and not item.is_symlink() and EPISODE_PATTERN.fullmatch(item.name)
        }
        if actual_names != set(expected) or len(entries) != len(expected):
            raise WorkflowError("local episode set differs from remote manifest")
        for name, record in expected.items():
            episode = local_task / name
            children = sorted(episode.iterdir(), key=lambda item: item.name)
            if any(item.is_symlink() or not item.is_file() for item in children):
                raise WorkflowError(f"local episode has unexpected entry: {episode}")
            actual_files = [
                {"name": item.name, "size": item.stat().st_size} for item in children
            ]
            if actual_files != record["files"]:
                raise WorkflowError(f"local episode differs from manifest: {episode}")

    def verify_rsync_checksum(
        self, *, remote_path: str, local_path: Path, log_path: Path
    ) -> None:
        command = self._rsync_command(remote_path, local_path)
        filtered: list[str] = []
        for item in command:
            if item == "--info=stats2,progress2":
                continue
            if (
                item in ("--partial", "--append-verify", "--timeout=120")
                or item.startswith("--chmod=")
            ):
                continue
            filtered.append(item)
        # No mutation occurs: --delete is used only with --dry-run so local
        # extras are included in the parity check.
        # Keep these after ``-a`` so the explicit permission and timestamp
        # exclusions override archive-mode defaults.
        archive_index = filtered.index("-a")
        filtered[archive_index + 1 : archive_index + 1] = [
            "--checksum",
            "--dry-run",
            "--itemize-changes",
            "--delete",
            "--no-perms",
            "--omit-dir-times",
            "--out-format=%i %n%L",
        ]
        result = subprocess.run(filtered, capture_output=True, text=True, check=False)
        log_path.write_text(
            "$ " + shlex.join(filtered) + "\n" + result.stdout + result.stderr,
            encoding="utf-8",
        )
        if result.returncode != 0 or result.stdout.strip():
            raise WorkflowError(f"rsync checksum parity failed; see {log_path}")

    def run_evaluator(
        self,
        *,
        task: TaskInventory,
        local_task: Path,
        manifest: Mapping[str, Any],
        task_log: Path,
    ) -> tuple[dict[str, Any], Path]:
        report_path = task_log / "evaluation.json"
        expected_episodes = [item["episode"] for item in manifest["episodes"]]

        def validate_report(report: Mapping[str, Any]) -> dict[str, int]:
            summary = validate_evaluation_report(
                report,
                expected_task=task.name,
                expected_total=task.episode_hdf5,
                expected_episodes=expected_episodes,
            )
            if report.get("success_mode") != self.config["success_mode"]:
                raise WorkflowError(
                    "evaluation success_mode differs from job config"
                )
            if report.get("seed") != self.config["seed"]:
                raise WorkflowError("evaluation seed differs from job config")
            selection = report.get("selection")
            if not isinstance(selection, Mapping):
                raise WorkflowError("evaluation selection is missing")
            if selection.get("tasks") != [task.name]:
                raise WorkflowError("evaluation selection task differs from job")
            if selection.get("candidate_count") != task.episode_hdf5:
                raise WorkflowError("evaluation candidate count differs from job")
            report_root = report.get("dataset_root")
            if not isinstance(report_root, str):
                raise WorkflowError("evaluation dataset_root is missing")
            if Path(report_root).resolve() != local_task.parent.resolve():
                raise WorkflowError("evaluation dataset_root differs from staging")
            execution = report.get("execution")
            if (
                not isinstance(execution, Mapping)
                or execution.get("episode_process_isolation") is not True
                or execution.get("episode_timeout_seconds")
                != self.config["evaluation_episode_timeout_seconds"]
            ):
                raise WorkflowError("evaluation process isolation is missing")
            child_reports_dir = execution.get("child_reports_dir")
            if not isinstance(child_reports_dir, str):
                raise WorkflowError("evaluation child report directory is missing")
            child_path = Path(child_reports_dir)
            if (
                not child_path.is_dir()
                or child_path.is_symlink()
                or child_path.parent.resolve() != task_log.resolve()
            ):
                raise WorkflowError("evaluation child report directory is unsafe")
            return summary

        if report_path.exists():
            try:
                existing = json.loads(report_path.read_text(encoding="utf-8"))
                validate_report(existing)
                self.logger.info("reusing complete evaluation report for %s", task.name)
                return existing, report_path
            except (OSError, json.JSONDecodeError, WorkflowError):
                archived = report_path.with_name(
                    f"evaluation.incomplete-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
                )
                os.replace(report_path, archived)

        evaluator = Path(self.config["evaluator_path"])
        command = [
            self.config["python_executable"],
            str(evaluator),
            "--dataset",
            str(local_task),
            "--task",
            task.name,
            "--success-mode",
            self.config["success_mode"],
            "--seed",
            str(self.config["seed"]),
            "--isolate-episodes",
            "--episode-timeout-seconds",
            str(self.config["evaluation_episode_timeout_seconds"]),
            "--report",
            str(report_path),
        ]
        self.run_logged_command(
            command,
            log_path=task_log / "evaluation.log",
            cwd=Path(self.config["repo_root"]),
        )
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkflowError(f"cannot read evaluation report {report_path}: {exc}") from exc
        validate_report(report)
        return report, report_path

    def classify_remote(
        self,
        *,
        task: TaskInventory,
        report: Mapping[str, Any],
        task_log: Path,
    ) -> dict[str, Any]:
        passed = sorted(
            result["episode"]
            for result in report["results"]
            if result["status"] == "passed"
        )
        failed = sorted(
            result["episode"]
            for result in report["results"]
            if result["status"] in ("failed", "invalid")
        )
        placement = self.state["tasks"][task.name].get("placement")
        if passed and placement not in ("local", "nas"):
            raise WorkflowError("task placement is missing before classification")
        local_passed = passed if placement == "local" else []
        nas_passed = passed if placement == "nas" else []
        for root, leaf in (
            (self.remote["clean_root"], f"{task.name}_sonic"),
            (self.remote["rejected_root"], f"{task.name}_sonic"),
        ):
            validate_remote_mutation_target(root / leaf, self.remote)
        payload = {
            "task": task.name,
            "local_passed": local_passed,
            "nas_passed": nas_passed,
            "failed": failed,
            "clean_root": str(self.remote["clean_root"]),
            "rejected_root": str(self.remote["rejected_root"]),
            "job_id": self.config["job_id"],
        }
        return self.remote_python(
            REMOTE_CLASSIFY_SCRIPT,
            payload,
            log_path=task_log / "remote-classify.log",
        )

    def place_successful_episodes(
        self,
        *,
        task: TaskInventory,
        task_state: dict[str, Any],
        local_task: Path,
        manifest: Mapping[str, Any],
        report: Mapping[str, Any],
        task_log: Path,
    ) -> None:
        """Persist one task-level local/NAS decision and execute local merge."""

        passed = sorted(
            result["episode"]
            for result in report["results"]
            if result["status"] == "passed"
        )
        records = {
            record["episode"]: record for record in manifest["episodes"]
        }
        required_bytes = sum(records[name]["bytes"] for name in passed)
        if not _phase_at_least(task_state, "placement_planned"):
            local_dataset_root = getattr(self, "local_dataset_root", None)
            reserve_bytes = self.config.get("local_reserve_bytes", 0)
            if local_dataset_root is None:
                placement = "nas"
                free_bytes = None
            else:
                root = _validate_local_merge_root(local_dataset_root)
                free_bytes = shutil.disk_usage(root).free
                placement = choose_task_placement(
                    task.name,
                    free_bytes,
                    required_bytes,
                    reserve_bytes,
                )
            plan = {
                "task": task.name,
                "placement": placement,
                "passed_episodes": passed,
                "required_bytes": required_bytes,
                "free_bytes_at_plan": free_bytes,
                "reserve_bytes": reserve_bytes,
                "local_dataset_root": (
                    str(local_dataset_root)
                    if local_dataset_root is not None
                    else None
                ),
            }
            plan_path = task_log / "placement-plan.json"
            atomic_write_json(plan_path, plan)
            self.update_task(
                task.name,
                phase="placement_planned",
                placement=placement,
                placement_plan=str(plan_path),
                placement_required_bytes=required_bytes,
            )
        else:
            placement = task_state.get("placement")
            if placement not in ("local", "nas"):
                raise WorkflowError("persisted task placement is invalid")

        if _phase_at_least(task_state, "placement_complete"):
            if task_state.get("placement") == "local":
                merge_path_value = task_state.get("local_merge")
                if not isinstance(merge_path_value, str):
                    raise WorkflowError("local merge report path is missing")
                try:
                    summary = json.loads(
                        Path(merge_path_value).read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    raise WorkflowError(
                        f"cannot load local merge report: {exc}"
                    ) from exc
                if (
                    summary.get("task") != task.name
                    or summary.get("expected") != len(passed)
                    or set(summary.get("copied", []))
                    | set(summary.get("reused", []))
                    != set(passed)
                ):
                    raise WorkflowError("local merge report differs from evaluation")
                target_value = summary.get("target_task")
                manifests = summary.get("episode_manifests")
                if not isinstance(target_value, str) or (
                    passed and not isinstance(manifests, dict)
                ):
                    raise WorkflowError("local merge report has no target manifests")
                target = Path(target_value)
                if passed and (
                    not target.is_dir()
                    or target.is_symlink()
                    or target.resolve() != target
                ):
                    raise WorkflowError("local merged task is missing or unsafe")
                for name in passed:
                    if manifests.get(name) != _safe_file_manifest(target / name):
                        raise WorkflowError(
                            f"local merged episode checksum changed: {name}"
                        )
            return
        if placement == "local":
            local_dataset_root = getattr(self, "local_dataset_root", None)
            if local_dataset_root is None:
                raise WorkflowError("local placement has no local dataset root")
            if not local_task.is_dir() or local_task.is_symlink():
                local_task.mkdir(parents=True, exist_ok=True)
                self.run_logged_command(
                    self._rsync_command(str(manifest["partial"]), local_task),
                    log_path=task_log / "placement-redownload.log",
                )
                self.verify_local_manifest(local_task, manifest)
                self.verify_rsync_checksum(
                    remote_path=str(manifest["partial"]),
                    local_path=local_task,
                    log_path=task_log / "placement-redownload-checksum.log",
                )
            summary = merge_successful_episodes(
                local_task,
                local_dataset_root,
                task.name,
                passed,
            )
            merge_path = task_log / "local-merge.json"
            atomic_write_json(merge_path, summary)
            self.update_task(
                task.name,
                phase="placement_complete",
                local_merge=str(merge_path),
                local_copied=len(summary["copied"]),
                local_reused=len(summary["reused"]),
                local_task_target=summary["target_task"],
            )
        else:
            self.update_task(
                task.name,
                phase="placement_complete",
                nas_offloaded=len(passed),
            )

    def safe_remove_local_task(self, task_stage: Path) -> None:
        staging = self.local_staging_root.resolve()
        if task_stage.is_symlink() or task_stage.parent.is_symlink():
            raise WorkflowError(f"local cleanup target is a symlink: {task_stage}")
        resolved = task_stage.resolve()
        try:
            relative = resolved.relative_to(staging)
        except ValueError as exc:
            raise WorkflowError(f"local cleanup target escaped staging: {resolved}") from exc
        if len(relative.parts) != 2 or relative.parts[0] != "pipeline":
            raise WorkflowError(f"unsafe local cleanup target: {resolved}")
        if task_stage.exists():
            shutil.rmtree(task_stage)

    def validate_local_task_paths(
        self, *, task: TaskInventory, task_stage: Path, local_task: Path
    ) -> None:
        """Reject symlinks and unexpected local staging destinations."""

        staging = self.local_staging_root
        if (
            not staging.is_dir()
            or staging.is_symlink()
            or staging.resolve() != staging
        ):
            raise WorkflowError(
                f"local staging root is missing or contains a symlink: {staging}"
            )
        expected_stage = staging / "pipeline" / task.name
        expected_task = expected_stage / "dataset" / f"{task.name}_sonic"
        if task_stage != expected_stage or local_task != expected_task:
            raise WorkflowError("local task paths do not match the selected task")
        for path in (
            staging / "pipeline",
            task_stage,
            task_stage / "dataset",
            local_task,
        ):
            if path.is_symlink():
                raise WorkflowError(f"local staging path is a symlink: {path}")
            if path.exists() and not path.is_dir():
                raise WorkflowError(f"local staging path is not a directory: {path}")

    def process_task(self, task: TaskInventory) -> None:
        name = task.name
        task_state = self.state["tasks"][name]
        if task_state.get("phase") == "complete":
            self.logger.info("skip completed task %s (%s)", name, task_state.get("status"))
            return
        task_log = self.log_root / "tasks" / name / "worker"
        task_log.mkdir(parents=True, exist_ok=True)
        task_stage = self.local_staging_root / "pipeline" / name
        local_task = task_stage / "dataset" / f"{name}_sonic"
        self.validate_local_task_paths(
            task=task,
            task_stage=task_stage,
            local_task=local_task,
        )
        manifest_path = task_log / "manifest.json"
        self.update_task(name, status="running", last_error=None)
        self.logger.info(
            "task %s started: episodes=%d bytes=%d phase=%s",
            name,
            task.episode_hdf5,
            task.valid_episode_bytes,
            task_state["phase"],
        )

        expected_partial = str(
            self.remote["clean_root"]
            / f".{name}_sonic.partial-{self.config['job_id']}"
        )
        expected_final = str(
            self.remote["clean_root"] / f"{name}_sonic"
        )
        if not _phase_at_least(task_state, "remote_partial_ready"):
            manifest = self.prepare_remote_task(task, task_log)
            atomic_write_json(manifest_path, manifest)
            manifest = self.load_manifest(
                manifest_path,
                task=task,
                expected_partial=expected_partial,
                expected_final=expected_final,
            )
            self.update_task(
                name,
                phase="remote_partial_ready",
                remote_partial=manifest["partial"],
                manifest=str(manifest_path),
            )
        else:
            manifest = self.load_manifest(
                manifest_path,
                task=task,
                expected_partial=expected_partial,
                expected_final=expected_final,
            )

        if (
            _phase_at_least(task_state, "downloaded")
            and not _phase_at_least(task_state, "evaluated")
            and (
            not local_task.is_dir() or local_task.is_symlink()
            )
        ):
            self.logger.warning(
                "local staging for %s is missing; resuming from download", name
            )
            self.update_task(
                name,
                phase="remote_partial_ready",
                status="running",
                local_task=None,
                evaluation_report=None,
                evaluation_summary=None,
            )

        if not _phase_at_least(task_state, "downloaded"):
            free = shutil.disk_usage(self.local_staging_root).free
            present_bytes = 0
            if local_task.exists():
                present_bytes = sum(
                    item.stat().st_size
                    for item in local_task.rglob("*")
                    if item.is_file() and not item.is_symlink()
                )
            required = max(0, task.valid_episode_bytes - present_bytes) + 2 * 1024**3
            if free < required:
                raise WorkflowError(
                    f"insufficient local space for {name}: free={free}, required={required}"
                )
            local_task.mkdir(parents=True, exist_ok=True)
            command = self._rsync_command(manifest["partial"], local_task)
            self.run_logged_command(command, log_path=task_log / "download.log")
            self.update_task(name, phase="downloaded", local_task=str(local_task))

        if not _phase_at_least(task_state, "transfer_verified"):
            self.verify_local_manifest(local_task, manifest)
            self.verify_rsync_checksum(
                remote_path=manifest["partial"],
                local_path=local_task,
                log_path=task_log / "download-checksum.log",
            )
            self.update_task(name, phase="transfer_verified")

        if not _phase_at_least(task_state, "evaluated"):
            report, report_path = self.run_evaluator(
                task=task,
                local_task=local_task,
                manifest=manifest,
                task_log=task_log,
            )
            self.update_task(
                name,
                phase="evaluated",
                evaluation_report=str(report_path),
                evaluation_summary=report["summary"],
            )
        else:
            report_path = Path(task_state["evaluation_report"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            validate_evaluation_report(
                report,
                expected_task=name,
                expected_total=task.episode_hdf5,
                expected_episodes=[item["episode"] for item in manifest["episodes"]],
            )
            if report.get("success_mode") != self.config["success_mode"]:
                raise WorkflowError(
                    "evaluation success_mode differs from job config"
                )
            if report.get("seed") != self.config["seed"]:
                raise WorkflowError("evaluation seed differs from job config")
            selection = report.get("selection")
            if not isinstance(selection, Mapping):
                raise WorkflowError("evaluation selection is missing")
            if selection.get("tasks") != [name]:
                raise WorkflowError("evaluation selection task differs from job")
            if selection.get("candidate_count") != task.episode_hdf5:
                raise WorkflowError("evaluation candidate count differs from job")
            report_root = report.get("dataset_root")
            if (
                not isinstance(report_root, str)
                or Path(report_root).resolve() != local_task.parent.resolve()
            ):
                raise WorkflowError(
                    "evaluation dataset_root differs from staging"
                )
            execution = report.get("execution")
            if (
                not isinstance(execution, Mapping)
                or execution.get("episode_process_isolation") is not True
                or execution.get("episode_timeout_seconds")
                != self.config["evaluation_episode_timeout_seconds"]
            ):
                raise WorkflowError("evaluation process isolation is missing")
            child_reports_dir = execution.get("child_reports_dir")
            if not isinstance(child_reports_dir, str):
                raise WorkflowError("evaluation child report directory is missing")
            child_path = Path(child_reports_dir)
            if (
                not child_path.is_dir()
                or child_path.is_symlink()
                or child_path.parent.resolve() != task_log.resolve()
            ):
                raise WorkflowError("evaluation child report directory is unsafe")

        self.place_successful_episodes(
            task=task,
            task_state=task_state,
            local_task=local_task,
            manifest=manifest,
            report=report,
            task_log=task_log,
        )

        if not _phase_at_least(task_state, "published"):
            publication = self.classify_remote(
                task=task,
                report=report,
                task_log=task_log,
            )
            self.update_task(
                name,
                phase="published",
                status="published",
                passed=publication["passed"],
                failed=publication["failed"],
                invalid=report["summary"].get("invalid", 0),
                remote_clean=(
                    str(self.remote["clean_root"] / f"{name}_sonic")
                    if publication.get("nas_passed", publication["passed"])
                    else None
                ),
                local_passed=publication.get("local_passed", 0),
                nas_passed=publication.get("nas_passed", publication["passed"]),
                remote_rejected=(
                    str(self.remote["rejected_root"] / f"{name}_sonic")
                    if publication["failed"]
                    else None
                ),
            )

        if not self.args.keep_local:
            self.safe_remove_local_task(task_stage)
        self.update_task(name, phase="complete", status="complete", local_released=not self.args.keep_local)
        self.logger.info(
            "task %s complete: passed=%s failed=%s",
            name,
            task_state.get("passed"),
            task_state.get("failed"),
        )

    def run(self, tasks: Sequence[TaskInventory]) -> int:
        self.state["status"] = "running"
        self.state.pop("completed_at", None)
        self.save_state()
        try:
            for task in tasks:
                try:
                    self.process_task(task)
                except Exception as exc:
                    self.update_task(
                        task.name,
                        status="error",
                        last_error=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc(),
                    )
                    self.state["status"] = "stopped_on_error"
                    self.save_state()
                    self.logger.exception("task %s stopped the job", task.name)
                    return 1
            incomplete = [
                name
                for name, item in self.state["tasks"].items()
                if item.get("phase") != "complete"
            ]
            self.state["status"] = "complete" if not incomplete else "selection_complete"
            self.state["completed_at"] = _utc_now()
            self.save_state()
            return 0
        except KeyboardInterrupt:
            self.state["status"] = "interrupted"
            self.save_state()
            return 130


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--ssh-host", default="ly_nas")
    parser.add_argument(
        "--raw-root",
        default="/volume1/unitree_ly_robocasa_data/datasets/sonic_raw",
    )
    parser.add_argument("--source-backup", required=True)
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--rejected-root", required=True)
    parser.add_argument("--local-staging-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument(
        "--minimum-episodes",
        type=int,
        default=10,
        help="Skip whole tasks with fewer valid episodes (default: 10; zero disables).",
    )
    parser.add_argument(
        "--local-dataset-root",
        type=Path,
        default=None,
        help="Prefer merging passed episodes into this local sonic_raw root.",
    )
    parser.add_argument(
        "--local-reserve-gib",
        type=float,
        default=10.0,
        help="Free-space reserve retained after a task-level local merge.",
    )
    parser.add_argument("--success-mode", choices=("any", "final"), default="any")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--evaluation-episode-timeout-seconds",
        type=float,
        default=300.0,
        help="Per-episode timeout used by isolated replay evaluation.",
    )
    parser.add_argument("--keep-local", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--remote-rsync-path",
        default=(
            "/volume1/unitree_ly_robocasa_data/datasets/"
            ".codex_rsync_runtime_20260813/ld-linux-x86-64.so.2 "
            "--library-path /volume1/unitree_ly_robocasa_data/datasets/"
            ".codex_rsync_runtime_20260813 /volume1/unitree_ly_robocasa_data/"
            "datasets/.codex_rsync_runtime_20260813/rsync"
        ),
    )
    return parser


def _configure_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("nas_sonic_raw_cleanup")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}", args.job_id):
        print("error: --job-id must use YYYYMMDD-HHMMSS", file=sys.stderr)
        return 2
    if HOST_PATTERN.fullmatch(args.ssh_host) is None:
        print("error: unsafe --ssh-host", file=sys.stderr)
        return 2
    if args.minimum_episodes < 0:
        print("error: --minimum-episodes must be non-negative", file=sys.stderr)
        return 2
    if not (args.local_reserve_gib >= 0):
        print("error: --local-reserve-gib must be non-negative", file=sys.stderr)
        return 2
    if not (args.evaluation_episode_timeout_seconds > 0):
        print(
            "error: --evaluation-episode-timeout-seconds must be positive",
            file=sys.stderr,
        )
        return 2
    repo_root = Path(__file__).resolve().parents[2]
    evaluator = repo_root / "robocasa/scripts/evaluate_sonic_raw.py"
    try:
        inventory = load_inventory(args.inventory)
        tasks = [
            task
            for task in ordered_tasks(inventory, requested_tasks=args.task)
            if task.episode_hdf5 >= args.minimum_episodes
        ]
        if not tasks:
            raise WorkflowError(
                "no selected task meets --minimum-episodes"
            )
        remote_config = validate_remote_config(
            {
                "raw_root": args.raw_root,
                "backup_root": args.source_backup,
                "clean_root": args.clean_root,
                "rejected_root": args.rejected_root,
            }
        )
        if PurePosixPath(inventory.root) != remote_config["raw_root"]:
            raise WorkflowError(
                "inventory root differs from the configured raw_root"
            )
        inventory_path = args.inventory.expanduser().resolve(strict=True)
        staging_root = args.local_staging_root.expanduser().resolve()
        log_root = args.log_root.expanduser().resolve()
        local_dataset_root = (
            args.local_dataset_root.expanduser().resolve()
            if args.local_dataset_root is not None
            else None
        )
        evaluator = evaluator.resolve(strict=True)
        worker_path = Path(__file__).resolve(strict=True)
        config: dict[str, Any] = {
            "job_id": args.job_id,
            "raw_root": str(remote_config["raw_root"]),
            "backup_root": str(remote_config["backup_root"]),
            "clean_root": str(remote_config["clean_root"]),
            "rejected_root": str(remote_config["rejected_root"]),
            "ssh_host": args.ssh_host,
            "inventory": str(inventory_path),
            "inventory_sha256": _sha256(inventory_path),
            "local_staging_root": str(staging_root),
            "local_dataset_root": (
                str(local_dataset_root)
                if local_dataset_root is not None
                else None
            ),
            "local_reserve_bytes": int(args.local_reserve_gib * 1024**3),
            "minimum_episodes": args.minimum_episodes,
            "selected_tasks": [task.name for task in tasks],
            "log_root": str(log_root),
            "repo_root": str(repo_root),
            "evaluator_path": str(evaluator),
            "evaluator_sha256": _sha256(evaluator),
            "worker_path": str(worker_path),
            "worker_sha256": _sha256(worker_path),
            "python_executable": str(Path(sys.executable).resolve()),
            "remote_rsync_path": validate_remote_rsync_path(
                args.remote_rsync_path
            ),
            "success_mode": args.success_mode,
            "seed": args.seed,
            "evaluation_isolation": "episode_process",
            "evaluation_episode_timeout_seconds": (
                args.evaluation_episode_timeout_seconds
            ),
        }
        if args.dry_run:
            print(json.dumps({
                "config": config,
                "task_order": [
                    {
                        "task": task.name,
                        "episode_hdf5": task.episode_hdf5,
                        "valid_episode_bytes": task.valid_episode_bytes,
                    }
                    for task in tasks
                ],
            }, indent=2, sort_keys=True))
            return 0

        staging_root.mkdir(parents=True, exist_ok=True)
        log_root.mkdir(parents=True, exist_ok=True)
        logger = _configure_logging(log_root / "worker.log")
        state_path = log_root / "job-state.json"
        lock_path = log_root / "worker.lock"
        worker_snapshot = log_root / "code_snapshot" / worker_path.name
        if worker_snapshot.exists():
            if _sha256(worker_snapshot) != config["worker_sha256"]:
                raise WorkflowError(
                    "worker code differs from the existing job snapshot"
                )
        else:
            worker_snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(worker_path, worker_snapshot)
            if _sha256(worker_snapshot) != config["worker_sha256"]:
                raise WorkflowError("worker code snapshot verification failed")
        with lock_path.open("a+", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise WorkflowError(f"another worker holds {lock_path}") from exc
            if state_path.exists():
                state = load_job_state(state_path, config)
            else:
                state = build_initial_state(config, inventory)
                atomic_write_json(state_path, state)
            worker = NasCleanupWorker(
                args=args,
                inventory=inventory,
                config=config,
                state_path=state_path,
                state=state,
                logger=logger,
            )
            return worker.run(tasks)
    except (WorkflowError, OSError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
