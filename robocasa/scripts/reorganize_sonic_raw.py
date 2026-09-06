#!/usr/bin/env python3
"""Safely reorganize timestamped SONIC runs into one directory per task.

The source layout is expected to look like::

    sonic_raw/
      2026-08-13-16-50-11_LoadDishwasher_sonic/
        demo.hdf5                 # deliberately not copied
        episodes/
          ep_.../
            ep_demo.hdf5
            ...

The resulting layout is::

    sonic_raw/
      LoadDishwasher_sonic/
        ep_.../
          ep_demo.hdf5
          ...

Dry-run is the default. With ``--apply``, a complete sibling ``.partial`` tree
is built and verified before the old root is renamed to a timestamped backup
and the partial tree is atomically renamed into place. The old tree is never
deleted.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
from typing import Any, Iterable, Sequence

import h5py


RUN_DIRECTORY_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_"
    r"(?P<task>[A-Za-z0-9][A-Za-z0-9_]*)_sonic$"
)
EPISODE_DIRECTORY_PATTERN = re.compile(r"^ep_[A-Za-z0-9_.-]+$")
REQUIRED_DATA_ATTRIBUTES = {
    "env",
    "env_args",
    "sonic_gains",
    "sonic_runtime",
    "total",
}
REQUIRED_DEMO_ATTRIBUTES = {"ep_meta", "model_file", "num_samples"}
REQUIRED_DEMO_DATASETS = {"actions", "states", "states_integration"}
HASH_CHUNK_SIZE = 4 * 1024 * 1024
MANIFEST_SCHEMA_VERSION = 1


class ReorganizationError(RuntimeError):
    """Raised when a plan or copied dataset fails a safety check."""


@dataclasses.dataclass(frozen=True)
class EpisodeSource:
    """One valid episode selected for the reorganized dataset."""

    task: str
    run_dir: Path
    episode_dir: Path
    frame_count: int

    @property
    def target_relative_path(self) -> Path:
        return Path(f"{self.task}_sonic") / self.episode_dir.name


@dataclasses.dataclass(frozen=True)
class ExcludedEpisode:
    """An episode excluded because it has no per-episode HDF5 file."""

    task: str
    run_dir: Path
    episode_dir: Path
    reason: str = "missing_ep_demo.hdf5"


@dataclasses.dataclass(frozen=True)
class ReorganizationPlan:
    """A fully preflighted, immutable reorganization plan."""

    root: Path
    run_dirs: tuple[Path, ...]
    episodes: tuple[EpisodeSource, ...]
    excluded_episodes: tuple[ExcludedEpisode, ...]
    excluded_batch_hdf5: tuple[Path, ...]

    @property
    def task_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for episode in self.episodes:
            counts[episode.task] = counts.get(episode.task, 0) + 1
        return dict(sorted(counts.items()))


@dataclasses.dataclass(frozen=True)
class OutputPaths:
    """Sibling paths used for staging, backup, and the external journal."""

    partial_root: Path
    backup_root: Path
    manifest_path: Path


def _decoded_string(value: Any, *, label: str, path: Path) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value:
        raise ReorganizationError(f"{path}: {label} must be a non-empty string")
    return value


def _json_attribute(
    attributes: h5py.AttributeManager,
    name: str,
    *,
    path: Path,
) -> Any:
    value = _decoded_string(attributes[name], label=name, path=path)
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ReorganizationError(f"{path}: invalid JSON attribute {name}: {exc}") from exc


def validate_episode_hdf5(path: Path, *, expected_task: str) -> int:
    """Validate one SONIC episode HDF5 and return its frame count."""

    try:
        hdf5_file = h5py.File(path, "r")
    except (OSError, ValueError) as exc:
        raise ReorganizationError(f"cannot open HDF5 {path}: {exc}") from exc

    try:
        with hdf5_file:
            if set(hdf5_file.keys()) != {"data"}:
                raise ReorganizationError(
                    f"{path}: expected only root group 'data', got "
                    f"{sorted(hdf5_file.keys())}"
                )
            data = hdf5_file["data"]
            if not isinstance(data, h5py.Group):
                raise ReorganizationError(f"{path}: 'data' is not an HDF5 group")

            missing_data_attrs = REQUIRED_DATA_ATTRIBUTES - set(data.attrs.keys())
            if missing_data_attrs:
                raise ReorganizationError(
                    f"{path}: missing data attributes {sorted(missing_data_attrs)}"
                )

            recorded_task = _decoded_string(
                data.attrs["env"], label="env", path=path
            )
            if recorded_task != expected_task:
                raise ReorganizationError(
                    f"{path}: run task {expected_task!r} does not match HDF5 env "
                    f"{recorded_task!r}"
                )

            env_args = _json_attribute(data.attrs, "env_args", path=path)
            if not isinstance(env_args, dict) or env_args.get("env_name") != expected_task:
                raise ReorganizationError(
                    f"{path}: env_args.env_name does not match {expected_task!r}"
                )
            for json_attr in ("sonic_gains", "sonic_runtime"):
                parsed = _json_attribute(data.attrs, json_attr, path=path)
                if not isinstance(parsed, dict):
                    raise ReorganizationError(
                        f"{path}: {json_attr} must decode to a JSON object"
                    )

            demo_names = list(data.keys())
            if len(demo_names) != 1:
                raise ReorganizationError(
                    f"{path}: expected exactly one demo, found {len(demo_names)}"
                )
            demo = data[demo_names[0]]
            if not isinstance(demo, h5py.Group):
                raise ReorganizationError(
                    f"{path}: {demo_names[0]!r} is not an HDF5 group"
                )

            missing_demo_attrs = REQUIRED_DEMO_ATTRIBUTES - set(demo.attrs.keys())
            if missing_demo_attrs:
                raise ReorganizationError(
                    f"{path}: missing demo attributes {sorted(missing_demo_attrs)}"
                )
            missing_datasets = REQUIRED_DEMO_DATASETS - set(demo.keys())
            if missing_datasets:
                raise ReorganizationError(
                    f"{path}: missing demo datasets {sorted(missing_datasets)}"
                )

            lengths: dict[str, int] = {}
            for dataset_name in sorted(REQUIRED_DEMO_DATASETS):
                dataset = demo[dataset_name]
                if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 2:
                    raise ReorganizationError(
                        f"{path}: {dataset_name} must be a two-dimensional dataset"
                    )
                lengths[dataset_name] = int(dataset.shape[0])
            if len(set(lengths.values())) != 1 or not next(iter(lengths.values())):
                raise ReorganizationError(
                    f"{path}: state/action lengths must be equal and nonzero: {lengths}"
                )

            frame_count = lengths["actions"]
            try:
                num_samples = int(demo.attrs["num_samples"])
                total = int(data.attrs["total"])
            except (TypeError, ValueError) as exc:
                raise ReorganizationError(
                    f"{path}: num_samples and total must be integers"
                ) from exc
            if num_samples != frame_count or total != frame_count:
                raise ReorganizationError(
                    f"{path}: frame count {frame_count}, num_samples {num_samples}, "
                    f"total {total} disagree"
                )

            ep_meta_text = _decoded_string(
                demo.attrs["ep_meta"], label="ep_meta", path=path
            )
            try:
                ep_meta = json.loads(ep_meta_text)
            except json.JSONDecodeError as exc:
                raise ReorganizationError(f"{path}: invalid ep_meta JSON: {exc}") from exc
            if not isinstance(ep_meta, dict):
                raise ReorganizationError(f"{path}: ep_meta must decode to an object")
            model_xml = _decoded_string(
                demo.attrs["model_file"], label="model_file", path=path
            )

        episode_dir = path.parent
        ep_meta_path = episode_dir / "ep_meta.json"
        model_path = episode_dir / "model.xml"
        if not ep_meta_path.is_file() or not model_path.is_file():
            raise ReorganizationError(
                f"{episode_dir}: ep_meta.json and model.xml are required"
            )
        if ep_meta_path.read_text(encoding="utf-8") != ep_meta_text:
            raise ReorganizationError(
                f"{path}: ep_meta HDF5 attribute differs from ep_meta.json"
            )
        if model_path.read_text(encoding="utf-8") != model_xml:
            raise ReorganizationError(
                f"{path}: model_file HDF5 attribute differs from model.xml"
            )
        return frame_count
    except KeyError as exc:
        raise ReorganizationError(f"{path}: missing HDF5 key or attribute {exc}") from exc


def _validate_source_tree(episode_dir: Path) -> None:
    """Reject links and special files before copying an episode recursively."""

    for current_root, directory_names, file_names in os.walk(
        episode_dir, followlinks=False
    ):
        current = Path(current_root)
        for name in [*directory_names, *file_names]:
            entry = current / name
            mode = entry.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ReorganizationError(f"source episode contains symlink: {entry}")
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise ReorganizationError(f"source episode contains special file: {entry}")


def _parse_run_directory(run_dir: Path) -> str:
    match = RUN_DIRECTORY_PATTERN.fullmatch(run_dir.name)
    if match is None:
        raise ReorganizationError(
            f"run directory name does not match timestamped SONIC format: {run_dir}"
        )
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ReorganizationError(f"run directory must be a real directory: {run_dir}")
    episodes_dir = run_dir / "episodes"
    if episodes_dir.is_symlink() or not episodes_dir.is_dir():
        raise ReorganizationError(f"run has no real episodes directory: {run_dir}")
    return match.group("task")


def _discover_root_run_dirs(root: Path) -> list[Path]:
    run_dirs: list[Path] = []
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if RUN_DIRECTORY_PATTERN.fullmatch(child.name) is None:
            continue
        if child.is_symlink() or not child.is_dir():
            raise ReorganizationError(
                f"timestamped run entry must be a real directory: {child}"
            )
        run_dirs.append(child.resolve(strict=True))
    return run_dirs


def build_plan(
    root: Path | str,
    *,
    extra_run_dirs: Iterable[Path | str] = (),
) -> ReorganizationPlan:
    """Discover and fully preflight all source runs and episodes."""

    root_input = Path(root).expanduser()
    if root_input.is_symlink() or not root_input.is_dir():
        raise ReorganizationError(f"root must be a real directory: {root_input}")
    root_path = root_input.resolve(strict=True)

    run_dirs = _discover_root_run_dirs(root_path)
    seen_runs = set(run_dirs)
    for extra in extra_run_dirs:
        extra_input = Path(extra).expanduser()
        if extra_input.is_symlink():
            raise ReorganizationError(f"extra run cannot be a symlink: {extra_input}")
        try:
            extra_path = extra_input.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ReorganizationError(f"extra run does not exist: {extra_input}") from exc
        if extra_path in seen_runs:
            raise ReorganizationError(f"run directory specified more than once: {extra_path}")
        seen_runs.add(extra_path)
        run_dirs.append(extra_path)
    run_dirs.sort(key=lambda path: str(path))

    episodes: list[EpisodeSource] = []
    excluded: list[ExcludedEpisode] = []
    batch_hdf5: list[Path] = []
    targets: dict[tuple[str, str], Path] = {}

    for run_dir in run_dirs:
        task = _parse_run_directory(run_dir)
        aggregate = run_dir / "demo.hdf5"
        if aggregate.exists() or aggregate.is_symlink():
            batch_hdf5.append(aggregate)

        episodes_dir = run_dir / "episodes"
        for episode_dir in sorted(episodes_dir.iterdir(), key=lambda path: path.name):
            if not EPISODE_DIRECTORY_PATTERN.fullmatch(episode_dir.name):
                continue
            if episode_dir.is_symlink() or not episode_dir.is_dir():
                raise ReorganizationError(
                    f"episode entry must be a real directory: {episode_dir}"
                )

            hdf5_path = episode_dir / "ep_demo.hdf5"
            if hdf5_path.is_symlink():
                raise ReorganizationError(
                    f"ep_demo.hdf5 must not be a symlink: {hdf5_path}"
                )
            if not hdf5_path.is_file():
                excluded.append(
                    ExcludedEpisode(
                        task=task,
                        run_dir=run_dir,
                        episode_dir=episode_dir,
                    )
                )
                continue

            target_key = (task, episode_dir.name)
            previous = targets.get(target_key)
            if previous is not None:
                raise ReorganizationError(
                    f"target collision for {task}_sonic/{episode_dir.name}: "
                    f"{previous} and {episode_dir}"
                )
            targets[target_key] = episode_dir

            _validate_source_tree(episode_dir)
            frame_count = validate_episode_hdf5(hdf5_path, expected_task=task)
            episodes.append(
                EpisodeSource(
                    task=task,
                    run_dir=run_dir,
                    episode_dir=episode_dir,
                    frame_count=frame_count,
                )
            )

    if not episodes:
        raise ReorganizationError("no valid ep_demo.hdf5 episodes were found")

    episodes.sort(key=lambda episode: (episode.task, episode.episode_dir.name))
    excluded.sort(key=lambda episode: (episode.task, episode.episode_dir.name))
    return ReorganizationPlan(
        root=root_path,
        run_dirs=tuple(run_dirs),
        episodes=tuple(episodes),
        excluded_episodes=tuple(excluded),
        excluded_batch_hdf5=tuple(sorted(batch_hdf5, key=str)),
    )


def make_output_paths(root: Path, *, run_id: str | None = None) -> OutputPaths:
    """Create deterministic sibling output names without touching the filesystem."""

    if run_id is None:
        run_id = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ReorganizationError(f"unsafe run id: {run_id!r}")
    parent = root.parent
    stem = root.name
    return OutputPaths(
        partial_root=parent / f".{stem}.reorganize-{run_id}.partial",
        backup_root=parent / f"{stem}.backup-{run_id}",
        manifest_path=parent / f"{stem}.reorganize-{run_id}.json",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file_inventory(root: Path) -> dict[str, tuple[int, int]]:
    """Return relative file -> (size, mtime_ns), rejecting links/special files."""

    inventory: dict[str, tuple[int, int]] = {}
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        for name in directory_names:
            directory = current / name
            mode = directory.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ReorganizationError(f"unexpected directory entry: {directory}")
        for name in file_names:
            file_path = current / name
            file_stat = file_path.lstat()
            if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
                raise ReorganizationError(f"unexpected file entry: {file_path}")
            relative = file_path.relative_to(root).as_posix()
            inventory[relative] = (file_stat.st_size, file_stat.st_mtime_ns)
    return inventory


def _verify_episode_copy(source: Path, target: Path) -> list[dict[str, Any]]:
    source_inventory = _regular_file_inventory(source)
    target_inventory = _regular_file_inventory(target)
    if set(source_inventory) != set(target_inventory):
        missing = sorted(set(source_inventory) - set(target_inventory))
        extra = sorted(set(target_inventory) - set(source_inventory))
        raise ReorganizationError(
            f"copied file set differs for {source}: missing={missing}, extra={extra}"
        )

    verified_files: list[dict[str, Any]] = []
    for relative in sorted(source_inventory):
        source_size, source_mtime_ns = source_inventory[relative]
        target_size, _ = target_inventory[relative]
        if source_size != target_size:
            raise ReorganizationError(
                f"size mismatch after copy: {source / relative} ({source_size}) != "
                f"{target / relative} ({target_size})"
            )
        source_hash = _sha256(source / relative)
        target_hash = _sha256(target / relative)
        if source_hash != target_hash:
            raise ReorganizationError(
                f"SHA-256 mismatch after copy: {source / relative}"
            )
        verified_files.append(
            {
                "relative_path": relative,
                "size": source_size,
                "sha256": source_hash,
                "source_mtime_ns": source_mtime_ns,
            }
        )
    return verified_files


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _manifest_for_plan(
    plan: ReorganizationPlan,
    outputs: OutputPaths,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "planned",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "root": str(plan.root),
        "partial_root": str(outputs.partial_root),
        "backup_root": str(outputs.backup_root),
        "manifest_path": str(outputs.manifest_path),
        "source_runs": [str(path) for path in plan.run_dirs],
        "task_counts": plan.task_counts,
        "excluded_batch_hdf5": [
            str(path) for path in plan.excluded_batch_hdf5
        ],
        "excluded_episodes": [
            {
                "task": episode.task,
                "run_dir": str(episode.run_dir),
                "episode_dir": str(episode.episode_dir),
                "reason": episode.reason,
            }
            for episode in plan.excluded_episodes
        ],
        "episodes": [
            {
                "task": episode.task,
                "source": str(episode.episode_dir),
                "target_relative_path": episode.target_relative_path.as_posix(),
                "frame_count": episode.frame_count,
                "copy_status": "pending",
                "files": [],
            }
            for episode in plan.episodes
        ],
    }


def _preflight_outputs(plan: ReorganizationPlan, outputs: OutputPaths) -> None:
    if plan.root.parent != outputs.partial_root.parent:
        raise ReorganizationError("partial root must be a sibling of the source root")
    if plan.root.parent != outputs.backup_root.parent:
        raise ReorganizationError("backup root must be a sibling of the source root")
    if plan.root.parent != outputs.manifest_path.parent:
        raise ReorganizationError("manifest must be outside and adjacent to the source root")
    output_paths = {
        outputs.partial_root,
        outputs.backup_root,
        outputs.manifest_path,
    }
    if len(output_paths) != 3 or plan.root in output_paths:
        raise ReorganizationError("output paths collide with one another or with root")
    for path in output_paths:
        if path.exists() or path.is_symlink():
            raise ReorganizationError(f"refusing to overwrite existing output: {path}")
    if os.stat(plan.root).st_dev != os.stat(plan.root.parent).st_dev:
        raise ReorganizationError("root and its parent must be on the same filesystem")


def _sources_unchanged(
    plan: ReorganizationPlan,
    manifest: dict[str, Any],
) -> None:
    """Detect files added, removed, or modified while staging was built."""

    manifest_episodes = {
        entry["target_relative_path"]: entry for entry in manifest["episodes"]
    }
    for episode in plan.episodes:
        entry = manifest_episodes[episode.target_relative_path.as_posix()]
        expected = {
            file_entry["relative_path"]: (
                file_entry["size"],
                file_entry["source_mtime_ns"],
            )
            for file_entry in entry["files"]
        }
        current = _regular_file_inventory(episode.episode_dir)
        if current != expected:
            raise ReorganizationError(
                f"source episode changed while copying: {episode.episode_dir}"
            )

    expected_root_runs = {
        run_dir for run_dir in plan.run_dirs if run_dir.parent == plan.root
    }
    current_root_runs = set(_discover_root_run_dirs(plan.root))
    if current_root_runs != expected_root_runs:
        raise ReorganizationError("timestamped run directories changed while copying")

    valid_by_run: dict[Path, set[Path]] = {run_dir: set() for run_dir in plan.run_dirs}
    excluded_by_run: dict[Path, set[Path]] = {
        run_dir: set() for run_dir in plan.run_dirs
    }
    for episode in plan.episodes:
        valid_by_run[episode.run_dir].add(episode.episode_dir)
    for episode in plan.excluded_episodes:
        excluded_by_run[episode.run_dir].add(episode.episode_dir)

    for run_dir in plan.run_dirs:
        current_episodes: set[Path] = set()
        for episode_dir in (run_dir / "episodes").iterdir():
            if EPISODE_DIRECTORY_PATTERN.fullmatch(episode_dir.name) is None:
                continue
            if episode_dir.is_symlink() or not episode_dir.is_dir():
                raise ReorganizationError(
                    f"source episode entry changed type while copying: {episode_dir}"
                )
            current_episodes.add(episode_dir)
        expected_episodes = valid_by_run[run_dir] | excluded_by_run[run_dir]
        if current_episodes != expected_episodes:
            raise ReorganizationError(
                f"episode set changed while copying in {run_dir}"
            )
        for episode_dir in valid_by_run[run_dir]:
            if not (episode_dir / "ep_demo.hdf5").is_file():
                raise ReorganizationError(
                    f"ep_demo.hdf5 disappeared while copying: {episode_dir}"
                )
        for episode_dir in excluded_by_run[run_dir]:
            hdf5_path = episode_dir / "ep_demo.hdf5"
            if hdf5_path.exists() or hdf5_path.is_symlink():
                raise ReorganizationError(
                    f"excluded episode gained ep_demo.hdf5 while copying: {episode_dir}"
                )


def _validate_staging_layout(
    plan: ReorganizationPlan,
    partial_root: Path,
) -> None:
    if any(partial_root.rglob("demo.hdf5")):
        raise ReorganizationError("staging unexpectedly contains a batch demo.hdf5")
    expected = {episode.target_relative_path.as_posix() for episode in plan.episodes}
    actual: set[str] = set()
    for task_dir in partial_root.iterdir():
        if not task_dir.is_dir() or task_dir.is_symlink():
            raise ReorganizationError(f"unexpected staging entry: {task_dir}")
        for episode_dir in task_dir.iterdir():
            if not episode_dir.is_dir() or episode_dir.is_symlink():
                raise ReorganizationError(f"unexpected staging entry: {episode_dir}")
            actual.add(episode_dir.relative_to(partial_root).as_posix())
    if actual != expected:
        raise ReorganizationError(
            f"staging episode set differs: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _rename(source: Path, target: Path) -> None:
    os.rename(source, target)


def execute_plan(plan: ReorganizationPlan, outputs: OutputPaths) -> Path:
    """Copy, verify, journal, and atomically publish a preflighted plan."""

    _preflight_outputs(plan, outputs)
    manifest = _manifest_for_plan(plan, outputs)
    _atomic_write_json(outputs.manifest_path, manifest)

    try:
        outputs.partial_root.mkdir(mode=0o755)
        manifest["status"] = "copying"
        _atomic_write_json(outputs.manifest_path, manifest)

        for index, episode in enumerate(plan.episodes):
            target = outputs.partial_root / episode.target_relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                episode.episode_dir,
                target,
                copy_function=shutil.copy2,
            )
            files = _verify_episode_copy(episode.episode_dir, target)
            copied_frames = validate_episode_hdf5(
                target / "ep_demo.hdf5", expected_task=episode.task
            )
            if copied_frames != episode.frame_count:
                raise ReorganizationError(
                    f"frame count changed after copy: {episode.episode_dir}"
                )
            manifest["episodes"][index]["copy_status"] = "verified"
            manifest["episodes"][index]["files"] = files
            manifest["copied_episode_count"] = index + 1
            _atomic_write_json(outputs.manifest_path, manifest)

        _validate_staging_layout(plan, outputs.partial_root)
        _sources_unchanged(plan, manifest)
        manifest["status"] = "ready_to_switch"
        _atomic_write_json(outputs.manifest_path, manifest)
    except (Exception, KeyboardInterrupt) as exc:
        manifest["status"] = "staging_failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _atomic_write_json(outputs.manifest_path, manifest)
        raise

    try:
        _rename(plan.root, outputs.backup_root)
    except (Exception, KeyboardInterrupt) as exc:
        manifest["status"] = "switch_failed_before_backup"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _atomic_write_json(outputs.manifest_path, manifest)
        raise
    try:
        manifest["status"] = "root_backed_up"
        _atomic_write_json(outputs.manifest_path, manifest)
        _rename(outputs.partial_root, plan.root)
    except (Exception, KeyboardInterrupt) as exc:
        rollback_error: str | None = None
        if not plan.root.exists() and outputs.backup_root.exists():
            try:
                _rename(outputs.backup_root, plan.root)
            except Exception as rollback_exc:  # pragma: no cover - catastrophic OS failure
                rollback_error = f"{type(rollback_exc).__name__}: {rollback_exc}"
        manifest["status"] = (
            "switch_failed_rolled_back"
            if rollback_error is None
            else "switch_failed_manual_recovery_required"
        )
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        if rollback_error is not None:
            manifest["rollback_error"] = rollback_error
        _atomic_write_json(outputs.manifest_path, manifest)
        raise

    manifest["status"] = "complete"
    manifest["completed_at"] = dt.datetime.now().astimezone().isoformat()
    manifest["published_root"] = str(plan.root)
    manifest["preserved_backup_root"] = str(outputs.backup_root)
    _atomic_write_json(outputs.manifest_path, manifest)
    return outputs.manifest_path


def _print_plan(plan: ReorganizationPlan, outputs: OutputPaths) -> None:
    print("SONIC raw reorganization dry-run")
    print(f"  root: {plan.root}")
    print(f"  valid episodes: {len(plan.episodes)}")
    print(f"  excluded episodes without ep_demo.hdf5: {len(plan.excluded_episodes)}")
    print(f"  excluded batch demo.hdf5 files: {len(plan.excluded_batch_hdf5)}")
    print("  task counts:")
    for task, count in plan.task_counts.items():
        print(f"    {task}_sonic: {count}")
    for episode in plan.excluded_episodes:
        print(f"  EXCLUDE {episode.episode_dir}: {episode.reason}")
    print("  --apply would use:")
    print(f"    partial: {outputs.partial_root}")
    print(f"    backup: {outputs.backup_root}")
    print(f"    manifest: {outputs.manifest_path}")
    print("No files were changed. Re-run with --apply to execute this exact workflow.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Current sonic_raw root containing timestamped collection runs.",
    )
    parser.add_argument(
        "--extra-run-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional timestamped run outside root (repeat for multiple runs, "
            "for example recoverable runs in Trash)."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Build, verify, and atomically publish the reorganized root.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = build_plan(args.root, extra_run_dirs=args.extra_run_dir)
        outputs = make_output_paths(plan.root)
        if not args.apply:
            _print_plan(plan, outputs)
            return 0
        manifest_path = execute_plan(plan, outputs)
    except ReorganizationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (OSError, KeyboardInterrupt) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Reorganization complete. Manifest: {manifest_path}")
    print(f"Old tree preserved at: {outputs.backup_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
