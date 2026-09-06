#!/usr/bin/env python3
"""Organize local SONIC raw data by task using only replay-passed episodes.

The active output layout is::

    sonic_raw/
      atomic_tasks/
        <Task>_sonic/
          episodes/ep_.../
      composite_tasks/
        <Task>_sonic/
          episodes/ep_.../

Replay results are read from top-level JSON files in ``--reports-dir``.  For
each current raw episode the newest result is selected using the isolated
result completion timestamp, falling back to the report completion timestamp.
Only a strict ``success_mode=any`` result backed by raw NPZ state replay may be
published.

Dry-run is the default.  ``--apply`` hashes and journals every selected file,
moves passed episode directories into a same-filesystem sibling staging tree,
verifies that tree, and then atomically switches the active root.  The old root
(containing failed, invalid, incomplete, and aggregate batch data) is renamed
to a recoverable backup.  No raw data is deleted and no large copy is made.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Iterable, Sequence

_VALIDATOR_PATH = Path(__file__).with_name("reorganize_sonic_raw.py")
_VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "_robocasa_reorganize_sonic_raw_local", _VALIDATOR_PATH
)
if _VALIDATOR_SPEC is None or _VALIDATOR_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load SONIC raw validator: {_VALIDATOR_PATH}")
_VALIDATOR_MODULE = importlib.util.module_from_spec(_VALIDATOR_SPEC)
sys.modules[_VALIDATOR_SPEC.name] = _VALIDATOR_MODULE
_VALIDATOR_SPEC.loader.exec_module(_VALIDATOR_MODULE)
ReorganizationError = _VALIDATOR_MODULE.ReorganizationError
validate_episode_hdf5 = _VALIDATOR_MODULE.validate_episode_hdf5


RUN_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_"
    r"(?P<task>[A-Za-z0-9][A-Za-z0-9_]*)_sonic$"
)
TASK_PATTERN = re.compile(r"^(?P<task>[A-Za-z0-9][A-Za-z0-9_]*)_sonic$")
EPISODE_PATTERN = re.compile(r"^ep_[A-Za-z0-9_.-]+$")
HASH_CHUNK_SIZE = 4 * 1024 * 1024
SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class RawEpisode:
    task: str
    source_batch: Path
    source: Path
    has_hdf5: bool

    @property
    def key(self) -> tuple[str, str]:
        return (self.task, self.source.name)


@dataclasses.dataclass(frozen=True)
class ReplayEvidence:
    task: str
    episode: str
    status: str
    completed_at: dt.datetime
    report: Path
    report_sha256: str
    result: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class SelectedEpisode:
    raw: RawEpisode
    category: str
    evidence: ReplayEvidence
    frame_count: int

    @property
    def target_relative(self) -> Path:
        return (
            Path(f"{self.category}_tasks")
            / f"{self.raw.task}_sonic"
            / "episodes"
            / self.raw.source.name
        )


@dataclasses.dataclass(frozen=True)
class Plan:
    root: Path
    reports_dir: Path
    reports: tuple[Path, ...]
    all_episodes: tuple[RawEpisode, ...]
    selected: tuple[SelectedEpisode, ...]
    resolved: dict[tuple[str, str], ReplayEvidence]
    source_snapshot: dict[str, tuple[int, int]]

    @property
    def task_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for episode in self.selected:
            counts[episode.raw.task] = counts.get(episode.raw.task, 0) + 1
        return dict(sorted(counts.items()))


@dataclasses.dataclass(frozen=True)
class Outputs:
    partial_root: Path
    backup_root: Path
    manifest: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_time(value: Any, *, label: str) -> dt.datetime:
    if not isinstance(value, str) or not value:
        raise ReorganizationError(f"{label}: missing completion timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise ReorganizationError(f"{label}: invalid timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        raise ReorganizationError(f"{label}: completion timestamp has no timezone")
    return parsed


def _result_time(result: dict[str, Any], report: dict[str, Any], path: Path) -> dt.datetime:
    isolation = result.get("isolation")
    if isinstance(isolation, dict) and isolation.get("completed_at"):
        value = isolation["completed_at"]
    else:
        value = result.get("completed_at") or report.get("completed_at")
    return _parse_time(value, label=str(path))


def _real_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ReorganizationError(f"{label} must be a real directory: {path}")
    return path.resolve(strict=True)


def _episode_dirs(container: Path) -> Iterable[Path]:
    if not container.is_dir() or container.is_symlink():
        return ()
    return (
        child
        for child in sorted(container.iterdir(), key=lambda item: item.name)
        if EPISODE_PATTERN.fullmatch(child.name)
    )


def discover_raw_episodes(root: Path) -> tuple[RawEpisode, ...]:
    episodes: list[RawEpisode] = []
    seen: dict[tuple[str, str], Path] = {}
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if child.name in {"atomic_tasks", "composite_tasks"}:
            raise ReorganizationError(
                f"root already contains organized category directory: {child}"
            )
        run_match = RUN_PATTERN.fullmatch(child.name)
        task_match = TASK_PATTERN.fullmatch(child.name)
        if run_match is not None:
            task = run_match.group("task")
            containers = (child / "episodes",)
        elif task_match is not None:
            task = task_match.group("task")
            containers = (child, child / "episodes")
        else:
            continue
        _real_directory(child, label="raw source directory")
        for container in containers:
            for episode_dir in _episode_dirs(container):
                if episode_dir.is_symlink() or not episode_dir.is_dir():
                    raise ReorganizationError(
                        f"episode must be a real directory: {episode_dir}"
                    )
                episode_path = episode_dir.resolve(strict=True)
                key = (task, episode_path.name)
                previous = seen.get(key)
                if previous is not None:
                    raise ReorganizationError(
                        f"episode collision for {task}/{episode_path.name}: "
                        f"{previous} and {episode_path}"
                    )
                seen[key] = episode_path
                hdf5 = episode_path / "ep_demo.hdf5"
                if hdf5.is_symlink():
                    raise ReorganizationError(f"ep_demo.hdf5 is a symlink: {hdf5}")
                episodes.append(
                    RawEpisode(
                        task=task,
                        source_batch=child.resolve(strict=True),
                        source=episode_path,
                        has_hdf5=hdf5.is_file(),
                    )
                )
    if not episodes:
        raise ReorganizationError(f"no raw episodes found below {root}")
    return tuple(sorted(episodes, key=lambda item: (item.task, item.source.name)))


def _load_report_evidence(
    reports_dir: Path,
    episodes: tuple[RawEpisode, ...],
) -> tuple[tuple[Path, ...], dict[tuple[str, str], ReplayEvidence]]:
    local = {episode.key: episode for episode in episodes}
    candidates: dict[tuple[str, str], list[ReplayEvidence]] = {
        key: [] for key in local
    }
    used_reports: set[Path] = set()
    for report_path in sorted(reports_dir.glob("*.json"), key=lambda item: item.name):
        if report_path.is_symlink() or not report_path.is_file():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(report, dict) or not isinstance(report.get("results"), list):
            continue
        report_hash: str | None = None
        for result in report["results"]:
            if not isinstance(result, dict):
                continue
            key = (result.get("task"), result.get("episode"))
            raw = local.get(key)
            if raw is None:
                continue
            episode_dir_text = result.get("episode_dir")
            if not isinstance(episode_dir_text, str):
                continue
            result_source = Path(episode_dir_text).expanduser().resolve(strict=False)
            if result_source != raw.source:
                continue
            status = result.get("status")
            if not isinstance(status, str):
                continue
            if report_hash is None:
                report_hash = _sha256(report_path)
            evidence = ReplayEvidence(
                task=raw.task,
                episode=raw.source.name,
                status=status,
                completed_at=_result_time(result, report, report_path),
                report=report_path.resolve(strict=True),
                report_sha256=report_hash,
                result=result,
            )
            candidates[key].append(evidence)
            used_reports.add(evidence.report)

    resolved: dict[tuple[str, str], ReplayEvidence] = {}
    for key, evidence_items in candidates.items():
        if not evidence_items:
            continue
        latest_time = max(item.completed_at for item in evidence_items)
        latest = [item for item in evidence_items if item.completed_at == latest_time]
        statuses = {item.status for item in latest}
        if len(statuses) != 1:
            details = [(str(item.report), item.status) for item in latest]
            raise ReorganizationError(
                f"conflicting latest replay results for {key}: {details}"
            )
        latest.sort(key=lambda item: str(item.report))
        resolved[key] = latest[-1]
    return tuple(sorted(used_reports, key=str)), resolved


def _class_names_below(path: Path) -> set[str]:
    names: set[str] = set()
    if not path.is_dir():
        return names
    for source in path.rglob("*.py"):
        try:
            module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise ReorganizationError(f"cannot inspect task module {source}: {exc}") from exc
        names.update(node.name for node in ast.walk(module) if isinstance(node, ast.ClassDef))
    return names


def _task_categories(repo_root: Path, tasks: set[str]) -> dict[str, str]:
    kitchen = repo_root / "robocasa" / "environments" / "kitchen"
    atomic = _class_names_below(kitchen / "atomic")
    composite = _class_names_below(kitchen / "composite")
    categories: dict[str, str] = {}
    for task in sorted(tasks):
        matches = []
        if task in atomic:
            matches.append("atomic")
        if task in composite:
            matches.append("composite")
        if len(matches) != 1:
            raise ReorganizationError(
                f"task {task!r} has ambiguous or missing category: {matches}"
            )
        categories[task] = matches[0]
    return categories


def _regular_file_inventory(root: Path) -> dict[str, tuple[int, int]]:
    inventory: dict[str, tuple[int, int]] = {}
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        for name in directory_names:
            path = current / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ReorganizationError(f"unexpected directory entry: {path}")
        for name in file_names:
            path = current / name
            file_stat = path.lstat()
            if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
                raise ReorganizationError(f"unexpected file entry: {path}")
            inventory[path.relative_to(root).as_posix()] = (
                file_stat.st_size,
                file_stat.st_mtime_ns,
            )
    return inventory


def _validate_episode_files(episode: RawEpisode) -> None:
    inventory = _regular_file_inventory(episode.source)
    names = set(inventory)
    required = {"ep_demo.hdf5", "ep_meta.json", "model.xml"}
    missing = required - names
    state_files = sorted(name for name in names if Path(name).name.startswith("state_") and name.endswith(".npz"))
    if missing or len(state_files) != 1:
        raise ReorganizationError(
            f"{episode.source}: required raw layout failed; "
            f"missing={sorted(missing)}, state_files={state_files}"
        )
    if names != required | set(state_files):
        raise ReorganizationError(
            f"{episode.source}: unexpected episode files: "
            f"{sorted(names - required - set(state_files))}"
        )


def _validate_pass_evidence(raw: RawEpisode, evidence: ReplayEvidence) -> None:
    result = evidence.result
    expected_hdf5 = raw.source / "ep_demo.hdf5"
    hdf5_text = result.get("hdf5")
    if not isinstance(hdf5_text, str):
        raise ReorganizationError(f"passed result has no HDF5 path: {raw.source}")
    result_hdf5 = Path(hdf5_text).expanduser().resolve(strict=False)
    checks = {
        "status": evidence.status == "passed",
        "success": result.get("success") is True,
        "any_success": result.get("any_success") is True,
        "success_mode": result.get("success_mode") == "any",
        "state_source": str(result.get("state_source", "")).startswith("raw_npz/"),
        "hdf5_path": result_hdf5 == expected_hdf5,
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise ReorganizationError(
            f"passed replay evidence is not strict for {raw.source}: {failed}"
        )


def build_plan(root: Path, reports_dir: Path, repo_root: Path) -> Plan:
    root = _real_directory(root, label="raw root")
    reports_dir = _real_directory(reports_dir, label="reports directory")
    repo_root = _real_directory(repo_root, label="repository root")
    source_snapshot = _regular_file_inventory(root)
    all_episodes = discover_raw_episodes(root)
    reports, resolved = _load_report_evidence(reports_dir, all_episodes)
    complete = [episode for episode in all_episodes if episode.has_hdf5]
    unresolved_complete = [episode for episode in complete if episode.key not in resolved]
    if unresolved_complete:
        raise ReorganizationError(
            "complete episodes without replay results: "
            + ", ".join(str(item.source) for item in unresolved_complete)
        )
    passed = [episode for episode in complete if resolved[episode.key].status == "passed"]
    if not passed:
        raise ReorganizationError("no replay-passed episodes were found")
    categories = _task_categories(repo_root, {episode.task for episode in passed})
    selected: list[SelectedEpisode] = []
    target_keys: set[Path] = set()
    for raw in passed:
        evidence = resolved[raw.key]
        _validate_pass_evidence(raw, evidence)
        _validate_episode_files(raw)
        frames = validate_episode_hdf5(raw.source / "ep_demo.hdf5", expected_task=raw.task)
        item = SelectedEpisode(
            raw=raw,
            category=categories[raw.task],
            evidence=evidence,
            frame_count=frames,
        )
        if item.target_relative in target_keys:
            raise ReorganizationError(f"target collision: {item.target_relative}")
        target_keys.add(item.target_relative)
        selected.append(item)
    selected.sort(key=lambda item: item.target_relative.as_posix())
    return Plan(
        root=root,
        reports_dir=reports_dir,
        reports=reports,
        all_episodes=all_episodes,
        selected=tuple(selected),
        resolved=resolved,
        source_snapshot=source_snapshot,
    )


def make_outputs(
    root: Path,
    *,
    run_id: str,
    backup_root: Path | None,
    manifest: Path | None,
) -> Outputs:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ReorganizationError(f"unsafe run id: {run_id!r}")
    partial = root.parent / f".{root.name}.organize-{run_id}.partial"
    backup = (
        backup_root.expanduser().resolve(strict=False)
        if backup_root is not None
        else root.parent / f"{root.name}.backup-{run_id}"
    )
    journal = (
        manifest.expanduser().resolve(strict=False)
        if manifest is not None
        else root.parent / f"{root.name}.organize-{run_id}.json"
    )
    return Outputs(partial_root=partial, backup_root=backup, manifest=journal)


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            raise ReorganizationError(f"cannot find existing parent for {path}")
        current = current.parent
    return current


def _preflight_outputs(plan: Plan, outputs: Outputs) -> None:
    paths = {plan.root, outputs.partial_root, outputs.backup_root, outputs.manifest}
    if len(paths) != 4:
        raise ReorganizationError("root, partial, backup, and manifest paths must differ")
    for output in (outputs.partial_root, outputs.backup_root, outputs.manifest):
        if _path_is_within(output, plan.root):
            raise ReorganizationError(f"output must be outside active root: {output}")
        if output.exists() or output.is_symlink():
            raise ReorganizationError(f"refusing to overwrite output: {output}")
        parent = _nearest_existing_parent(output.parent)
        if os.stat(parent).st_dev != os.stat(plan.root).st_dev:
            raise ReorganizationError(f"output is not on source filesystem: {output}")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _selection_digest(selected: Sequence[SelectedEpisode]) -> str:
    lines = []
    for item in selected:
        lines.append(
            "\t".join(
                (
                    item.raw.task,
                    item.raw.source_batch.name,
                    item.raw.source.name,
                    str(item.raw.source),
                )
            )
        )
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _status_counts(plan: Plan) -> dict[str, int]:
    counts: dict[str, int] = {}
    for episode in plan.all_episodes:
        evidence = plan.resolved.get(episode.key)
        status = evidence.status if evidence is not None else "unjudged"
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _hash_selected(plan: Plan) -> tuple[list[dict[str, Any]], int, int]:
    entries: list[dict[str, Any]] = []
    total_files = 0
    total_bytes = 0
    for item in plan.selected:
        inventory = _regular_file_inventory(item.raw.source)
        files = []
        for relative, (size, mtime_ns) in sorted(inventory.items()):
            files.append(
                {
                    "relative_path": relative,
                    "size": size,
                    "mtime_ns": mtime_ns,
                    "sha256": _sha256(item.raw.source / relative),
                }
            )
            total_files += 1
            total_bytes += size
        result = item.evidence.result
        entries.append(
            {
                "task": item.raw.task,
                "category": item.category,
                "episode": item.raw.source.name,
                "source_batch": str(item.raw.source_batch),
                "source": str(item.raw.source),
                "target_relative": item.target_relative.as_posix(),
                "frame_count": item.frame_count,
                "move_status": "pending",
                "replay": {
                    "status": item.evidence.status,
                    "success_mode": result.get("success_mode"),
                    "any_success": result.get("any_success"),
                    "final_success": result.get("final_success"),
                    "state_source": result.get("state_source"),
                    "completed_at": item.evidence.completed_at.isoformat(),
                    "report": str(item.evidence.report),
                    "report_sha256": item.evidence.report_sha256,
                },
                "files": files,
            }
        )
    return entries, total_files, total_bytes


def _manifest_payload(
    plan: Plan,
    outputs: Outputs,
    episode_entries: list[dict[str, Any]],
    total_files: int,
    total_bytes: int,
) -> dict[str, Any]:
    complete = sum(episode.has_hdf5 for episode in plan.all_episodes)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "planned",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "root": str(plan.root),
        "reports_dir": str(plan.reports_dir),
        "reports_used": [str(path) for path in plan.reports],
        "partial_root": str(outputs.partial_root),
        "backup_root": str(outputs.backup_root),
        "manifest": str(outputs.manifest),
        "source_summary": {
            "episode_directories": len(plan.all_episodes),
            "complete_hdf5": complete,
            "without_hdf5": len(plan.all_episodes) - complete,
            "resolved_status_counts": _status_counts(plan),
        },
        "selection": {
            "rule": (
                "latest exact-path raw replay result is passed, success_mode=any, "
                "success=true, any_success=true, state_source=raw_npz"
            ),
            "episode_count": len(plan.selected),
            "regular_file_count": total_files,
            "bytes": total_bytes,
            "task_counts": plan.task_counts,
            "selection_tsv_sha256": _selection_digest(plan.selected),
        },
        "episodes": episode_entries,
    }


def _verify_target_episode(target: Path, entry: dict[str, Any]) -> None:
    expected = {
        item["relative_path"]: (item["size"], item["mtime_ns"], item["sha256"])
        for item in entry["files"]
    }
    actual = _regular_file_inventory(target)
    expected_stat = {name: (value[0], value[1]) for name, value in expected.items()}
    if actual != expected_stat:
        raise ReorganizationError(f"moved inventory changed for {target}")
    for relative, (_, _, expected_hash) in sorted(expected.items()):
        if _sha256(target / relative) != expected_hash:
            raise ReorganizationError(f"SHA-256 changed after move: {target / relative}")
    frames = validate_episode_hdf5(
        target / "ep_demo.hdf5", expected_task=entry["task"]
    )
    if frames != entry["frame_count"]:
        raise ReorganizationError(f"frame count changed after move: {target}")


def _validate_partial(outputs: Outputs, manifest: dict[str, Any]) -> None:
    expected = {entry["target_relative"] for entry in manifest["episodes"]}
    actual: set[str] = set()
    immediate = {path.name for path in outputs.partial_root.iterdir()}
    if immediate != {"atomic_tasks", "composite_tasks"}:
        raise ReorganizationError(f"unexpected staging categories: {sorted(immediate)}")
    if any(outputs.partial_root.rglob("demo.hdf5")):
        raise ReorganizationError("aggregate demo.hdf5 leaked into staging")
    for category in ("atomic_tasks", "composite_tasks"):
        category_dir = outputs.partial_root / category
        for task_dir in category_dir.iterdir():
            if task_dir.is_symlink() or not task_dir.is_dir() or not TASK_PATTERN.fullmatch(task_dir.name):
                raise ReorganizationError(f"unexpected task entry: {task_dir}")
            children = list(task_dir.iterdir())
            if len(children) != 1 or children[0].name != "episodes" or not children[0].is_dir():
                raise ReorganizationError(f"unexpected task layout: {task_dir}")
            for episode_dir in children[0].iterdir():
                if episode_dir.is_symlink() or not episode_dir.is_dir():
                    raise ReorganizationError(f"unexpected episode entry: {episode_dir}")
                actual.add(episode_dir.relative_to(outputs.partial_root).as_posix())
    if actual != expected:
        raise ReorganizationError(
            f"staging episode set changed: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _remove_empty_tree(path: Path) -> None:
    if not path.exists():
        return
    for current_root, _directory_names, file_names in os.walk(path, topdown=False):
        if file_names:
            raise ReorganizationError(f"refusing to remove non-empty staging path: {current_root}")
        Path(current_root).rmdir()


def _rollback_moves(
    plan: Plan,
    outputs: Outputs,
    manifest: dict[str, Any],
) -> None:
    entry_by_target = {entry["target_relative"]: entry for entry in manifest["episodes"]}
    selected_by_target = {item.target_relative.as_posix(): item for item in plan.selected}
    for relative in reversed([entry["target_relative"] for entry in manifest["episodes"]]):
        entry = entry_by_target[relative]
        if entry["move_status"] not in {"moving", "moved"}:
            continue
        item = selected_by_target[relative]
        current = outputs.partial_root / relative
        if item.raw.source.is_dir() and not current.exists():
            entry["move_status"] = "rolled_back"
            continue
        if not current.is_dir() or item.raw.source.exists():
            raise ReorganizationError(
                f"cannot automatically roll back {current} to {item.raw.source}"
            )
        item.raw.source.parent.mkdir(parents=True, exist_ok=True)
        os.rename(current, item.raw.source)
        entry["move_status"] = "rolled_back"
    _remove_empty_tree(outputs.partial_root)


def execute(plan: Plan, outputs: Outputs) -> Path:
    _preflight_outputs(plan, outputs)
    print("Hashing selected episode files before migration...", flush=True)
    episode_entries, total_files, total_bytes = _hash_selected(plan)
    if _regular_file_inventory(plan.root) != plan.source_snapshot:
        raise ReorganizationError("source raw tree changed during preflight")
    manifest = _manifest_payload(plan, outputs, episode_entries, total_files, total_bytes)
    _atomic_write_json(outputs.manifest, manifest)
    outputs.partial_root.mkdir(mode=0o755)
    (outputs.partial_root / "atomic_tasks").mkdir()
    (outputs.partial_root / "composite_tasks").mkdir()
    _fsync_dir(outputs.partial_root)
    manifest["status"] = "moving_passed_episodes"
    _atomic_write_json(outputs.manifest, manifest)

    selected_by_target = {item.target_relative.as_posix(): item for item in plan.selected}
    try:
        for index, entry in enumerate(manifest["episodes"]):
            item = selected_by_target[entry["target_relative"]]
            target = outputs.partial_root / item.target_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            entry["move_status"] = "moving"
            os.rename(item.raw.source, target)
            entry["move_status"] = "moved"
            _fsync_dir(item.raw.source.parent)
            _fsync_dir(target.parent)
            manifest["moved_episode_count"] = index + 1
            _atomic_write_json(outputs.manifest, manifest)

        manifest["status"] = "verifying_staging"
        _atomic_write_json(outputs.manifest, manifest)
        for index, entry in enumerate(manifest["episodes"]):
            target = outputs.partial_root / entry["target_relative"]
            _verify_target_episode(target, entry)
            entry["move_status"] = "verified"
            manifest["verified_episode_count"] = index + 1
            _atomic_write_json(outputs.manifest, manifest)
        _validate_partial(outputs, manifest)
        manifest["status"] = "ready_to_switch"
        _atomic_write_json(outputs.manifest, manifest)
    except (Exception, KeyboardInterrupt) as exc:
        for entry in manifest["episodes"]:
            if entry["move_status"] == "verified":
                entry["move_status"] = "moved"
        try:
            _rollback_moves(plan, outputs, manifest)
            manifest["status"] = "staging_failed_rolled_back"
        except Exception as rollback_exc:
            manifest["status"] = "staging_failed_manual_recovery_required"
            manifest["rollback_error"] = f"{type(rollback_exc).__name__}: {rollback_exc}"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _atomic_write_json(outputs.manifest, manifest)
        raise

    outputs.backup_root.parent.mkdir(parents=True, exist_ok=True)
    published = False
    try:
        os.rename(plan.root, outputs.backup_root)
        _fsync_dir(plan.root.parent)
        _fsync_dir(outputs.backup_root.parent)
        manifest["status"] = "source_root_backed_up"
        _atomic_write_json(outputs.manifest, manifest)
        os.rename(outputs.partial_root, plan.root)
        published = True
        _fsync_dir(plan.root.parent)
    except (Exception, KeyboardInterrupt) as exc:
        if published:
            manifest["status"] = "published_with_directory_fsync_error"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            manifest["published_root"] = str(plan.root)
            manifest["preserved_nonpassed_backup"] = str(outputs.backup_root)
            _atomic_write_json(outputs.manifest, manifest)
            raise
        recovery_error: str | None = None
        try:
            if not plan.root.exists() and outputs.backup_root.exists():
                os.rename(outputs.backup_root, plan.root)
            if outputs.partial_root.exists():
                for entry in manifest["episodes"]:
                    if entry["move_status"] == "verified":
                        entry["move_status"] = "moved"
                _rollback_moves(plan, outputs, manifest)
        except Exception as recovery_exc:
            recovery_error = f"{type(recovery_exc).__name__}: {recovery_exc}"
        manifest["status"] = (
            "switch_failed_rolled_back"
            if recovery_error is None
            else "switch_failed_manual_recovery_required"
        )
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        if recovery_error is not None:
            manifest["recovery_error"] = recovery_error
        _atomic_write_json(outputs.manifest, manifest)
        raise

    manifest["status"] = "complete"
    manifest["completed_at"] = dt.datetime.now().astimezone().isoformat()
    manifest["published_root"] = str(plan.root)
    manifest["preserved_nonpassed_backup"] = str(outputs.backup_root)
    for entry in manifest["episodes"]:
        entry["move_status"] = "published"
    _atomic_write_json(outputs.manifest, manifest)
    return outputs.manifest


def reconcile_late_passed_episode(
    *,
    root: Path,
    reports_dir: Path,
    repo_root: Path,
    manifest_path: Path,
    source: Path,
    task: str,
) -> Path:
    """Add one replay-passed episode that appeared in the backup after planning."""

    root = _real_directory(root, label="organized raw root")
    reports_dir = _real_directory(reports_dir, label="reports directory")
    repo_root = _real_directory(repo_root, label="repository root")
    source = _real_directory(source, label="late episode source")
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ReorganizationError("reconciliation requires a complete manifest")
    if Path(manifest.get("root", "")).resolve(strict=False) != root:
        raise ReorganizationError("manifest root does not match organized root")
    backup = Path(manifest.get("backup_root", "")).resolve(strict=True)
    try:
        backup_relative = source.relative_to(backup)
    except ValueError as exc:
        raise ReorganizationError(f"late episode is outside manifest backup: {source}") from exc
    original = root / backup_relative
    original_batch = original
    while original_batch.parent != root:
        original_batch = original_batch.parent
    if not EPISODE_PATTERN.fullmatch(source.name):
        raise ReorganizationError(f"unsafe episode name: {source.name}")

    fake_original = RawEpisode(
        task=task,
        source_batch=original_batch,
        source=original,
        has_hdf5=True,
    )
    _reports, resolved = _load_report_evidence(reports_dir, (fake_original,))
    evidence = resolved.get(fake_original.key)
    if evidence is None or evidence.status != "passed":
        raise ReorganizationError(f"late episode has no latest passed replay: {task}/{source.name}")
    _validate_pass_evidence(fake_original, evidence)

    actual = RawEpisode(
        task=task,
        source_batch=source.parents[1],
        source=source,
        has_hdf5=(source / "ep_demo.hdf5").is_file(),
    )
    if not actual.has_hdf5:
        raise ReorganizationError(f"late episode has no ep_demo.hdf5: {source}")
    _validate_episode_files(actual)
    frame_count = validate_episode_hdf5(source / "ep_demo.hdf5", expected_task=task)
    category = _task_categories(repo_root, {task})[task]
    target_relative = (
        Path(f"{category}_tasks") / f"{task}_sonic" / "episodes" / source.name
    )
    target = root / target_relative
    if target.exists() or target.is_symlink():
        raise ReorganizationError(f"late episode target already exists: {target}")
    if any(
        entry.get("task") == task and entry.get("episode") == source.name
        for entry in manifest.get("episodes", [])
    ):
        raise ReorganizationError(f"late episode is already in manifest: {task}/{source.name}")

    inventory = _regular_file_inventory(source)
    files = [
        {
            "relative_path": relative,
            "size": size,
            "mtime_ns": mtime_ns,
            "sha256": _sha256(source / relative),
        }
        for relative, (size, mtime_ns) in sorted(inventory.items())
    ]
    result = evidence.result
    entry = {
        "task": task,
        "category": category,
        "episode": source.name,
        "source_batch": str(original_batch),
        "source": str(original),
        "recovered_from": str(source),
        "target_relative": target_relative.as_posix(),
        "frame_count": frame_count,
        "move_status": "pending",
        "replay": {
            "status": evidence.status,
            "success_mode": result.get("success_mode"),
            "any_success": result.get("any_success"),
            "final_success": result.get("final_success"),
            "state_source": result.get("state_source"),
            "completed_at": evidence.completed_at.isoformat(),
            "report": str(evidence.report),
            "report_sha256": evidence.report_sha256,
        },
        "files": files,
    }
    manifest["status"] = "reconciling_late_episode"
    manifest["pending_reconciliation"] = {
        "source": str(source),
        "target": str(target),
        "task": task,
        "episode": source.name,
    }
    _atomic_write_json(manifest_path, manifest)

    moved = False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, target)
        moved = True
        _fsync_dir(source.parent)
        _fsync_dir(target.parent)
        _verify_target_episode(target, entry)
    except (Exception, KeyboardInterrupt) as exc:
        rollback_error: str | None = None
        if moved:
            try:
                os.rename(target, source)
            except Exception as rollback_exc:
                rollback_error = f"{type(rollback_exc).__name__}: {rollback_exc}"
        manifest["status"] = (
            "late_reconciliation_failed_rolled_back"
            if rollback_error is None
            else "late_reconciliation_failed_manual_recovery_required"
        )
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        if rollback_error is not None:
            manifest["rollback_error"] = rollback_error
        _atomic_write_json(manifest_path, manifest)
        raise

    entry["move_status"] = "published"
    manifest["episodes"].append(entry)
    manifest["episodes"].sort(
        key=lambda item: (item["category"], item["task"], item["episode"])
    )
    selection = manifest["selection"]
    selection["episode_count"] += 1
    selection["regular_file_count"] += len(files)
    selection["bytes"] += sum(item["size"] for item in files)
    selection["task_counts"][task] = selection["task_counts"].get(task, 0) + 1
    lines = [
        "\t".join(
            (
                item["task"],
                Path(item["source_batch"]).name,
                item["episode"],
                item["source"],
            )
        )
        for item in manifest["episodes"]
    ]
    selection["selection_tsv_sha256"] = hashlib.sha256(
        ("\n".join(lines) + "\n").encode("utf-8")
    ).hexdigest()
    source_summary = manifest["source_summary"]
    source_summary["episode_directories"] += 1
    source_summary["complete_hdf5"] += 1
    source_summary["resolved_status_counts"]["passed"] += 1
    manifest.setdefault("post_switch_reconciliation", []).append(
        {
            "reason": "episode appeared in preserved backup after initial plan snapshot",
            "task": task,
            "episode": source.name,
            "recovered_from": str(source),
            "target": str(target),
            "completed_at": dt.datetime.now().astimezone().isoformat(),
        }
    )
    manifest.pop("pending_reconciliation", None)
    manifest.pop("error", None)
    manifest["status"] = "complete"
    manifest["updated_at"] = dt.datetime.now().astimezone().isoformat()
    _atomic_write_json(manifest_path, manifest)
    return manifest_path


def _print_plan(plan: Plan, outputs: Outputs) -> None:
    complete = sum(episode.has_hdf5 for episode in plan.all_episodes)
    print("SONIC replay-success organization dry-run")
    print(f"  root: {plan.root}")
    print(f"  raw episode directories: {len(plan.all_episodes)}")
    print(f"  complete ep_demo.hdf5: {complete}")
    print(f"  replay-passed selected: {len(plan.selected)}")
    print(f"  status counts: {_status_counts(plan)}")
    print(f"  selection TSV SHA-256: {_selection_digest(plan.selected)}")
    print("  task counts:")
    for task, count in plan.task_counts.items():
        print(f"    composite_tasks/{task}_sonic: {count}")
    print(f"  partial root: {outputs.partial_root}")
    print(f"  recoverable backup: {outputs.backup_root}")
    print(f"  manifest: {outputs.manifest}")
    print("No files were changed. Re-run with --apply to execute.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id")
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--reconcile-episode",
        type=Path,
        help="Move one late-discovered passed episode from the preserved backup.",
    )
    parser.add_argument("--reconcile-task")
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_id = args.run_id or dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    try:
        if args.reconcile_episode is not None:
            if not args.apply:
                raise ReorganizationError("--reconcile-episode requires --apply")
            if args.manifest is None or not args.reconcile_task:
                raise ReorganizationError(
                    "--reconcile-episode requires --manifest and --reconcile-task"
                )
            result = reconcile_late_passed_episode(
                root=args.root,
                reports_dir=args.reports_dir,
                repo_root=args.repo_root,
                manifest_path=args.manifest,
                source=args.reconcile_episode,
                task=args.reconcile_task,
            )
            print(f"Late episode reconciliation complete. Manifest: {result}")
            return 0
        plan = build_plan(args.root, args.reports_dir, args.repo_root)
        outputs = make_outputs(
            plan.root,
            run_id=run_id,
            backup_root=args.backup_root,
            manifest=args.manifest,
        )
        _preflight_outputs(plan, outputs)
        if not args.apply:
            _print_plan(plan, outputs)
            return 0
        result = execute(plan, outputs)
    except ReorganizationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (OSError, KeyboardInterrupt) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Organization complete. Manifest: {result}")
    print(f"Non-passed source data preserved at: {outputs.backup_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
