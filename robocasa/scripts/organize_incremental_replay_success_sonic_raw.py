#!/usr/bin/env python3
"""Incrementally organize replay-passed SONIC raw episodes.

Only top-level timestamped collection batches are consumed.  Existing
``atomic_tasks`` and ``composite_tasks`` trees are never scanned as sources or
rewritten.  Passed episodes are moved (never copied) to::

    <root>/<category>/<Task>_sonic/episodes/<ep_*>

After the passed episodes have been verified at their destinations, each
remaining timestamped batch is moved intact to a recoverable quarantine root.
The default mode is a read-only dry-run.  ``--apply`` journals and fsyncs every
rename and automatically rolls all completed renames back if an error occurs.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any

try:
    # Keep direct CLI execution usable in the lightweight data environment;
    # importing the top-level robocasa package also imports the simulator.
    from reorganize_sonic_raw import (  # type: ignore[import-not-found]
        ReorganizationError,
        validate_episode_hdf5,
    )
except ImportError:  # imported as robocasa.scripts.* in tests or as a module
    from robocasa.scripts.reorganize_sonic_raw import (
        ReorganizationError,
        validate_episode_hdf5,
    )


RUN_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_"
    r"(?P<task>[A-Za-z][A-Za-z0-9_]*)_sonic$"
)
TASK_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
EPISODE_PATTERN = re.compile(r"^ep_[A-Za-z0-9_.-]+$")
STATE_PATTERN = re.compile(r"^state_.*\.npz$")
REPORT_STATUSES = {"complete", "complete_with_errors"}
CATEGORIES = ("atomic_tasks", "composite_tasks")
MANIFEST_SCHEMA_VERSION = 1
HASH_CHUNK_SIZE = 4 * 1024 * 1024


class OrganizationError(RuntimeError):
    """Raised when an incremental organization invariant is violated."""


@dataclasses.dataclass(frozen=True)
class FileRecord:
    relative_path: str
    size: int
    mtime_ns: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class ReplayEvidence:
    report: Path
    report_sha256: str
    report_status: str
    report_completed_at: str | None
    evaluated_at: str
    evaluated_at_source: str
    evaluated_at_utc: dt.datetime
    status: str
    state_source: str | None
    hdf5: str | None
    success: bool | None
    any_success: bool | None
    result_success_mode: str | None


@dataclasses.dataclass(frozen=True)
class SourceEpisode:
    task: str
    batch: Path
    source: Path
    has_hdf5: bool


@dataclasses.dataclass(frozen=True)
class EpisodeMove:
    task: str
    category: str
    batch: Path
    source: Path
    source_relative_path: Path
    target_relative_path: Path
    frame_count: int
    files: tuple[FileRecord, ...]
    evidence: tuple[ReplayEvidence, ...]
    target_task_preexisting: bool
    target_episodes_preexisting: bool

    @property
    def total_bytes(self) -> int:
        return sum(record.size for record in self.files)


@dataclasses.dataclass(frozen=True)
class ExcludedEpisode:
    task: str
    source: Path
    has_hdf5: bool
    decision: str
    evidence: tuple[ReplayEvidence, ...]


@dataclasses.dataclass(frozen=True)
class BatchMove:
    task: str
    source: Path
    source_relative_path: Path
    quarantine: Path


@dataclasses.dataclass(frozen=True)
class AcceptedReport:
    path: Path
    sha256: str
    status: str
    success_mode: str
    completed_at: str | None
    matched_results: int


@dataclasses.dataclass(frozen=True)
class OrganizationPlan:
    root: Path
    reports_dir: Path
    quarantine_root: Path
    episodes: tuple[EpisodeMove, ...]
    excluded_episodes: tuple[ExcludedEpisode, ...]
    batches: tuple[BatchMove, ...]
    reports: tuple[AcceptedReport, ...]

    @property
    def task_counts(self) -> dict[str, int]:
        counts = Counter(move.task for move in self.episodes)
        return dict(sorted(counts.items()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_real_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise OrganizationError(f"{label} must be a real directory: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path.absolute():
        raise OrganizationError(f"{label} path contains a symlink: {path}")
    return resolved


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and not current.is_symlink():
        if current.parent == current:
            break
        current = current.parent
    return current


def _registry_memberships(repo_root: Path) -> dict[str, set[str]]:
    registry = repo_root / "robocasa/utils/dataset_registry.py"
    memberships: dict[str, set[str]] = defaultdict(set)
    if not registry.is_file():
        return memberships
    try:
        tree = ast.parse(registry.read_text(encoding="utf-8"), filename=str(registry))
    except (OSError, SyntaxError) as exc:
        raise OrganizationError(f"cannot parse task registry {registry}: {exc}") from exc
    assignments = {
        "ATOMIC_TASK_DATASETS": "atomic_tasks",
        "COMPOSITE_TASK_DATASETS": "composite_tasks",
    }
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in assignments:
            continue
        if not isinstance(node.value, ast.Call):
            continue
        for keyword in node.value.keywords:
            if keyword.arg is not None:
                memberships[keyword.arg].add(assignments[target.id])
    return memberships


def _module_memberships(repo_root: Path, task: str) -> set[str]:
    memberships: set[str] = set()
    kitchen = repo_root / "robocasa/environments/kitchen"
    for source_leaf, category in (
        ("atomic", "atomic_tasks"),
        ("composite", "composite_tasks"),
    ):
        source_root = kitchen / source_leaf
        if not source_root.is_dir():
            continue
        for path in source_root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError) as exc:
                raise OrganizationError(f"cannot parse task module {path}: {exc}") from exc
            if any(
                isinstance(node, ast.ClassDef) and node.name == task
                for node in ast.walk(tree)
            ):
                memberships.add(category)
    return memberships


def classify_task(task: str, *, repo_root: Path) -> str:
    """Classify one task using the registry and class module path, fail closed."""

    if TASK_PATTERN.fullmatch(task) is None:
        raise OrganizationError(f"unsafe task name: {task!r}")
    memberships = set(_registry_memberships(repo_root).get(task, set()))
    memberships.update(_module_memberships(repo_root, task))
    if len(memberships) != 1:
        raise OrganizationError(
            f"task {task!r} has ambiguous or unknown category: {sorted(memberships)}"
        )
    return next(iter(memberships))


def _discover_timestamped_batches(root: Path) -> tuple[list[SourceEpisode], list[Path]]:
    episodes: list[SourceEpisode] = []
    batches: list[Path] = []
    for entry in sorted(root.iterdir(), key=lambda item: item.name):
        match = RUN_PATTERN.fullmatch(entry.name)
        if match is None:
            continue
        if entry.is_symlink() or not entry.is_dir():
            raise OrganizationError(f"timestamped batch is not a real directory: {entry}")
        batch = entry.resolve(strict=True)
        batches.append(batch)
        episodes_root = batch / "episodes"
        if episodes_root.exists() or episodes_root.is_symlink():
            if episodes_root.is_symlink() or not episodes_root.is_dir():
                raise OrganizationError(f"unsafe episodes directory: {episodes_root}")
            for episode in sorted(episodes_root.iterdir(), key=lambda item: item.name):
                if EPISODE_PATTERN.fullmatch(episode.name) is None:
                    continue
                if episode.is_symlink() or not episode.is_dir():
                    raise OrganizationError(f"unsafe episode directory: {episode}")
                hdf5 = episode / "ep_demo.hdf5"
                if hdf5.is_symlink():
                    raise OrganizationError(f"episode HDF5 is a symlink: {hdf5}")
                episodes.append(
                    SourceEpisode(
                        task=match.group("task"),
                        batch=batch,
                        source=episode.resolve(strict=True),
                        has_hdf5=hdf5.is_file(),
                    )
                )
    return episodes, batches


def _episode_inventory(episode: Path, *, include_hashes: bool = True) -> tuple[FileRecord, ...]:
    records: list[FileRecord] = []
    for current_root, directory_names, file_names in os.walk(episode, followlinks=False):
        current = Path(current_root)
        for name in directory_names:
            directory = current / name
            mode = directory.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise OrganizationError(f"episode contains unsafe directory: {directory}")
        for name in file_names:
            path = current / name
            before = path.lstat()
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                raise OrganizationError(f"episode contains unsafe file: {path}")
            digest = _sha256(path) if include_hashes else ""
            after = path.lstat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ino,
            ):
                raise OrganizationError(f"episode file changed while hashing: {path}")
            records.append(
                FileRecord(
                    relative_path=path.relative_to(episode).as_posix(),
                    size=before.st_size,
                    mtime_ns=before.st_mtime_ns,
                    sha256=digest,
                )
            )
    names = {record.relative_path for record in records}
    if not {"ep_demo.hdf5", "ep_meta.json", "model.xml"}.issubset(names):
        raise OrganizationError(f"passed episode lacks required raw files: {episode}")
    if not any(STATE_PATTERN.fullmatch(Path(name).name) for name in names):
        raise OrganizationError(f"passed episode has no raw state NPZ: {episode}")
    return tuple(sorted(records, key=lambda record: record.relative_path))


def _verify_inventory(episode: Path, expected: Sequence[FileRecord]) -> None:
    actual = _episode_inventory(episode, include_hashes=True)
    if tuple(expected) != actual:
        raise OrganizationError(f"episode file inventory changed: {episode}")


def _normalized_report_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value).expanduser().resolve(strict=False)


def _evaluation_time(
    result: Mapping[str, Any],
    document: Mapping[str, Any],
    *,
    report: Path,
) -> tuple[str, str, dt.datetime]:
    isolation = result.get("isolation")
    candidates: list[tuple[str, Any]] = []
    if isinstance(isolation, Mapping):
        candidates.append(("isolation.completed_at", isolation.get("completed_at")))
    candidates.extend(
        (
            ("result.completed_at", result.get("completed_at")),
            ("report.completed_at", document.get("completed_at")),
        )
    )
    for source, value in candidates:
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise OrganizationError(f"invalid {source} in replay report {report}")
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = dt.datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise OrganizationError(
                f"invalid {source} timestamp {value!r} in {report}"
            ) from exc
        if parsed.tzinfo is None:
            raise OrganizationError(
                f"naive {source} timestamp {value!r} in {report}"
            )
        return value, source, parsed.astimezone(dt.timezone.utc)
    raise OrganizationError(f"matched replay result has no completion time: {report}")


def _load_report_evidence(
    reports_dir: Path,
    candidates: Mapping[Path, SourceEpisode],
    *,
    success_mode: str,
) -> tuple[dict[Path, list[ReplayEvidence]], tuple[AcceptedReport, ...]]:
    evidence: dict[Path, list[ReplayEvidence]] = defaultdict(list)
    accepted: list[AcceptedReport] = []
    for report_input in sorted(reports_dir.glob("*.json"), key=lambda item: item.name):
        if report_input.is_symlink() or not report_input.is_file():
            continue
        try:
            document = json.loads(report_input.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(document, dict) or not isinstance(document.get("results"), list):
            continue
        if document.get("status") not in REPORT_STATUSES:
            continue
        if document.get("success_mode") != success_mode:
            continue
        cleanup = document.get("cleanup")
        if not isinstance(cleanup, Mapping) or cleanup.get("requested") is not False:
            continue

        report = report_input.resolve(strict=True)
        report_digest = _sha256(report)
        matched = 0
        matched_in_report: set[Path] = set()
        for result in document["results"]:
            if not isinstance(result, Mapping):
                continue
            episode_path = _normalized_report_path(result.get("episode_dir"))
            if episode_path not in candidates:
                continue
            if episode_path in matched_in_report:
                raise OrganizationError(
                    f"report repeats current source episode {episode_path}: {report}"
                )
            matched_in_report.add(episode_path)
            candidate = candidates[episode_path]
            if result.get("task") != candidate.task:
                raise OrganizationError(
                    f"report task mismatch for {episode_path}: {result.get('task')!r}"
                )
            if result.get("episode") != episode_path.name:
                raise OrganizationError(f"report episode name mismatch: {episode_path}")
            status_value = result.get("status")
            if status_value not in {"passed", "failed", "invalid", "error", "missing_hdf5"}:
                raise OrganizationError(
                    f"unsupported replay status {status_value!r} in {report}"
                )
            evaluated_at, evaluated_at_source, evaluated_at_utc = _evaluation_time(
                result, document, report=report
            )
            matched += 1
            evidence[episode_path].append(
                ReplayEvidence(
                    report=report,
                    report_sha256=report_digest,
                    report_status=str(document["status"]),
                    report_completed_at=(
                        str(document["completed_at"])
                        if document.get("completed_at") is not None
                        else None
                    ),
                    evaluated_at=evaluated_at,
                    evaluated_at_source=evaluated_at_source,
                    evaluated_at_utc=evaluated_at_utc,
                    status=str(status_value),
                    state_source=(
                        str(result["state_source"])
                        if result.get("state_source") is not None
                        else None
                    ),
                    hdf5=(str(result["hdf5"]) if result.get("hdf5") else None),
                    success=(
                        result["success"]
                        if isinstance(result.get("success"), bool)
                        else None
                    ),
                    any_success=(
                        result["any_success"]
                        if isinstance(result.get("any_success"), bool)
                        else None
                    ),
                    result_success_mode=(
                        str(result["success_mode"])
                        if result.get("success_mode") is not None
                        else None
                    ),
                )
            )
        if matched:
            accepted.append(
                AcceptedReport(
                    path=report,
                    sha256=report_digest,
                    status=str(document["status"]),
                    success_mode=success_mode,
                    completed_at=(
                        str(document["completed_at"])
                        if document.get("completed_at") is not None
                        else None
                    ),
                    matched_results=matched,
                )
            )
    return evidence, tuple(accepted)


def _latest_evidence(
    evidence: Sequence[ReplayEvidence],
    *,
    episode: SourceEpisode,
    success_mode: str,
) -> tuple[ReplayEvidence, ...]:
    if not evidence:
        return ()
    latest_time = max(item.evaluated_at_utc for item in evidence)
    latest = tuple(
        sorted(
            (item for item in evidence if item.evaluated_at_utc == latest_time),
            key=lambda item: (str(item.report), item.status),
        )
    )
    statuses = {item.status for item in latest}
    if len(statuses) != 1:
        raise OrganizationError(
            f"same-time replay status conflict for {episode.source} at "
            f"{latest[0].evaluated_at}: {sorted(statuses)}"
        )
    status_value = next(iter(statuses))
    if status_value in {"passed", "failed"}:
        expected_hdf5 = (episode.source / "ep_demo.hdf5").resolve(strict=False)
        for item in latest:
            if _normalized_report_path(item.hdf5) != expected_hdf5:
                raise OrganizationError(
                    f"latest replay HDF5 path mismatch: {episode.source}"
                )
            if not isinstance(item.state_source, str) or not item.state_source.startswith(
                "raw_npz/"
            ):
                raise OrganizationError(
                    f"unsafe latest replay state source for {episode.source}: "
                    f"{item.state_source!r}"
                )
            if item.result_success_mode != success_mode:
                raise OrganizationError(
                    f"latest replay success mode mismatch for {episode.source}: "
                    f"{item.result_success_mode!r}"
                )
    if status_value == "passed":
        for item in latest:
            if item.success is not True or item.any_success is not True:
                raise OrganizationError(
                    f"latest passed replay lacks true success flags: {episode.source}"
                )
            if success_mode != "any":
                raise OrganizationError(
                    "strict incremental passed selection currently requires success_mode=any"
                )
    return latest


def _decision(evidence: Sequence[ReplayEvidence], *, has_hdf5: bool) -> str:
    if not has_hdf5:
        return "missing_hdf5"
    statuses = {item.status for item in evidence}
    if "passed" in statuses:
        return "passed"
    if "failed" in statuses:
        return "failed"
    if statuses:
        return "nondefinitive"
    return "unreported"


def _validate_output_roots(root: Path, quarantine_root: Path) -> None:
    for category in CATEGORIES:
        _require_real_directory(root / category, label=category)
    quarantine_candidate = quarantine_root.expanduser().absolute()
    existing = _nearest_existing_parent(quarantine_candidate)
    if existing.is_symlink() or not existing.is_dir():
        raise OrganizationError(f"unsafe quarantine ancestor: {existing}")
    if existing.resolve(strict=True) != existing.absolute():
        raise OrganizationError(f"quarantine path contains a symlink: {existing}")
    root_resolved = root.resolve(strict=True)
    quarantine_resolved = quarantine_candidate.resolve(strict=False)
    if _is_relative_to(quarantine_resolved, root_resolved) or _is_relative_to(
        root_resolved, quarantine_resolved
    ):
        raise OrganizationError("root and quarantine paths must not overlap")
    if os.stat(root_resolved).st_dev != os.stat(existing).st_dev:
        raise OrganizationError("root and quarantine must be on the same filesystem")


def build_plan(
    root: Path | str,
    reports_dir: Path | str,
    quarantine_root: Path | str,
    *,
    repo_root: Path | str | None = None,
    success_mode: str = "any",
) -> OrganizationPlan:
    """Build and fully validate a read-only incremental organization plan."""

    root_path = _require_real_directory(Path(root).expanduser(), label="raw root")
    reports_path = _require_real_directory(
        Path(reports_dir).expanduser(), label="reports directory"
    )
    quarantine_path = Path(quarantine_root).expanduser().absolute()
    _validate_output_roots(root_path, quarantine_path)
    repository = (
        Path(repo_root).expanduser().resolve(strict=True)
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    if success_mode not in {"any", "final"}:
        raise OrganizationError(f"unsupported success mode: {success_mode!r}")

    source_episodes, batch_paths = _discover_timestamped_batches(root_path)
    if not batch_paths:
        raise OrganizationError("no top-level timestamped SONIC batches were found")
    candidate_by_path = {episode.source: episode for episode in source_episodes}
    if len(candidate_by_path) != len(source_episodes):
        raise OrganizationError("a source episode was discovered more than once")
    evidence_by_path, reports = _load_report_evidence(
        reports_path, candidate_by_path, success_mode=success_mode
    )

    categories = {
        task: classify_task(task, repo_root=repository)
        for task in sorted({episode.task for episode in source_episodes} | {
            RUN_PATTERN.fullmatch(batch.name).group("task")  # type: ignore[union-attr]
            for batch in batch_paths
        })
    }
    moves: list[EpisodeMove] = []
    excluded: list[ExcludedEpisode] = []
    targets: dict[Path, Path] = {}
    for source_episode in source_episodes:
        all_replay_evidence = tuple(
            sorted(
                evidence_by_path.get(source_episode.source, []),
                key=lambda item: (
                    item.evaluated_at_utc,
                    str(item.report),
                    item.status,
                ),
            )
        )
        replay_evidence = _latest_evidence(
            all_replay_evidence,
            episode=source_episode,
            success_mode=success_mode,
        )
        decision = _decision(replay_evidence, has_hdf5=source_episode.has_hdf5)
        if source_episode.has_hdf5 and decision not in {"passed", "failed"}:
            raise OrganizationError(
                f"complete HDF5 episode has no definitive latest replay result: "
                f"{source_episode.source} ({decision})"
            )
        if decision != "passed":
            excluded.append(
                ExcludedEpisode(
                    task=source_episode.task,
                    source=source_episode.source,
                    has_hdf5=source_episode.has_hdf5,
                    decision=decision,
                    evidence=replay_evidence,
                )
            )
            continue

        category = categories[source_episode.task]
        target_relative = (
            Path(category)
            / f"{source_episode.task}_sonic"
            / "episodes"
            / source_episode.source.name
        )
        target = root_path / target_relative
        if target in targets:
            raise OrganizationError(
                f"incremental target collision: {target}; sources "
                f"{targets[target]} and {source_episode.source}"
            )
        targets[target] = source_episode.source
        if target.exists() or target.is_symlink():
            raise OrganizationError(
                f"refusing to touch existing organized episode: {target}"
            )
        try:
            frame_count = validate_episode_hdf5(
                source_episode.source / "ep_demo.hdf5",
                expected_task=source_episode.task,
            )
        except ReorganizationError as exc:
            raise OrganizationError(str(exc)) from exc
        files = _episode_inventory(source_episode.source)
        task_target = root_path / category / f"{source_episode.task}_sonic"
        episodes_target = task_target / "episodes"
        if task_target.exists() and (task_target.is_symlink() or not task_target.is_dir()):
            raise OrganizationError(f"unsafe organized task path: {task_target}")
        if episodes_target.exists() and (
            episodes_target.is_symlink() or not episodes_target.is_dir()
        ):
            raise OrganizationError(f"unsafe organized episodes path: {episodes_target}")
        moves.append(
            EpisodeMove(
                task=source_episode.task,
                category=category,
                batch=source_episode.batch,
                source=source_episode.source,
                source_relative_path=source_episode.source.relative_to(root_path),
                target_relative_path=target_relative,
                frame_count=frame_count,
                files=files,
                evidence=replay_evidence,
                target_task_preexisting=task_target.is_dir(),
                target_episodes_preexisting=episodes_target.is_dir(),
            )
        )
    if not moves:
        raise OrganizationError("no strict replay-passed new episodes were found")

    batch_moves: list[BatchMove] = []
    for batch in batch_paths:
        target = quarantine_path / batch.name
        if target.exists() or target.is_symlink():
            raise OrganizationError(f"quarantine batch target already exists: {target}")
        match = RUN_PATTERN.fullmatch(batch.name)
        assert match is not None
        batch_moves.append(
            BatchMove(
                task=match.group("task"),
                source=batch,
                source_relative_path=batch.relative_to(root_path),
                quarantine=target,
            )
        )

    return OrganizationPlan(
        root=root_path,
        reports_dir=reports_path,
        quarantine_root=quarantine_path,
        episodes=tuple(sorted(moves, key=lambda item: str(item.target_relative_path))),
        excluded_episodes=tuple(sorted(excluded, key=lambda item: str(item.source))),
        batches=tuple(sorted(batch_moves, key=lambda item: item.source.name)),
        reports=reports,
    )


def _evidence_json(items: Sequence[ReplayEvidence]) -> list[dict[str, Any]]:
    return [
        {
            "report": str(item.report),
            "report_sha256": item.report_sha256,
            "report_status": item.report_status,
            "report_completed_at": item.report_completed_at,
            "evaluated_at": item.evaluated_at,
            "evaluated_at_source": item.evaluated_at_source,
            "status": item.status,
            "state_source": item.state_source,
            "hdf5": item.hdf5,
            "success": item.success,
            "any_success": item.any_success,
            "result_success_mode": item.result_success_mode,
        }
        for item in items
    ]


def _manifest_for_plan(plan: OrganizationPlan, manifest_path: Path) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "planned",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "root": str(plan.root),
        "reports_dir": str(plan.reports_dir),
        "quarantine_root": str(plan.quarantine_root),
        "manifest_path": str(manifest_path),
        "operation": "same_filesystem_directory_rename_no_copy",
        "task_counts": plan.task_counts,
        "summary": {
            "passed_to_move": len(plan.episodes),
            "excluded": len(plan.excluded_episodes),
            "batches_to_quarantine": len(plan.batches),
            "passed_bytes": sum(item.total_bytes for item in plan.episodes),
        },
        "reports": [
            {
                "path": str(item.path),
                "sha256": item.sha256,
                "status": item.status,
                "success_mode": item.success_mode,
                "completed_at": item.completed_at,
                "matched_results": item.matched_results,
            }
            for item in plan.reports
        ],
        "episodes": [
            {
                "task": item.task,
                "category": item.category,
                "source": str(item.source),
                "source_relative_path": item.source_relative_path.as_posix(),
                "target": str(plan.root / item.target_relative_path),
                "target_relative_path": item.target_relative_path.as_posix(),
                "frame_count": item.frame_count,
                "total_bytes": item.total_bytes,
                "selection_status": "passed",
                "move_status": "pending",
                "target_task_preexisting": item.target_task_preexisting,
                "target_episodes_preexisting": item.target_episodes_preexisting,
                "replay_evidence": _evidence_json(item.evidence),
                "files": [dataclasses.asdict(record) for record in item.files],
            }
            for item in plan.episodes
        ],
        "excluded_episodes": [
            {
                "task": item.task,
                "source": str(item.source),
                "has_hdf5": item.has_hdf5,
                "selection_status": item.decision,
                "replay_evidence": _evidence_json(item.evidence),
            }
            for item in plan.excluded_episodes
        ],
        "batches": [
            {
                "task": item.task,
                "source": str(item.source),
                "source_relative_path": item.source_relative_path.as_posix(),
                "quarantine": str(item.quarantine),
                "move_status": "pending",
            }
            for item in plan.batches
        ],
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename(source: Path, target: Path) -> None:
    os.rename(source, target)


def _file_records_from_manifest(entry: Mapping[str, Any]) -> tuple[FileRecord, ...]:
    records = entry.get("files")
    if not isinstance(records, list):
        raise OrganizationError("manifest episode file inventory is missing")
    return tuple(FileRecord(**record) for record in records)


def _remove_created_target_directories(manifest: Mapping[str, Any]) -> None:
    root = Path(str(manifest["root"]))
    for entry in reversed(list(manifest["episodes"])):
        target = root / str(entry["target_relative_path"])
        episodes_dir = target.parent
        task_dir = episodes_dir.parent
        if not entry.get("target_episodes_preexisting") and episodes_dir.is_dir():
            try:
                episodes_dir.rmdir()
            except OSError:
                pass
        if not entry.get("target_task_preexisting") and task_dir.is_dir():
            try:
                task_dir.rmdir()
            except OSError:
                pass


def rollback_manifest(manifest_path: Path | str) -> None:
    """Reverse completed journaled renames, preserving all episode contents."""

    path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    root = Path(str(manifest["root"]))
    quarantine_root = Path(str(manifest["quarantine_root"]))
    manifest["status"] = "rolling_back"
    _atomic_write_json(path, manifest)

    for entry in reversed(list(manifest["batches"])):
        source = Path(str(entry["source"]))
        quarantine = Path(str(entry["quarantine"]))
        if quarantine.is_dir() and not quarantine.is_symlink() and not source.exists():
            _rename(quarantine, source)
            _fsync_directory(source.parent)
            _fsync_directory(quarantine.parent)
        elif source.is_dir() and not source.is_symlink() and not quarantine.exists():
            pass
        else:
            raise OrganizationError(
                f"cannot safely roll back batch: source={source}, quarantine={quarantine}"
            )
        entry["move_status"] = "rolled_back"
        _atomic_write_json(path, manifest)

    for entry in reversed(list(manifest["episodes"])):
        source = Path(str(entry["source"]))
        target = root / str(entry["target_relative_path"])
        expected = _file_records_from_manifest(entry)
        if target.is_dir() and not target.is_symlink() and not source.exists():
            _verify_inventory(target, expected)
            source.parent.mkdir(parents=True, exist_ok=True)
            _rename(target, source)
            _fsync_directory(source.parent)
            _fsync_directory(target.parent)
        elif source.is_dir() and not source.is_symlink() and not target.exists():
            _verify_inventory(source, expected)
        else:
            raise OrganizationError(
                f"cannot safely roll back episode: source={source}, target={target}"
            )
        entry["move_status"] = "rolled_back"
        _atomic_write_json(path, manifest)

    _remove_created_target_directories(manifest)
    if quarantine_root.is_dir():
        try:
            quarantine_root.rmdir()
        except OSError:
            pass
    manifest["status"] = "rolled_back"
    manifest["rolled_back_at"] = dt.datetime.now().astimezone().isoformat()
    _atomic_write_json(path, manifest)


def execute_plan(plan: OrganizationPlan, manifest_path: Path | str) -> Path:
    """Apply an incremental plan with per-rename journaling and rollback."""

    output = Path(manifest_path).expanduser().absolute()
    if output.parent.resolve(strict=True) != plan.root.parent.resolve(strict=True):
        raise OrganizationError("manifest must be an existing sibling of raw root")
    if output.exists() or output.is_symlink():
        raise OrganizationError(f"refusing to overwrite manifest: {output}")
    if plan.quarantine_root.exists() and (
        plan.quarantine_root.is_symlink() or not plan.quarantine_root.is_dir()
    ):
        raise OrganizationError(f"unsafe quarantine root: {plan.quarantine_root}")

    manifest = _manifest_for_plan(plan, output)
    _atomic_write_json(output, manifest)
    try:
        for report in plan.reports:
            if report.path.is_symlink() or not report.path.is_file():
                raise OrganizationError(f"replay report disappeared: {report.path}")
            if _sha256(report.path) != report.sha256:
                raise OrganizationError(f"replay report changed: {report.path}")
        plan.quarantine_root.mkdir(mode=0o755, parents=True, exist_ok=True)
        for index, move in enumerate(plan.episodes):
            source = move.source
            target = plan.root / move.target_relative_path
            if target.exists() or target.is_symlink():
                raise OrganizationError(f"organized target appeared: {target}")
            _verify_inventory(source, move.files)
            try:
                validate_episode_hdf5(
                    source / "ep_demo.hdf5", expected_task=move.task
                )
            except ReorganizationError as exc:
                raise OrganizationError(str(exc)) from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            manifest["episodes"][index]["move_status"] = "rename_planned"
            _atomic_write_json(output, manifest)
            _rename(source, target)
            _fsync_directory(source.parent)
            _fsync_directory(target.parent)
            _verify_inventory(target, move.files)
            try:
                validate_episode_hdf5(
                    target / "ep_demo.hdf5", expected_task=move.task
                )
            except ReorganizationError as exc:
                raise OrganizationError(str(exc)) from exc
            manifest["episodes"][index]["move_status"] = "moved_verified"
            _atomic_write_json(output, manifest)

        for index, batch in enumerate(plan.batches):
            if not batch.source.is_dir() or batch.source.is_symlink():
                raise OrganizationError(f"source batch changed: {batch.source}")
            if batch.quarantine.exists() or batch.quarantine.is_symlink():
                raise OrganizationError(
                    f"quarantine target appeared: {batch.quarantine}"
                )
            manifest["batches"][index]["move_status"] = "rename_planned"
            _atomic_write_json(output, manifest)
            _rename(batch.source, batch.quarantine)
            _fsync_directory(batch.source.parent)
            _fsync_directory(batch.quarantine.parent)
            manifest["batches"][index]["move_status"] = "moved_verified"
            _atomic_write_json(output, manifest)

        for entry in manifest["episodes"]:
            destination = plan.root / entry["target_relative_path"]
            _verify_inventory(destination, _file_records_from_manifest(entry))
        for entry in manifest["batches"]:
            if Path(entry["source"]).exists() or not Path(entry["quarantine"]).is_dir():
                raise OrganizationError(f"batch publication verification failed: {entry}")
        manifest["status"] = "complete"
        manifest["completed_at"] = dt.datetime.now().astimezone().isoformat()
        _atomic_write_json(output, manifest)
        return output
    except BaseException as exc:
        manifest["status"] = "failed_rolling_back"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _atomic_write_json(output, manifest)
        try:
            rollback_manifest(output)
        except BaseException as rollback_exc:
            manifest = json.loads(output.read_text(encoding="utf-8"))
            manifest["status"] = "rollback_failed_manual_recovery_required"
            manifest["rollback_error"] = (
                f"{type(rollback_exc).__name__}: {rollback_exc}"
            )
            _atomic_write_json(output, manifest)
            raise OrganizationError(
                f"organization failed ({exc}); rollback also failed ({rollback_exc})"
            ) from exc
        raise


def make_manifest_path(root: Path, *, run_id: str | None = None) -> Path:
    if run_id is None:
        run_id = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) is None:
        raise OrganizationError(f"unsafe run id: {run_id!r}")
    return root.parent / f"{root.name}.incremental-replay-organize-{run_id}.json"


def _print_plan(plan: OrganizationPlan, manifest_path: Path) -> None:
    print("Incremental SONIC raw replay-success organization dry-run")
    print(f"  root: {plan.root}")
    print(f"  replay-passed episodes to move: {len(plan.episodes)}")
    print(f"  excluded episodes retained in quarantine: {len(plan.excluded_episodes)}")
    print(f"  timestamped batches to quarantine: {len(plan.batches)}")
    print(f"  manifest: {manifest_path}")
    print("  task counts:")
    for task, count in plan.task_counts.items():
        print(f"    {task}: {count}")
    print("No files were changed. Re-run with --apply to execute this plan.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--quarantine-root", type=Path, required=True)
    parser.add_argument("--success-mode", choices=("any", "final"), default="any")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = build_plan(
            args.root,
            args.reports_dir,
            args.quarantine_root,
            success_mode=args.success_mode,
        )
        manifest_path = make_manifest_path(plan.root, run_id=args.run_id)
        if not args.apply:
            _print_plan(plan, manifest_path)
            return 0
        execute_plan(plan, manifest_path)
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    except (OrganizationError, OSError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Organization complete. Manifest: {manifest_path}")
    print(f"Recoverable quarantine: {plan.quarantine_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
