import json
from pathlib import Path

import h5py
import pytest

from robocasa.scripts import organize_incremental_replay_success_sonic_raw as organizer
from robocasa.scripts.organize_incremental_replay_success_sonic_raw import (
    OrganizationError,
    build_plan,
    execute_plan,
    rollback_manifest,
)


TASK = "DryDishes"


def _make_root(tmp_path: Path) -> Path:
    root = tmp_path / "sonic_raw"
    (root / "atomic_tasks").mkdir(parents=True)
    (root / "composite_tasks").mkdir()
    return root


def _make_batch(root: Path, timestamp: str = "2026-09-01-10-00-00") -> Path:
    batch = root / f"{timestamp}_{TASK}_sonic"
    (batch / "episodes").mkdir(parents=True)
    return batch


def _write_episode(parent: Path, name: str, *, task: str = TASK) -> Path:
    episode = parent / name
    episode.mkdir(parents=True)
    ep_meta = json.dumps({"lang": f"Do {task}", "layout_id": 1})
    model_xml = "<mujoco model='test'/>"
    (episode / "ep_meta.json").write_text(ep_meta, encoding="utf-8")
    (episode / "model.xml").write_text(model_xml, encoding="utf-8")
    (episode / "state_1.npz").write_bytes(b"raw-state")
    with h5py.File(episode / "ep_demo.hdf5", "w") as hdf5_file:
        data = hdf5_file.create_group("data")
        data.attrs["env"] = task
        data.attrs["env_args"] = json.dumps({"env_name": task, "env_kwargs": {}})
        data.attrs["sonic_gains"] = json.dumps({"body": [[1.0], [0.1]]})
        data.attrs["sonic_runtime"] = json.dumps(
            {"control_freq": 200, "sim_dt": 0.005}
        )
        data.attrs["total"] = 2
        demo = data.create_group("demo_1")
        demo.attrs["ep_meta"] = ep_meta
        demo.attrs["model_file"] = model_xml
        demo.attrs["num_samples"] = 2
        demo.create_dataset("actions", data=[[0.0], [1.0]])
        demo.create_dataset("states", data=[[0.0], [1.0]])
        demo.create_dataset("states_integration", data=[[0.0], [1.0]])
    return episode


def _write_report(
    reports: Path,
    name: str,
    results: list[tuple[Path, str]],
    *,
    completed_at: str = "2026-09-01T12:00:00+08:00",
    passed_success: bool = True,
    passed_any_success: bool = True,
) -> Path:
    reports.mkdir(exist_ok=True)
    payload_results = []
    for episode, status in results:
        result = {
            "episode": episode.name,
            "episode_dir": str(episode.resolve()),
            "hdf5": (
                str((episode / "ep_demo.hdf5").resolve())
                if (episode / "ep_demo.hdf5").is_file()
                else None
            ),
            "task": TASK,
            "status": status,
        }
        if status in {"passed", "failed"}:
            result["state_source"] = "raw_npz/true_terminal_state"
            result["success_mode"] = "any"
            result["success"] = passed_success if status == "passed" else False
            result["any_success"] = (
                passed_any_success if status == "passed" else False
            )
        result["isolation"] = {"completed_at": completed_at}
        payload_results.append(result)
    path = reports / name
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete_with_errors",
                "completed_at": completed_at,
                "success_mode": "any",
                "cleanup": {"requested": False, "status": "not_started"},
                "results": payload_results,
            }
        ),
        encoding="utf-8",
    )
    return path


def _basic_dataset(tmp_path: Path):
    root = _make_root(tmp_path)
    batch = _make_batch(root)
    passed = _write_episode(batch / "episodes", "ep_passed")
    failed = _write_episode(batch / "episodes", "ep_failed")
    incomplete = batch / "episodes" / "ep_incomplete"
    incomplete.mkdir()
    (incomplete / "state_2.npz").write_bytes(b"discard")
    (batch / "demo.hdf5").write_bytes(b"legacy aggregate")
    existing = _write_episode(
        root / "composite_tasks" / f"{TASK}_sonic" / "episodes",
        "ep_existing",
    )
    reports = tmp_path / "artifacts"
    _write_report(
        reports,
        "replay.json",
        [(passed, "passed"), (failed, "failed"), (incomplete, "missing_hdf5")],
    )
    quarantine = tmp_path / "sonic_raw_quarantine"
    return root, batch, passed, failed, incomplete, existing, reports, quarantine


def test_build_plan_is_read_only_and_ignores_existing_organized_episodes(tmp_path):
    root, batch, passed, failed, incomplete, existing, reports, quarantine = (
        _basic_dataset(tmp_path)
    )

    plan = build_plan(root, reports, quarantine)

    assert [move.source for move in plan.episodes] == [passed]
    assert plan.episodes[0].target_relative_path == Path(
        f"composite_tasks/{TASK}_sonic/episodes/ep_passed"
    )
    assert {item.decision for item in plan.excluded_episodes} == {
        "failed",
        "missing_hdf5",
    }
    assert [item.source for item in plan.batches] == [batch]
    assert passed.is_dir() and failed.is_dir() and incomplete.is_dir()
    assert existing.is_dir()
    assert not quarantine.exists()


def test_apply_uses_rename_quarantines_residual_and_can_roll_back(tmp_path):
    root, batch, passed, failed, incomplete, existing, reports, quarantine = (
        _basic_dataset(tmp_path)
    )
    original_inode = (passed / "ep_demo.hdf5").stat().st_ino
    plan = build_plan(root, reports, quarantine)
    manifest_path = tmp_path / "sonic_raw.incremental-test.json"

    execute_plan(plan, manifest_path)

    target = root / f"composite_tasks/{TASK}_sonic/episodes/ep_passed"
    quarantined = quarantine / batch.name
    assert target.is_dir()
    assert (target / "ep_demo.hdf5").stat().st_ino == original_inode
    assert existing.is_dir()
    assert not batch.exists()
    assert (quarantined / "episodes" / failed.name).is_dir()
    assert (quarantined / "episodes" / incomplete.name).is_dir()
    assert (quarantined / "demo.hdf5").is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["episodes"][0]["move_status"] == "moved_verified"
    assert manifest["episodes"][0]["files"]
    assert manifest["episodes"][0]["replay_evidence"]

    rollback_manifest(manifest_path)

    assert batch.is_dir()
    assert passed.is_dir() and failed.is_dir() and incomplete.is_dir()
    assert not target.exists()
    assert existing.is_dir()
    assert not quarantined.exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == (
        "rolled_back"
    )


def test_pass_fail_conflict_fails_before_writing(tmp_path):
    root, _batch, passed, _failed, _incomplete, _existing, reports, quarantine = (
        _basic_dataset(tmp_path)
    )
    _write_report(reports, "conflict.json", [(passed, "failed")])

    with pytest.raises(OrganizationError, match="same-time replay status conflict"):
        build_plan(root, reports, quarantine)

    assert passed.is_dir()
    assert not quarantine.exists()


def test_apply_failure_automatically_restores_source(monkeypatch, tmp_path):
    root, batch, passed, failed, incomplete, existing, reports, quarantine = (
        _basic_dataset(tmp_path)
    )
    plan = build_plan(root, reports, quarantine)
    manifest_path = tmp_path / "sonic_raw.incremental-failure.json"
    real_rename = organizer._rename
    failed_once = False

    def fail_batch_once(source: Path, target: Path) -> None:
        nonlocal failed_once
        if source == batch and not failed_once:
            failed_once = True
            raise OSError("injected quarantine failure")
        real_rename(source, target)

    monkeypatch.setattr(organizer, "_rename", fail_batch_once)

    with pytest.raises(OSError, match="injected quarantine failure"):
        execute_plan(plan, manifest_path)

    assert batch.is_dir()
    assert passed.is_dir() and failed.is_dir() and incomplete.is_dir()
    assert existing.is_dir()
    assert not (
        root / f"composite_tasks/{TASK}_sonic/episodes/ep_passed"
    ).exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == (
        "rolled_back"
    )


def test_newer_episode_completion_supersedes_older_status(tmp_path):
    root, _batch, passed, _failed, _incomplete, _existing, reports, quarantine = (
        _basic_dataset(tmp_path)
    )
    _write_report(
        reports,
        "older-failure.json",
        [(passed, "failed")],
        completed_at="2026-09-01T11:00:00+08:00",
    )

    plan = build_plan(root, reports, quarantine)

    assert [move.source for move in plan.episodes] == [passed]
    assert {item.status for item in plan.episodes[0].evidence} == {"passed"}
    assert {item.evaluated_at for item in plan.episodes[0].evidence} == {
        "2026-09-01T12:00:00+08:00"
    }


def test_latest_passed_result_requires_strict_success_flags(tmp_path):
    root, _batch, passed, _failed, _incomplete, _existing, reports, quarantine = (
        _basic_dataset(tmp_path)
    )
    _write_report(
        reports,
        "newer-malformed-pass.json",
        [(passed, "passed")],
        completed_at="2026-09-01T13:00:00+08:00",
        passed_success=False,
    )

    with pytest.raises(OrganizationError, match="lacks true success flags"):
        build_plan(root, reports, quarantine)


def test_complete_hdf5_with_latest_invalid_result_fails_closed(tmp_path):
    root, _batch, _passed, failed, _incomplete, _existing, reports, quarantine = (
        _basic_dataset(tmp_path)
    )
    _write_report(
        reports,
        "newer-invalid.json",
        [(failed, "invalid")],
        completed_at="2026-09-01T13:00:00+08:00",
    )

    with pytest.raises(OrganizationError, match="no definitive latest replay"):
        build_plan(root, reports, quarantine)
