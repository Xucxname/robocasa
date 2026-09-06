import json
from pathlib import Path

import h5py
import pytest

from robocasa.scripts import organize_replay_success_sonic_raw as organizer


def _write_episode(run: Path, name: str, *, task: str) -> Path:
    episode = run / "episodes" / name
    episode.mkdir(parents=True)
    ep_meta = json.dumps({"lang": "test", "layout_id": 1})
    model = "<mujoco model='test'/>"
    (episode / "ep_meta.json").write_text(ep_meta, encoding="utf-8")
    (episode / "model.xml").write_text(model, encoding="utf-8")
    (episode / "state_1.npz").write_bytes(b"state")
    with h5py.File(episode / "ep_demo.hdf5", "w") as hdf5_file:
        data = hdf5_file.create_group("data")
        data.attrs["env"] = task
        data.attrs["env_args"] = json.dumps({"env_name": task})
        data.attrs["sonic_gains"] = json.dumps({})
        data.attrs["sonic_runtime"] = json.dumps({})
        data.attrs["total"] = 2
        demo = data.create_group("demo_1")
        demo.attrs["ep_meta"] = ep_meta
        demo.attrs["model_file"] = model
        demo.attrs["num_samples"] = 2
        demo.create_dataset("actions", data=[[0.0], [1.0]])
        demo.create_dataset("states", data=[[0.0], [1.0]])
        demo.create_dataset("states_integration", data=[[0.0], [1.0]])
    return episode


def _write_report(path: Path, results: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "completed_at": "2026-09-01T12:00:00+08:00",
                "success_mode": "any",
                "results": results,
            }
        ),
        encoding="utf-8",
    )


def _result(episode: Path, *, task: str, status: str) -> dict:
    passed = status == "passed"
    return {
        "task": task,
        "episode": episode.name,
        "episode_dir": str(episode),
        "hdf5": str(episode / "ep_demo.hdf5"),
        "status": status,
        "success": passed,
        "any_success": passed,
        "final_success": passed,
        "success_mode": "any",
        "state_source": "raw_npz/true_terminal_state",
        "isolation": {"completed_at": "2026-09-01T11:59:00+08:00"},
    }


def _fixture(tmp_path: Path):
    root = tmp_path / "sonic_raw"
    run = root / "2026-09-01-10-00-00_DryDishes_sonic"
    (run / "episodes").mkdir(parents=True)
    passed = _write_episode(run, "ep_passed", task="DryDishes")
    failed = _write_episode(run, "ep_failed", task="DryDishes")
    incomplete = run / "episodes" / "ep_incomplete"
    incomplete.mkdir()
    (incomplete / "state_1.npz").write_bytes(b"incomplete")
    (run / "demo.hdf5").write_bytes(b"aggregate")
    reports = tmp_path / "reports"
    reports.mkdir()
    _write_report(
        reports / "replay.json",
        [
            _result(passed, task="DryDishes", status="passed"),
            _result(failed, task="DryDishes", status="failed"),
            {
                "task": "DryDishes",
                "episode": incomplete.name,
                "episode_dir": str(incomplete),
                "status": "missing_hdf5",
                "isolation": {"completed_at": "2026-09-01T11:59:00+08:00"},
            },
        ],
    )
    repo_root = Path(__file__).resolve().parents[1]
    plan = organizer.build_plan(root, reports, repo_root)
    outputs = organizer.Outputs(
        partial_root=tmp_path / ".partial",
        backup_root=tmp_path / "backup",
        manifest=tmp_path / "manifest.json",
    )
    return root, run, passed, failed, incomplete, plan, outputs


def test_apply_publishes_only_passed_and_preserves_nonpassed_backup(tmp_path):
    root, run, passed, failed, incomplete, plan, outputs = _fixture(tmp_path)

    assert len(plan.selected) == 1
    assert plan.task_counts == {"DryDishes": 1}

    organizer.execute(plan, outputs)

    target = (
        root
        / "composite_tasks"
        / "DryDishes_sonic"
        / "episodes"
        / passed.name
    )
    assert (target / "ep_demo.hdf5").is_file()
    assert list((root / "atomic_tasks").iterdir()) == []
    assert not list(root.rglob("demo.hdf5"))
    assert not (root / "composite_tasks" / "DryDishes_sonic" / "episodes" / failed.name).exists()

    old_run = outputs.backup_root / run.name
    assert (old_run / "episodes" / failed.name / "ep_demo.hdf5").is_file()
    assert (old_run / "episodes" / incomplete.name / "state_1.npz").is_file()
    assert not (old_run / "episodes" / passed.name).exists()
    assert (old_run / "demo.hdf5").is_file()
    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["selection"]["episode_count"] == 1
    assert manifest["episodes"][0]["move_status"] == "published"


def test_publish_failure_restores_every_episode(tmp_path, monkeypatch):
    root, run, passed, failed, incomplete, plan, outputs = _fixture(tmp_path)
    real_rename = organizer.os.rename

    def injected_rename(source, target):
        if Path(source) == outputs.partial_root and Path(target) == root:
            raise OSError("injected publish failure")
        return real_rename(source, target)

    monkeypatch.setattr(organizer.os, "rename", injected_rename)

    with pytest.raises(OSError, match="injected publish failure"):
        organizer.execute(plan, outputs)

    assert (run / "episodes" / passed.name / "ep_demo.hdf5").is_file()
    assert (run / "episodes" / failed.name / "ep_demo.hdf5").is_file()
    assert (run / "episodes" / incomplete.name / "state_1.npz").is_file()
    assert not outputs.backup_root.exists()
    assert not outputs.partial_root.exists()
    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "switch_failed_rolled_back"


def test_reconcile_late_passed_episode_from_backup(tmp_path):
    root, run, _passed, _failed, _incomplete, plan, outputs = _fixture(tmp_path)
    organizer.execute(plan, outputs)

    backup_run = outputs.backup_root / run.name
    late = _write_episode(backup_run, "ep_late", task="DryDishes")
    original = root / run.name / "episodes" / late.name
    late_result = _result(original, task="DryDishes", status="passed")
    _write_report(tmp_path / "reports" / "late.json", [late_result])

    organizer.reconcile_late_passed_episode(
        root=root,
        reports_dir=tmp_path / "reports",
        repo_root=Path(__file__).resolve().parents[1],
        manifest_path=outputs.manifest,
        source=late,
        task="DryDishes",
    )

    target = (
        root
        / "composite_tasks"
        / "DryDishes_sonic"
        / "episodes"
        / late.name
    )
    assert (target / "ep_demo.hdf5").is_file()
    assert not late.exists()
    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["selection"]["episode_count"] == 2
    assert manifest["selection"]["task_counts"] == {"DryDishes": 2}
    assert len(manifest["post_switch_reconciliation"]) == 1
