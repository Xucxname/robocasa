import hashlib
import json
from pathlib import Path

import h5py
import pytest

from robocasa.scripts import reorganize_sonic_raw
from robocasa.scripts.reorganize_sonic_raw import (
    OutputPaths,
    ReorganizationError,
    build_plan,
    execute_plan,
    main,
)


def _write_episode(run_dir: Path, episode_name: str, *, task: str) -> Path:
    episode_dir = run_dir / "episodes" / episode_name
    episode_dir.mkdir(parents=True)
    ep_meta = json.dumps({"lang": f"Do {task}", "layout_id": 1})
    model_xml = "<mujoco model='test'/>"
    (episode_dir / "ep_meta.json").write_text(ep_meta, encoding="utf-8")
    (episode_dir / "model.xml").write_text(model_xml, encoding="utf-8")
    (episode_dir / "state_1.npz").write_bytes(b"raw-state")

    hdf5_path = episode_dir / "ep_demo.hdf5"
    with h5py.File(hdf5_path, "w") as hdf5_file:
        data = hdf5_file.create_group("data")
        data.attrs["env"] = task
        data.attrs["env_args"] = json.dumps(
            {"env_name": task, "env_kwargs": {}}
        )
        data.attrs["sonic_gains"] = json.dumps({"body": [[1.0], [0.1]]})
        data.attrs["sonic_runtime"] = json.dumps(
            {"control_freq": 200, "sim_dt": 0.005}
        )
        data.attrs["total"] = 2
        demo = data.create_group("demo_1")
        demo.attrs["ep_meta"] = ep_meta
        demo.attrs["model_file"] = model_xml
        demo.attrs["num_samples"] = 2
        demo.create_dataset("actions", data=[[0.0, 1.0], [2.0, 3.0]])
        demo.create_dataset("states", data=[[4.0, 5.0], [6.0, 7.0]])
        demo.create_dataset(
            "states_integration", data=[[8.0, 9.0], [10.0, 11.0]]
        )
    return episode_dir


def _make_run(parent: Path, timestamp: str, task: str) -> Path:
    run_dir = parent / f"{timestamp}_{task}_sonic"
    (run_dir / "episodes").mkdir(parents=True)
    return run_dir


def _outputs(tmp_path: Path, name: str = "test") -> OutputPaths:
    return OutputPaths(
        partial_root=tmp_path / f".{name}.partial",
        backup_root=tmp_path / f"{name}.backup",
        manifest_path=tmp_path / f"{name}.manifest.json",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_plan_groups_valid_episodes_and_excludes_non_hdf5(tmp_path):
    root = tmp_path / "sonic_raw"
    root.mkdir()
    run = _make_run(root, "2026-08-13-10-00-00", "TaskA")
    valid = _write_episode(run, "ep_100", task="TaskA")
    invalid = run / "episodes" / "ep_101"
    invalid.mkdir()
    (invalid / "state_1.npz").write_bytes(b"incomplete")
    (run / "demo.hdf5").write_bytes(b"batch aggregate is ignored")

    plan = build_plan(root)

    assert plan.task_counts == {"TaskA": 1}
    assert plan.episodes[0].episode_dir == valid
    assert plan.episodes[0].target_relative_path == Path("TaskA_sonic/ep_100")
    assert [entry.episode_dir for entry in plan.excluded_episodes] == [invalid]
    assert list(plan.excluded_batch_hdf5) == [run / "demo.hdf5"]


def test_build_plan_rejects_cross_run_episode_collision(tmp_path):
    root = tmp_path / "sonic_raw"
    root.mkdir()
    first = _make_run(root, "2026-08-13-10-00-00", "TaskA")
    second = _make_run(root, "2026-08-13-11-00-00", "TaskA")
    _write_episode(first, "ep_same", task="TaskA")
    _write_episode(second, "ep_same", task="TaskA")

    with pytest.raises(ReorganizationError, match="target collision"):
        build_plan(root)


def test_build_plan_rejects_hdf5_task_metadata_mismatch(tmp_path):
    root = tmp_path / "sonic_raw"
    root.mkdir()
    run = _make_run(root, "2026-08-13-10-00-00", "TaskA")
    _write_episode(run, "ep_100", task="DifferentTask")

    with pytest.raises(ReorganizationError, match="does not match HDF5 env"):
        build_plan(root)


def test_default_cli_is_dry_run_and_writes_nothing(tmp_path, capsys):
    root = tmp_path / "sonic_raw"
    root.mkdir()
    run = _make_run(root, "2026-08-13-10-00-00", "TaskA")
    _write_episode(run, "ep_100", task="TaskA")

    assert main(["--root", str(root)]) == 0

    assert run.exists()
    assert list(tmp_path.glob("*.json")) == []
    assert list(tmp_path.glob("*.partial")) == []
    assert "No files were changed" in capsys.readouterr().out


def test_apply_copies_extra_runs_and_atomically_preserves_old_root(tmp_path):
    root = tmp_path / "sonic_raw"
    root.mkdir()
    run = _make_run(root, "2026-08-13-10-00-00", "TaskA")
    source_episode = _write_episode(run, "ep_100", task="TaskA")
    (run / "demo.hdf5").write_bytes(b"do not copy")

    legacy = root / "LoadDishwasher_sonic"
    legacy.mkdir()
    (legacy / "demo.hdf5").symlink_to("/missing/legacy/demo.hdf5")

    extras = tmp_path / "recoverable"
    extras.mkdir()
    extra_run = _make_run(extras, "2026-08-13-11-00-00", "TaskB")
    extra_episode = _write_episode(extra_run, "ep_200", task="TaskB")

    plan = build_plan(root, extra_run_dirs=[extra_run])
    outputs = _outputs(tmp_path)
    source_hash = _sha256(source_episode / "ep_demo.hdf5")
    extra_hash = _sha256(extra_episode / "ep_demo.hdf5")

    manifest_path = execute_plan(plan, outputs)

    assert manifest_path == outputs.manifest_path
    assert (root / "TaskA_sonic" / "ep_100" / "ep_demo.hdf5").is_file()
    assert (root / "TaskB_sonic" / "ep_200" / "ep_demo.hdf5").is_file()
    assert _sha256(root / "TaskA_sonic" / "ep_100" / "ep_demo.hdf5") == source_hash
    assert _sha256(root / "TaskB_sonic" / "ep_200" / "ep_demo.hdf5") == extra_hash
    assert not list(root.rglob("demo.hdf5"))

    assert (outputs.backup_root / run.name / "demo.hdf5").is_file()
    assert (outputs.backup_root / "LoadDishwasher_sonic" / "demo.hdf5").is_symlink()
    assert (extra_run / "episodes" / "ep_200" / "ep_demo.hdf5").is_file()
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["task_counts"] == {"TaskA": 1, "TaskB": 1}
    assert all(entry["copy_status"] == "verified" for entry in manifest["episodes"])
    assert all(entry["files"] for entry in manifest["episodes"])


def test_switch_failure_rolls_original_root_back(tmp_path, monkeypatch):
    root = tmp_path / "sonic_raw"
    root.mkdir()
    run = _make_run(root, "2026-08-13-10-00-00", "TaskA")
    _write_episode(run, "ep_100", task="TaskA")
    plan = build_plan(root)
    outputs = _outputs(tmp_path)
    real_rename = reorganize_sonic_raw._rename

    def fail_partial_publish(source: Path, target: Path) -> None:
        if source == outputs.partial_root:
            raise OSError("injected publish failure")
        real_rename(source, target)

    monkeypatch.setattr(reorganize_sonic_raw, "_rename", fail_partial_publish)

    with pytest.raises(OSError, match="injected publish failure"):
        execute_plan(plan, outputs)

    assert root.is_dir()
    assert (root / run.name / "episodes" / "ep_100").is_dir()
    assert not outputs.backup_root.exists()
    assert outputs.partial_root.is_dir()
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "switch_failed_rolled_back"
