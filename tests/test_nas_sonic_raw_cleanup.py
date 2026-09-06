import json
import os
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace

import pytest

from robocasa.scripts import nas_sonic_raw_cleanup as nas_cleanup
from robocasa.scripts.nas_sonic_raw_cleanup import (
    NasCleanupWorker,
    REMOTE_CLASSIFY_SCRIPT,
    TaskInventory,
    WorkflowError,
    atomic_write_json,
    build_initial_state,
    choose_task_placement,
    load_inventory,
    load_job_state,
    main,
    merge_successful_episodes,
    ordered_tasks,
    validate_evaluation_report,
    validate_remote_config,
    validate_remote_mutation_target,
    validate_task_name,
)


def _inventory_payload() -> dict:
    tasks = {
        "LargeTask": {
            "batch_hdf5": 1,
            "bytes": 900,
            "episode_dirs": 3,
            "episode_hdf5": 2,
            "missing_hdf5": 1,
            "regular_files": 8,
            "runs": ["2026-08-13-12-00-00_LargeTask_sonic"],
            "valid_episode_bytes": 800,
        },
        "EmptyTask": {
            "batch_hdf5": 0,
            "bytes": 20,
            "episode_dirs": 1,
            "episode_hdf5": 0,
            "missing_hdf5": 1,
            "regular_files": 2,
            "runs": ["2026-08-13-10-00-00_EmptyTask_sonic"],
            "valid_episode_bytes": 0,
        },
        "SmallTask": {
            "batch_hdf5": 1,
            "bytes": 300,
            "episode_dirs": 2,
            "episode_hdf5": 1,
            "missing_hdf5": 1,
            "regular_files": 5,
            "runs": ["2026-08-13-11-00-00_SmallTask_sonic"],
            "valid_episode_bytes": 200,
        },
    }
    return {
        "created_unix": 1_786_632_826.0,
        "root": "/volume1/share/datasets/sonic_raw",
        "schema_version": 1,
        "tasks": tasks,
        "total": {
            "batch_hdf5": 2,
            "bytes": 1_220,
            "episode_dirs": 6,
            "episode_hdf5": 3,
            "missing_hdf5": 3,
            "regular_files": 15,
            "runs": 3,
        },
    }


def _write_inventory(path: Path) -> Path:
    path.write_text(json.dumps(_inventory_payload()), encoding="utf-8")
    return path


def _write_minimum_inventory(path: Path) -> Path:
    tasks = {}
    for name, episode_count, valid_bytes, timestamp in (
        ("EmptyTask", 0, 0, "10-00-00"),
        ("UnderMinimum", 9, 90, "11-00-00"),
        ("AtMinimum", 10, 100, "12-00-00"),
    ):
        tasks[name] = {
            "batch_hdf5": 0,
            "bytes": valid_bytes,
            "episode_dirs": episode_count,
            "episode_hdf5": episode_count,
            "missing_hdf5": 0,
            "regular_files": episode_count * 4,
            "runs": [f"2026-08-13-{timestamp}_{name}_sonic"],
            "valid_episode_bytes": valid_bytes,
        }
    payload = {
        "created_unix": 1_786_632_826.0,
        "root": "/volume1/share/datasets/sonic_raw",
        "schema_version": 1,
        "tasks": tasks,
        "total": {
            "batch_hdf5": 0,
            "bytes": 190,
            "episode_dirs": 19,
            "episode_hdf5": 19,
            "missing_hdf5": 0,
            "regular_files": 76,
            "runs": 3,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_raw_episode(
    task_dir: Path,
    episode: str,
    *,
    payload: bytes = b"same",
) -> Path:
    episode_dir = task_dir / episode
    episode_dir.mkdir(parents=True)
    (episode_dir / "ep_demo.hdf5").write_bytes(b"hdf5-" + payload)
    (episode_dir / "ep_meta.json").write_bytes(b"meta-" + payload)
    (episode_dir / "model.xml").write_bytes(b"model-" + payload)
    (episode_dir / "state_1_0.npz").write_bytes(b"state-" + payload)
    return episode_dir


def _remote_config() -> dict[str, str]:
    parent = "/volume1/share/datasets"
    return {
        "raw_root": f"{parent}/sonic_raw",
        "backup_root": f"{parent}/sonic_raw.backup-20260813-225000",
        "clean_root": f"{parent}/sonic_raw_cleaned-20260813-225000",
        "rejected_root": f"{parent}/sonic_raw_replay_failed-20260813-225000",
    }


@pytest.mark.parametrize(
    "task",
    [
        "",
        ".",
        "..",
        "Task/Other",
        r"Task\Other",
        "../LoadDishwasher",
        "LoadDishwasher_sonic/ep_1",
        "Task;touch_bad",
        "Task $(bad)",
        " Task",
        "Task\nOther",
        "_HiddenTask",
    ],
)
def test_validate_task_name_rejects_paths_and_shell_syntax(task):
    with pytest.raises(WorkflowError, match="task"):
        validate_task_name(task)


def test_validate_task_name_accepts_inventory_style_identifier():
    assert validate_task_name("LoadDishwasher") == "LoadDishwasher"
    assert validate_task_name("PickPlaceCounterToStove") == (
        "PickPlaceCounterToStove"
    )
    assert validate_task_name("Task2") == "Task2"


def test_load_inventory_parses_fields_and_orders_all_tasks_smallest_first(
    tmp_path,
):
    inventory = load_inventory(_write_inventory(tmp_path / "inventory.json"))

    assert inventory.root == "/volume1/share/datasets/sonic_raw"
    assert inventory.tasks["SmallTask"].episode_hdf5 == 1
    assert inventory.tasks["SmallTask"].valid_episode_bytes == 200
    assert inventory.tasks["SmallTask"].runs == (
        "2026-08-13-11-00-00_SmallTask_sonic",
    )
    assert [task.name for task in ordered_tasks(inventory)] == [
        "EmptyTask",
        "SmallTask",
        "LargeTask",
    ]


def test_ordered_tasks_applies_selection_without_losing_size_order(tmp_path):
    inventory = load_inventory(_write_inventory(tmp_path / "inventory.json"))

    selected = ordered_tasks(
        inventory,
        requested_tasks=("LargeTask", "SmallTask"),
    )

    assert [task.name for task in selected] == ["SmallTask", "LargeTask"]
    with pytest.raises(WorkflowError, match="UnknownTask"):
        ordered_tasks(inventory, requested_tasks=("UnknownTask",))


def test_load_inventory_rejects_invalid_episode_accounting(tmp_path):
    payload = _inventory_payload()
    payload["tasks"]["SmallTask"]["missing_hdf5"] = 99
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(WorkflowError, match="SmallTask"):
        load_inventory(path)


def test_atomic_state_round_trip_and_failed_write_preserves_previous_state(
    tmp_path,
):
    inventory = load_inventory(_write_inventory(tmp_path / "inventory.json"))
    config = {"job_id": "20260813-225000", **_remote_config()}
    state_path = tmp_path / "job-state.json"
    original = build_initial_state(config, inventory)
    atomic_write_json(state_path, original)

    assert load_job_state(state_path, config) == original

    with pytest.raises((TypeError, WorkflowError)):
        atomic_write_json(state_path, {"not_json": object()})

    assert load_job_state(state_path, config) == original
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "inventory.json",
        "job-state.json",
    ]


def test_load_job_state_rejects_corruption_and_config_drift(tmp_path):
    inventory = load_inventory(_write_inventory(tmp_path / "inventory.json"))
    config = {"job_id": "20260813-225000", **_remote_config()}
    state_path = tmp_path / "job-state.json"
    atomic_write_json(state_path, build_initial_state(config, inventory))

    changed = {**config, "clean_root": config["clean_root"] + "-other"}
    with pytest.raises(WorkflowError, match="config"):
        load_job_state(state_path, changed)

    state_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(WorkflowError, match="JSON|state"):
        load_job_state(state_path, config)


@pytest.mark.parametrize(
    "changed_selection",
    [
        ["SmallTask"],
        ["SmallTask", "OtherTask"],
        ["LargeTask", "SmallTask"],
    ],
)
def test_job_state_locks_exact_ordered_selected_tasks(
    tmp_path,
    changed_selection,
):
    inventory = load_inventory(_write_inventory(tmp_path / "inventory.json"))
    config = {
        "job_id": "20260814-000000",
        "selected_tasks": ["SmallTask", "LargeTask"],
        **_remote_config(),
    }
    state_path = tmp_path / "job-state.json"
    original = build_initial_state(config, inventory)
    atomic_write_json(state_path, original)

    assert load_job_state(state_path, dict(config)) == original

    changed = {**config, "selected_tasks": changed_selection}
    with pytest.raises(WorkflowError, match="config"):
        load_job_state(state_path, changed)


def test_initial_state_marks_zero_valid_task_complete_without_replay(tmp_path):
    inventory = load_inventory(_write_inventory(tmp_path / "inventory.json"))
    config = {"job_id": "20260813-225000", **_remote_config()}

    state = build_initial_state(config, inventory)

    assert state["tasks"]["EmptyTask"]["status"] == "zero_valid"
    assert state["tasks"]["EmptyTask"]["phase"] == "complete"
    assert state["tasks"]["EmptyTask"]["source_episode_count"] == 0
    assert state["tasks"]["SmallTask"]["status"] == "pending"


def test_minimum_ten_is_explicit_in_state_and_task_selection(tmp_path, capsys):
    inventory_path = _write_minimum_inventory(tmp_path / "inventory.json")
    inventory = load_inventory(inventory_path)
    config = {
        "job_id": "20260814-000000",
        "minimum_episodes": 10,
        **_remote_config(),
    }

    state = build_initial_state(config, inventory)

    assert state["tasks"]["EmptyTask"]["status"] == "zero_valid"
    assert state["tasks"]["EmptyTask"]["phase"] == "complete"
    assert state["tasks"]["UnderMinimum"] == {
        "phase": "complete",
        "status": "below_minimum",
        "source_episode_count": 9,
        "source_valid_bytes": 90,
        "minimum_episodes": 10,
        "updated_at": state["tasks"]["UnderMinimum"]["updated_at"],
    }
    assert state["tasks"]["AtMinimum"]["status"] == "pending"

    result = main(
        [
            "--job-id",
            "20260814-000000",
            "--inventory",
            str(inventory_path),
            "--raw-root",
            "/volume1/share/datasets/sonic_raw",
            "--source-backup",
            "/volume1/share/datasets/sonic_raw.backup-20260814-000000",
            "--clean-root",
            "/volume1/share/datasets/sonic_raw_cleaned-20260814-000000",
            "--rejected-root",
            "/volume1/share/datasets/sonic_raw_replay_failed-20260814-000000",
            "--local-staging-root",
            str(tmp_path / "staging"),
            "--log-root",
            str(tmp_path / "logs"),
            "--dry-run",
        ]
    )
    dry_run = json.loads(capsys.readouterr().out)

    assert result == 0
    assert dry_run["config"]["minimum_episodes"] == 10
    assert dry_run["config"]["selected_tasks"] == ["AtMinimum"]
    assert [entry["task"] for entry in dry_run["task_order"]] == [
        "AtMinimum"
    ]


def _evaluation_report() -> dict:
    return {
        "status": "complete",
        "selection": {"tasks": ["SmallTask"], "candidate_count": 2},
        "summary": {
            "total": 2,
            "passed": 1,
            "failed": 1,
            "invalid": 0,
            "error": 0,
            "missing_hdf5": 0,
        },
        "cleanup": {
            "requested": False,
            "status": "not_started",
        },
        "results": [
            {
                "task": "SmallTask",
                "episode": "ep_1",
                "status": "passed",
                "state_source": "raw_npz/true_terminal_state",
            },
            {
                "task": "SmallTask",
                "episode": "ep_2",
                "status": "failed",
                "state_source": "raw_npz/true_terminal_state",
            },
        ],
    }


def test_complete_evaluation_with_task_failures_is_publishable_after_cleanup():
    report = _evaluation_report()

    summary = validate_evaluation_report(
        report,
        expected_task="SmallTask",
        expected_total=2,
        expected_episodes=("ep_1", "ep_2"),
    )

    assert summary == report["summary"]


def test_complete_evaluation_with_structurally_invalid_episode_is_publishable():
    report = _evaluation_report()
    report["summary"].update(passed=1, failed=0, invalid=1)
    report["results"][1] = {
        "task": "SmallTask",
        "episode": "ep_2",
        "status": "invalid",
        "error": "EvaluationError: demo has no states dataset",
    }

    summary = validate_evaluation_report(
        report,
        expected_task="SmallTask",
        expected_total=2,
        expected_episodes=("ep_1", "ep_2"),
    )

    assert summary == report["summary"]


def test_invalid_evaluation_requires_explicit_evaluation_error():
    report = _evaluation_report()
    report["summary"].update(passed=1, failed=0, invalid=1)
    report["results"][1] = {
        "task": "SmallTask",
        "episode": "ep_2",
        "status": "invalid",
        "error": "RuntimeError: transient simulator failure",
    }

    with pytest.raises(WorkflowError, match="not deterministic"):
        validate_evaluation_report(
            report,
            expected_task="SmallTask",
            expected_total=2,
            expected_episodes=("ep_1", "ep_2"),
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda report: report.update(status="running"), "complete"),
        (lambda report: report["summary"].update(error=1, failed=0), "error"),
        (
            lambda report: report["summary"].update(missing_hdf5=1, failed=0),
            "missing_hdf5",
        ),
        (lambda report: report["summary"].update(total=3), "total"),
        (lambda report: report["results"][0].update(task="OtherTask"), "task"),
        (
            lambda report: report["results"][0].update(status="unknown"),
            "status",
        ),
        (
            lambda report: report["results"][0].update(state_source="hdf5"),
            "state_source",
        ),
        (
            lambda report: report["results"][1].update(episode="ep_1"),
            "episode",
        ),
        (
            lambda report: report["summary"].update(passed=2, failed=0),
            "summary|passed|failed",
        ),
        (
            lambda report: report["cleanup"].update(requested=True),
            "cleanup",
        ),
    ],
)
def test_evaluation_report_blocks_publish_on_incomplete_or_unsafe_result(
    mutator,
    message,
):
    report = _evaluation_report()
    mutator(report)

    with pytest.raises(WorkflowError, match=message):
        validate_evaluation_report(
            report,
            expected_task="SmallTask",
            expected_total=2,
            expected_episodes=("ep_1", "ep_2"),
        )


def test_evaluation_report_episode_set_must_match_remote_manifest():
    report = _evaluation_report()

    with pytest.raises(WorkflowError, match="episode"):
        validate_evaluation_report(
            report,
            expected_task="SmallTask",
            expected_total=2,
            expected_episodes=("ep_1", "ep_other"),
        )


def test_remote_config_separates_read_only_sources_from_mutable_outputs():
    config = _remote_config()

    validated = validate_remote_config(config)

    assert str(validated["raw_root"]).endswith("/sonic_raw")
    assert str(validated["backup_root"]).endswith(
        "/sonic_raw.backup-20260813-225000"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("clean_root", "/volume1/share/datasets/sonic_raw"),
        ("clean_root", "/volume1/share/datasets/sonic_raw/cleaned"),
        (
            "rejected_root",
            "/volume1/share/datasets/sonic_raw.backup-20260813-225000/rejected",
        ),
        ("backup_root", "/volume1/share/datasets/not-the-selected-backup"),
        ("clean_root", "relative/cleaned"),
    ],
)
def test_remote_config_rejects_destinations_that_can_modify_sources(field, value):
    config = {**_remote_config(), field: value}

    with pytest.raises(WorkflowError, match=field):
        validate_remote_config(config)


def test_remote_mutation_guard_only_allows_clean_and_rejected_trees():
    config = validate_remote_config(_remote_config())

    for target in (
        config["clean_root"],
        f"{config['clean_root']}/.SmallTask_sonic.partial",
        f"{config['rejected_root']}/SmallTask_sonic/ep_2",
    ):
        assert validate_remote_mutation_target(target, config)

    forbidden = (
        config["raw_root"],
        f"{config['raw_root']}/2026-run/episodes/ep_1",
        config["backup_root"],
        f"{config['backup_root']}/2026-run/episodes/ep_1",
        str(Path(str(config["raw_root"])).parent),
        f"{config['clean_root']}/../sonic_raw/2026-run",
        "relative/path",
    )
    for target in forbidden:
        with pytest.raises(WorkflowError, match="mutation|target|path"):
            validate_remote_mutation_target(target, config)


def _execute_remote_script(script: str, payload: dict) -> None:
    try:
        exec(compile(script, "<remote-script>", "exec"), {"PAYLOAD": payload})
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise


def _classification_payload(tmp_path: Path) -> tuple[dict, Path, Path]:
    clean_root = tmp_path / "clean"
    rejected_root = tmp_path / "rejected"
    partial = clean_root / ".Task_sonic.partial-job"
    partial.mkdir(parents=True)
    payload = {
        "task": "Task",
        "passed": ["ep_passed"],
        "failed": ["ep_failed"],
        "clean_root": str(clean_root),
        "rejected_root": str(rejected_root),
        "job_id": "job",
    }
    return payload, partial, rejected_root


def _split_classification_payload(
    tmp_path: Path,
    *,
    local_passed: tuple[str, ...] = (),
    nas_passed: tuple[str, ...] = (),
    failed: tuple[str, ...] = (),
) -> tuple[dict, Path, Path, Path, Path]:
    clean_root = tmp_path / "clean"
    rejected_root = tmp_path / "rejected"
    partial = clean_root / ".Task_sonic.partial-job"
    partial.mkdir(parents=True)
    for name in (*local_passed, *nas_passed, *failed):
        (partial / name).mkdir()
    marker = clean_root / ".Task_sonic.classification-job.json"
    localized = clean_root / ".localized-Task_sonic-job"
    payload = {
        "task": "Task",
        "local_passed": list(local_passed),
        "nas_passed": list(nas_passed),
        "failed": list(failed),
        "clean_root": str(clean_root),
        "rejected_root": str(rejected_root),
        "job_id": "job",
    }
    return payload, partial, rejected_root, marker, localized


def _make_writable(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for current, directories, files in os.walk(root):
        os.chmod(current, 0o755)
        for name in directories:
            path = Path(current) / name
            if not path.is_symlink():
                os.chmod(path, 0o755)
        for name in files:
            path = Path(current) / name
            if not path.is_symlink():
                os.chmod(path, 0o644)


def test_remote_classification_is_idempotent_after_publish(tmp_path):
    payload, partial, rejected_root = _classification_payload(tmp_path)
    (partial / "ep_passed").mkdir()
    (partial / "ep_failed").mkdir()

    try:
        _execute_remote_script(REMOTE_CLASSIFY_SCRIPT, payload)
        _execute_remote_script(REMOTE_CLASSIFY_SCRIPT, payload)

        clean_names = {
            item.name
            for item in (tmp_path / "clean" / "Task_sonic").iterdir()
        }
        assert clean_names == {"ep_passed"}
        assert {
            item.name for item in (rejected_root / "Task_sonic").iterdir()
        } == {"ep_failed"}
        assert not partial.exists()
    finally:
        _make_writable(tmp_path / "clean")
        _make_writable(rejected_root)


def test_remote_classification_resumes_after_failed_episode_was_moved(tmp_path):
    payload, partial, rejected_root = _classification_payload(tmp_path)
    (partial / "ep_passed").mkdir()
    rejected_partial = rejected_root / ".Task_sonic.partial-job"
    (rejected_partial / "ep_failed").mkdir(parents=True)

    try:
        _execute_remote_script(REMOTE_CLASSIFY_SCRIPT, payload)

        assert (tmp_path / "clean" / "Task_sonic" / "ep_passed").is_dir()
        assert (rejected_root / "Task_sonic" / "ep_failed").is_dir()
        assert not partial.exists()
        assert not rejected_partial.exists()
    finally:
        _make_writable(tmp_path / "clean")
        _make_writable(rejected_root)


def test_remote_classification_handles_synology_read_only_episode_move(
    tmp_path,
    monkeypatch,
):
    payload, partial, rejected_root = _classification_payload(tmp_path)
    passed = partial / "ep_passed"
    failed = partial / "ep_failed"
    passed.mkdir()
    failed.mkdir()
    passed.chmod(0o555)
    failed.chmod(0o555)
    real_rename = os.rename
    moved_episode_modes = []

    def synology_rename(source, target):
        source_path = Path(source)
        if source_path.name.startswith("ep_"):
            mode = stat.S_IMODE(source_path.stat().st_mode)
            moved_episode_modes.append(mode)
            if not mode & stat.S_IWUSR:
                raise PermissionError(
                    "Synology requires owner-write on the moved directory"
                )
        return real_rename(source, target)

    monkeypatch.setattr(os, "rename", synology_rename)

    try:
        _execute_remote_script(REMOTE_CLASSIFY_SCRIPT, payload)

        rejected_episode = rejected_root / "Task_sonic" / "ep_failed"
        clean_episode = tmp_path / "clean" / "Task_sonic" / "ep_passed"
        assert moved_episode_modes == [0o755]
        assert stat.S_IMODE(rejected_episode.stat().st_mode) == 0o555
        assert stat.S_IMODE(clean_episode.stat().st_mode) == 0o555
    finally:
        _make_writable(tmp_path / "clean")
        _make_writable(rejected_root)


def test_remote_classification_resumes_after_chmod_before_rename_failure(
    tmp_path,
    monkeypatch,
):
    payload, partial, rejected_root = _classification_payload(tmp_path)
    (partial / "ep_passed").mkdir()
    failed = partial / "ep_failed"
    failed.mkdir()
    failed.chmod(0o555)
    real_rename = os.rename
    fail_once = True

    def interrupted_rename(source, target):
        nonlocal fail_once
        source_path = Path(source)
        if source_path.name == "ep_failed" and fail_once:
            fail_once = False
            raise OSError("injected interruption after chmod")
        return real_rename(source, target)

    monkeypatch.setattr(os, "rename", interrupted_rename)

    try:
        with pytest.raises(OSError, match="injected interruption"):
            _execute_remote_script(REMOTE_CLASSIFY_SCRIPT, payload)
        assert stat.S_IMODE(failed.stat().st_mode) == 0o755

        _execute_remote_script(REMOTE_CLASSIFY_SCRIPT, payload)

        rejected_episode = rejected_root / "Task_sonic" / "ep_failed"
        assert rejected_episode.is_dir()
        assert stat.S_IMODE(rejected_episode.stat().st_mode) == 0o555
        assert not partial.exists()
    finally:
        _make_writable(tmp_path / "clean")
        _make_writable(rejected_root)


def test_remote_classification_splits_local_nas_and_failed_episodes(tmp_path):
    payload, partial, rejected_root, marker, localized = (
        _split_classification_payload(
            tmp_path,
            local_passed=("ep_local",),
            nas_passed=("ep_nas",),
            failed=("ep_failed",),
        )
    )

    try:
        _execute_remote_script(REMOTE_CLASSIFY_SCRIPT, payload)
        _execute_remote_script(REMOTE_CLASSIFY_SCRIPT, payload)

        clean_final = tmp_path / "clean" / "Task_sonic"
        rejected_final = rejected_root / "Task_sonic"
        assert {item.name for item in clean_final.iterdir()} == {"ep_nas"}
        assert {item.name for item in rejected_final.iterdir()} == {"ep_failed"}
        assert not partial.exists()
        assert not localized.exists()
        assert json.loads(marker.read_text(encoding="utf-8")) == {
            "task": "Task",
            "job_id": "job",
            "nas_passed": ["ep_nas"],
            "local_passed": ["ep_local"],
            "failed": ["ep_failed"],
            "localized_pruned": True,
        }
    finally:
        _make_writable(tmp_path / "clean")
        _make_writable(rejected_root)


def test_local_only_classification_leaves_no_nas_clean_task(tmp_path):
    payload, partial, rejected_root, marker, localized = (
        _split_classification_payload(
            tmp_path,
            local_passed=("ep_local",),
        )
    )

    _execute_remote_script(REMOTE_CLASSIFY_SCRIPT, payload)

    assert not (tmp_path / "clean" / "Task_sonic").exists()
    assert not partial.exists()
    assert not localized.exists()
    assert marker.is_file()
    assert json.loads(marker.read_text(encoding="utf-8"))["localized_pruned"] is True
    assert not rejected_root.exists()


def test_marker_local_placement_rejects_dangling_clean_final_symlink(
    tmp_path,
):
    payload, partial, _, marker, _ = _split_classification_payload(
        tmp_path,
        local_passed=("ep_local",),
    )
    _execute_remote_script(REMOTE_CLASSIFY_SCRIPT, payload)
    assert marker.is_file()
    assert not partial.exists()
    final = tmp_path / "clean" / "Task_sonic"
    final.symlink_to(tmp_path / "missing-final", target_is_directory=True)

    with pytest.raises(
        RuntimeError,
        match="clean final or partial task path is a symlink",
    ):
        _execute_remote_script(REMOTE_CLASSIFY_SCRIPT, payload)

    assert final.is_symlink()
    assert not final.exists()


def test_marker_local_placement_rejects_dangling_clean_partial_symlink(
    tmp_path,
):
    payload, partial, _, marker, _ = _split_classification_payload(
        tmp_path,
        local_passed=("ep_local",),
    )
    _execute_remote_script(REMOTE_CLASSIFY_SCRIPT, payload)
    assert marker.is_file()
    assert not partial.exists()
    partial.symlink_to(tmp_path / "missing-partial", target_is_directory=True)

    with pytest.raises(
        RuntimeError,
        match="clean final or partial task path is a symlink",
    ):
        _execute_remote_script(REMOTE_CLASSIFY_SCRIPT, payload)

    assert partial.is_symlink()
    assert not partial.exists()


@pytest.mark.parametrize("crash_after_marker_rename", [False, True])
def test_local_classification_recovers_around_atomic_marker_rename(
    tmp_path,
    monkeypatch,
    crash_after_marker_rename,
):
    payload, partial, rejected_root, marker, localized = (
        _split_classification_payload(
            tmp_path,
            local_passed=("ep_local",),
        )
    )
    real_rename = os.rename
    fail_once = True

    def interrupted_marker_rename(source, target):
        nonlocal fail_once
        if Path(target) == marker and fail_once:
            fail_once = False
            if crash_after_marker_rename:
                real_rename(source, target)
            raise OSError("injected marker interruption")
        return real_rename(source, target)

    monkeypatch.setattr(os, "rename", interrupted_marker_rename)

    with pytest.raises(OSError, match="marker interruption"):
        _execute_remote_script(REMOTE_CLASSIFY_SCRIPT, payload)
    assert marker.exists() is crash_after_marker_rename
    assert localized.is_dir()
    assert partial.is_dir()

    _execute_remote_script(REMOTE_CLASSIFY_SCRIPT, payload)

    assert not partial.exists()
    assert not localized.exists()
    assert json.loads(marker.read_text(encoding="utf-8"))["localized_pruned"] is True
    assert not (tmp_path / "clean" / "Task_sonic").exists()
    assert not rejected_root.exists()


def test_local_classification_recovers_after_prune_before_marker_update(
    tmp_path,
    monkeypatch,
):
    payload, partial, rejected_root, marker, localized = (
        _split_classification_payload(
            tmp_path,
            local_passed=("ep_local",),
        )
    )
    real_rmtree = nas_cleanup.shutil.rmtree
    fail_once = True

    def interrupted_rmtree(path, *args, **kwargs):
        nonlocal fail_once
        result = real_rmtree(path, *args, **kwargs)
        if Path(path) == localized and fail_once:
            fail_once = False
            raise OSError("injected post-prune interruption")
        return result

    monkeypatch.setattr(nas_cleanup.shutil, "rmtree", interrupted_rmtree)

    with pytest.raises(OSError, match="post-prune interruption"):
        _execute_remote_script(REMOTE_CLASSIFY_SCRIPT, payload)
    assert not partial.exists()
    assert not localized.exists()
    assert json.loads(marker.read_text(encoding="utf-8"))["localized_pruned"] is False

    _execute_remote_script(REMOTE_CLASSIFY_SCRIPT, payload)

    assert json.loads(marker.read_text(encoding="utf-8"))["localized_pruned"] is True
    assert not (tmp_path / "clean" / "Task_sonic").exists()
    assert not rejected_root.exists()


def test_remote_classification_rejects_ambiguous_source_and_target(tmp_path):
    payload, partial, rejected_root = _classification_payload(tmp_path)
    (partial / "ep_passed").mkdir()
    (partial / "ep_failed").mkdir()
    rejected_partial = rejected_root / ".Task_sonic.partial-job"
    (rejected_partial / "ep_failed").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="ambiguous|both clean and rejected"):
        _execute_remote_script(REMOTE_CLASSIFY_SCRIPT, payload)

    assert (partial / "ep_failed").is_dir()
    assert (rejected_partial / "ep_failed").is_dir()


def test_remote_classification_refuses_rejected_final_symlink_before_move(
    tmp_path,
):
    payload, partial, rejected_root = _classification_payload(tmp_path)
    (partial / "ep_passed").mkdir()
    (partial / "ep_failed").mkdir()
    protected = tmp_path / "protected"
    protected.mkdir()
    rejected_root.mkdir()
    (rejected_root / "Task_sonic").symlink_to(
        protected,
        target_is_directory=True,
    )

    with pytest.raises(RuntimeError, match="symlink|regular|task path"):
        _execute_remote_script(REMOTE_CLASSIFY_SCRIPT, payload)

    assert (partial / "ep_failed").is_dir()
    assert list(protected.iterdir()) == []


def test_remote_classification_refuses_clean_root_symlink_before_publish(
    tmp_path,
):
    protected = tmp_path / "protected"
    protected.mkdir()
    clean_root = tmp_path / "clean"
    clean_root.symlink_to(protected, target_is_directory=True)
    partial = protected / ".Task_sonic.partial-job"
    (partial / "ep_passed").mkdir(parents=True)
    payload = {
        "task": "Task",
        "passed": ["ep_passed"],
        "failed": [],
        "clean_root": str(clean_root),
        "rejected_root": str(tmp_path / "rejected"),
        "job_id": "job",
    }

    with pytest.raises(RuntimeError, match="symlink|root"):
        _execute_remote_script(REMOTE_CLASSIFY_SCRIPT, payload)

    assert partial.is_dir()
    assert not (protected / "Task_sonic").exists()


def test_rsync_checksum_is_dry_run_and_detects_local_extras(tmp_path, monkeypatch):
    worker = NasCleanupWorker.__new__(NasCleanupWorker)
    worker.config = {
        "ssh_host": "nas",
        "remote_rsync_path": "/safe/rsync",
    }
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    worker.verify_rsync_checksum(
        remote_path="/remote/.Task.partial",
        local_path=tmp_path / "local",
        log_path=tmp_path / "checksum.log",
    )

    command = observed["command"]
    assert "--checksum" in command
    assert "--dry-run" in command
    assert "--delete" in command
    assert "--itemize-changes" in command
    assert "--no-perms" in command
    assert "--omit-dir-times" in command
    assert "--partial" not in command
    assert "--append-verify" not in command
    assert not any(item.startswith("--chmod=") for item in command)
    assert observed["kwargs"]["capture_output"] is True


def test_evaluator_does_not_reuse_report_from_different_success_mode(
    tmp_path,
):
    worker = NasCleanupWorker.__new__(NasCleanupWorker)
    worker.config = {
        "evaluator_path": str(tmp_path / "evaluate.py"),
        "python_executable": "/usr/bin/python3",
        "repo_root": str(tmp_path),
        "success_mode": "any",
        "seed": 7,
        "evaluation_episode_timeout_seconds": 300.0,
    }
    worker.logger = type(
        "Logger",
        (),
        {"info": staticmethod(lambda *args, **kwargs: None)},
    )()
    task = TaskInventory(
        name="SmallTask",
        episode_hdf5=2,
        valid_episode_bytes=10,
        runs=("2026-08-13-11-00-00_SmallTask_sonic",),
        episode_dirs=2,
        missing_hdf5=0,
        batch_hdf5=0,
    )
    manifest = {"episodes": [{"episode": "ep_1"}, {"episode": "ep_2"}]}
    task_log = tmp_path / "logs"
    task_log.mkdir()
    stale_report = {
        **_evaluation_report(),
        "dataset_root": str(tmp_path),
        "success_mode": "final",
        "seed": 999,
    }
    (task_log / "evaluation.json").write_text(
        json.dumps(stale_report),
        encoding="utf-8",
    )
    fresh_report = {
        **_evaluation_report(),
        "dataset_root": str(tmp_path),
        "success_mode": "any",
        "seed": 7,
        "execution": {
            "episode_process_isolation": True,
            "episode_timeout_seconds": 300.0,
            "child_reports_dir": str(task_log / "evaluation.episodes-test"),
        },
    }
    (task_log / "evaluation.episodes-test").mkdir()
    commands = []

    def fake_run_logged(command, *, log_path, cwd=None):
        commands.append(command)
        (task_log / "evaluation.json").write_text(
            json.dumps(fresh_report),
            encoding="utf-8",
        )

    worker.run_logged_command = fake_run_logged

    report, report_path = worker.run_evaluator(
        task=task,
        local_task=tmp_path / "SmallTask_sonic",
        manifest=manifest,
        task_log=task_log,
    )

    assert commands, "stale report must be archived and evaluated again"
    assert report["success_mode"] == "any"
    assert report["seed"] == 7
    assert report_path == task_log / "evaluation.json"
    assert "--isolate-episodes" in commands[0]
    assert len(list(task_log.glob("evaluation.incomplete-*.json"))) == 1


def test_evaluated_resume_with_missing_local_task_does_not_redownload(
    tmp_path,
):
    task = TaskInventory(
        name="SmallTask",
        episode_hdf5=2,
        valid_episode_bytes=0,
        runs=("2026-08-13-11-00-00_SmallTask_sonic",),
        episode_dirs=2,
        missing_hdf5=0,
        batch_hdf5=0,
    )
    remote = validate_remote_config(_remote_config())
    staging = tmp_path / "staging"
    log_root = tmp_path / "logs"
    staging.mkdir()
    task_log = log_root / "tasks" / task.name / "worker"
    task_log.mkdir(parents=True)
    local_task = (
        staging / "pipeline" / task.name / "dataset" / f"{task.name}_sonic"
    )
    expected_partial = str(
        remote["clean_root"] / f".{task.name}_sonic.partial-job"
    )
    expected_final = str(remote["clean_root"] / f"{task.name}_sonic")
    manifest = {
        "task": task.name,
        "partial": expected_partial,
        "final": expected_final,
        "episode_count": 2,
        "total_bytes": 0,
        "episodes": [
            {"episode": "ep_1", "bytes": 0, "files": []},
            {"episode": "ep_2", "bytes": 0, "files": []},
        ],
    }
    (task_log / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    report = {
        **_evaluation_report(),
        "dataset_root": str(local_task.parent),
        "success_mode": "any",
        "seed": 0,
    }
    report_path = task_log / "evaluation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    state = {
        "tasks": {
            task.name: {
                "phase": "evaluated",
                "status": "running",
                "evaluation_report": str(report_path),
                "evaluation_summary": report["summary"],
            }
        }
    }
    worker = NasCleanupWorker.__new__(NasCleanupWorker)
    worker.args = SimpleNamespace(keep_local=True)
    child_reports = task_log / "evaluation.episodes-test"
    child_reports.mkdir()
    report["execution"] = {
        "episode_process_isolation": True,
        "episode_timeout_seconds": 300.0,
        "child_reports_dir": str(child_reports),
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    worker.config = {
        "job_id": "job",
        "success_mode": "any",
        "seed": 0,
        "evaluation_episode_timeout_seconds": 300.0,
    }
    worker.remote = remote
    worker.state = state
    worker.local_staging_root = staging
    worker.log_root = log_root
    worker.logger = type(
        "Logger",
        (),
        {
            "info": staticmethod(lambda *args, **kwargs: None),
            "warning": staticmethod(lambda *args, **kwargs: None),
        },
    )()
    updates = []
    classifications = []

    def update_task(name, **values):
        updates.append(dict(values))
        state["tasks"][name].update(values)

    def classify_remote(*, task, report, task_log):
        classifications.append(
            {
                "task": task.name,
                "report": report,
                "task_log": task_log,
            }
        )
        return {"passed": 1, "failed": 1}

    def unexpected(*args, **kwargs):
        raise AssertionError("evaluated resume must not redownload or reevaluate")

    worker.update_task = update_task
    worker.classify_remote = classify_remote
    worker.prepare_remote_task = unexpected
    worker.run_logged_command = unexpected
    worker.run_evaluator = unexpected

    worker.process_task(task)

    assert not local_task.exists()
    assert len(classifications) == 1
    assert all(update.get("phase") != "remote_partial_ready" for update in updates)
    assert all(update.get("phase") != "downloaded" for update in updates)
    assert state["tasks"][task.name]["phase"] == "complete"
    assert state["tasks"][task.name]["status"] == "complete"


@pytest.mark.parametrize(
    ("free_bytes", "required_bytes", "reserve_bytes", "expected"),
    [
        (110, 10, 100, "local"),
        (109, 10, 100, "nas"),
        (1_000, 0, 100, "local"),
        (100, 1_000, 100, "nas"),
    ],
)
def test_choose_task_placement_uses_whole_task_capacity_threshold(
    free_bytes,
    required_bytes,
    reserve_bytes,
    expected,
):
    assert (
        choose_task_placement(
            "LoadDishwasher",
            free_bytes,
            required_bytes,
            reserve_bytes,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("free_bytes", "required_bytes", "reserve_bytes"),
    [
        (-1, 1, 1),
        (1, -1, 1),
        (1, 1, -1),
        (True, 1, 1),
    ],
)
def test_choose_task_placement_rejects_invalid_byte_accounting(
    free_bytes,
    required_bytes,
    reserve_bytes,
):
    with pytest.raises(WorkflowError, match="bytes|capacity"):
        choose_task_placement(
            "LoadDishwasher",
            free_bytes,
            required_bytes,
            reserve_bytes,
        )


@pytest.mark.parametrize("persisted_placement", ["local", "nas"])
def test_placement_resume_never_recomputes_from_current_free_space(
    tmp_path,
    monkeypatch,
    persisted_placement,
):
    task = TaskInventory(
        name="Task",
        episode_hdf5=1,
        valid_episode_bytes=100,
        runs=("2026-08-13-11-00-00_Task_sonic",),
        episode_dirs=1,
        missing_hdf5=0,
        batch_hdf5=0,
    )
    task_state = {
        "phase": "placement_planned",
        "status": "running",
        "placement": persisted_placement,
        "placement_required_bytes": 40,
    }
    local_task = tmp_path / "staging" / "Task_sonic"
    local_task.mkdir(parents=True)
    local_dataset_root = tmp_path / "sonic_raw"
    manifest = {
        "episodes": [{"episode": "ep_1", "bytes": 40, "files": []}]
    }
    report = {
        "results": [{"episode": "ep_1", "status": "passed"}],
    }
    task_log = tmp_path / "logs"
    task_log.mkdir()
    worker = NasCleanupWorker.__new__(NasCleanupWorker)
    worker.config = {"local_reserve_bytes": 10_000}
    worker.local_dataset_root = local_dataset_root
    updates = []
    merges = []

    def fail_disk_usage(*args, **kwargs):
        raise AssertionError("persisted placement must not check current capacity")

    def fake_merge(source, target_root, task_name, expected):
        merges.append((source, target_root, task_name, tuple(expected)))
        return {
            "task": task_name,
            "target_task": str(Path(target_root) / f"{task_name}_sonic"),
            "expected": len(expected),
            "copied": list(expected),
            "reused": [],
        }

    def update_task(name, **values):
        updates.append(dict(values))
        task_state.update(values)

    monkeypatch.setattr(nas_cleanup.shutil, "disk_usage", fail_disk_usage)
    monkeypatch.setattr(nas_cleanup, "merge_successful_episodes", fake_merge)
    worker.update_task = update_task

    worker.place_successful_episodes(
        task=task,
        task_state=task_state,
        local_task=local_task,
        manifest=manifest,
        report=report,
        task_log=task_log,
    )

    assert task_state["placement"] == persisted_placement
    assert task_state["phase"] == "placement_complete"
    if persisted_placement == "local":
        assert merges == [
            (local_task, local_dataset_root, "Task", ("ep_1",))
        ]
        assert updates[-1]["local_copied"] == 1
    else:
        assert merges == []
        assert updates[-1]["nas_offloaded"] == 1


def test_merge_successful_episodes_combines_same_task_into_local_root(tmp_path):
    source = tmp_path / "staging" / "Task_sonic"
    target_root = tmp_path / "sonic_raw"
    existing = target_root / "Task_sonic"
    _write_raw_episode(existing, "ep_existing", payload=b"existing")
    _write_raw_episode(source, "ep_2", payload=b"second")
    _write_raw_episode(source, "ep_1", payload=b"first")
    _write_raw_episode(source, "ep_failed_not_selected", payload=b"failed")

    summary = merge_successful_episodes(
        source,
        target_root,
        "Task",
        ("ep_2", "ep_1"),
    )

    target = target_root / "Task_sonic"
    assert {item.name for item in target.iterdir()} == {
        "ep_existing",
        "ep_1",
        "ep_2",
    }
    assert (target / "ep_1" / "ep_demo.hdf5").read_bytes() == b"hdf5-first"
    assert (source / "ep_1" / "ep_demo.hdf5").is_file()
    assert not (target / "ep_failed_not_selected").exists()
    assert summary["task"] == "Task"
    assert Path(summary["target_task"]) == target
    assert summary["expected"] == 2
    assert summary["copied"] == ["ep_1", "ep_2"]
    assert summary["reused"] == []


def test_merge_with_no_passed_episodes_does_not_create_empty_local_task(
    tmp_path,
):
    source = tmp_path / "staging" / "Task_sonic"
    target_root = tmp_path / "sonic_raw"
    source.mkdir(parents=True)
    target_root.mkdir()

    summary = merge_successful_episodes(
        source,
        target_root,
        "Task",
        (),
    )

    assert not (target_root / "Task_sonic").exists()
    assert summary == {
        "task": "Task",
        "target_task": str(target_root / "Task_sonic"),
        "expected": 0,
        "copied": [],
        "reused": [],
    }


def test_merge_successful_episodes_reuses_same_name_same_content(tmp_path):
    source = tmp_path / "staging" / "Task_sonic"
    target_root = tmp_path / "sonic_raw"
    target = target_root / "Task_sonic"
    source_episode = _write_raw_episode(source, "ep_same", payload=b"identical")
    target_episode = _write_raw_episode(target, "ep_same", payload=b"identical")
    before = {
        path.name: (path.stat().st_ino, path.read_bytes())
        for path in target_episode.iterdir()
    }

    summary = merge_successful_episodes(
        source,
        target_root,
        "Task",
        ("ep_same",),
    )

    after = {
        path.name: (path.stat().st_ino, path.read_bytes())
        for path in target_episode.iterdir()
    }
    assert after == before
    assert source_episode.is_dir()
    assert summary["copied"] == []
    assert summary["reused"] == ["ep_same"]


def test_merge_recovers_complete_incoming_episode_by_checksum(tmp_path):
    source = tmp_path / "staging" / "Task_sonic"
    target_root = tmp_path / "sonic_raw"
    target_task = target_root / "Task_sonic"
    _write_raw_episode(source, "ep_1", payload=b"recoverable")
    incoming = _write_raw_episode(
        target_task,
        ".copying-ep_1",
        payload=b"recoverable",
    )

    summary = merge_successful_episodes(
        source,
        target_root,
        "Task",
        ("ep_1",),
    )

    target_episode = target_task / "ep_1"
    assert target_episode.is_dir()
    assert not incoming.exists()
    assert summary["copied"] == ["ep_1"]
    assert summary["reused"] == []
    assert summary["episode_manifests"]["ep_1"]
    assert all(
        set(record) == {"path", "size", "sha256"}
        for record in summary["episode_manifests"]["ep_1"]
    )


def test_merge_rejects_incomplete_incoming_without_mutation(tmp_path):
    source = tmp_path / "staging" / "Task_sonic"
    target_root = tmp_path / "sonic_raw"
    target_task = target_root / "Task_sonic"
    _write_raw_episode(source, "ep_1", payload=b"complete")
    incoming = target_task / ".copying-ep_1"
    incoming.mkdir(parents=True)
    (incoming / "ep_demo.hdf5").write_bytes(b"incomplete")

    with pytest.raises(WorkflowError, match="incoming.*checksum|conflict"):
        merge_successful_episodes(
            source,
            target_root,
            "Task",
            ("ep_1",),
        )

    assert incoming.is_dir()
    assert (incoming / "ep_demo.hdf5").read_bytes() == b"incomplete"
    assert not (target_task / "ep_1").exists()


def test_incoming_recovery_preflight_detects_later_conflict_before_rename(
    tmp_path,
):
    source = tmp_path / "staging" / "Task_sonic"
    target_root = tmp_path / "sonic_raw"
    target_task = target_root / "Task_sonic"
    _write_raw_episode(source, "ep_a", payload=b"recoverable")
    _write_raw_episode(source, "ep_z", payload=b"source")
    incoming = _write_raw_episode(
        target_task,
        ".copying-ep_a",
        payload=b"recoverable",
    )
    conflicting = _write_raw_episode(target_task, "ep_z", payload=b"target")
    before_conflict = {
        path.name: path.read_bytes() for path in conflicting.iterdir()
    }

    with pytest.raises(WorkflowError, match="checksum conflict"):
        merge_successful_episodes(
            source,
            target_root,
            "Task",
            ("ep_a", "ep_z"),
        )

    assert incoming.is_dir()
    assert not (target_task / "ep_a").exists()
    assert {
        path.name: path.read_bytes() for path in conflicting.iterdir()
    } == before_conflict


def test_merge_successful_episodes_conflict_fails_before_any_write(tmp_path):
    source = tmp_path / "staging" / "Task_sonic"
    target_root = tmp_path / "sonic_raw"
    target = target_root / "Task_sonic"
    _write_raw_episode(source, "ep_a_new", payload=b"new")
    _write_raw_episode(source, "ep_z_conflict", payload=b"source")
    conflicting = _write_raw_episode(
        target,
        "ep_z_conflict",
        payload=b"target",
    )
    before = {
        path.name: path.read_bytes()
        for path in conflicting.iterdir()
    }

    with pytest.raises(WorkflowError, match="conflict|different|checksum"):
        merge_successful_episodes(
            source,
            target_root,
            "Task",
            ("ep_a_new", "ep_z_conflict"),
        )

    assert not (target / "ep_a_new").exists()
    assert {
        path.name: path.read_bytes()
        for path in conflicting.iterdir()
    } == before
    assert not list(target.glob(".copying-*"))


def test_merge_successful_episodes_rejects_unsafe_local_target(tmp_path):
    source = tmp_path / "staging" / "Task_sonic"
    _write_raw_episode(source, "ep_1")
    protected = tmp_path / "protected"
    protected.mkdir()
    target_root_link = tmp_path / "sonic_raw"
    target_root_link.symlink_to(protected, target_is_directory=True)

    with pytest.raises(WorkflowError, match="symlink|target|root"):
        merge_successful_episodes(
            source,
            target_root_link,
            "Task",
            ("ep_1",),
        )

    assert list(protected.iterdir()) == []


def test_merge_successful_episodes_rejects_symlink_task_or_episode_target(
    tmp_path,
):
    source = tmp_path / "staging" / "Task_sonic"
    _write_raw_episode(source, "ep_1")
    target_root = tmp_path / "sonic_raw"
    protected = tmp_path / "protected"
    target_root.mkdir()
    protected.mkdir()
    (target_root / "Task_sonic").symlink_to(
        protected,
        target_is_directory=True,
    )

    with pytest.raises(WorkflowError, match="symlink|target"):
        merge_successful_episodes(
            source,
            target_root,
            "Task",
            ("ep_1",),
        )
    assert list(protected.iterdir()) == []

    (target_root / "Task_sonic").unlink()
    (target_root / "Task_sonic").mkdir()
    (target_root / "Task_sonic" / "ep_1").symlink_to(
        protected,
        target_is_directory=True,
    )
    with pytest.raises(WorkflowError, match="symlink|target|episode"):
        merge_successful_episodes(
            source,
            target_root,
            "Task",
            ("ep_1",),
        )
    assert list(protected.iterdir()) == []


def test_merge_successful_episodes_rejects_source_target_overlap(tmp_path):
    target_root = tmp_path / "sonic_raw"
    source = target_root / "Task_sonic"
    _write_raw_episode(source, "ep_1")

    with pytest.raises(WorkflowError, match="source|target|overlap"):
        merge_successful_episodes(
            source,
            target_root,
            "Task",
            ("ep_1",),
        )

    assert (source / "ep_1" / "ep_demo.hdf5").is_file()
    assert not list(target_root.glob(".incoming-*"))


def test_merge_successful_episodes_rejects_wrong_task_or_missing_episode(tmp_path):
    source = tmp_path / "staging" / "Other_sonic"
    target_root = tmp_path / "sonic_raw"
    _write_raw_episode(source, "ep_1")
    _write_raw_episode(source, "ep_extra")

    with pytest.raises(WorkflowError, match="task|source"):
        merge_successful_episodes(
            source,
            target_root,
            "Task",
            ("ep_1",),
        )

    correct_source = source.with_name("Task_sonic")
    source.rename(correct_source)
    with pytest.raises(WorkflowError, match="episode|expected"):
        merge_successful_episodes(
            correct_source,
            target_root,
            "Task",
            ("ep_missing",),
        )
    assert not (target_root / "Task_sonic").exists()
