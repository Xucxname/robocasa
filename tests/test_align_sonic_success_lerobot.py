import json
import os
from pathlib import Path

import pytest

from robocasa.scripts.align_sonic_success_lerobot import (
    AlignmentError,
    align_by_completion_time,
    build_selection_plan,
    cleaner_python_path,
    discover_lerobot_episodes,
    discover_raw_episodes,
    load_replay_statuses,
)


def _write_raw_episode(
    raw_root: Path,
    *,
    task: str,
    episode: str,
    end_timestamp: str,
    with_hdf5: bool,
) -> Path:
    task_dir = raw_root / f"2026-08-19-12-00-00_{task}_sonic"
    episode_dir = task_dir / "episodes" / episode
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / f"state_{end_timestamp}.npz").write_bytes(b"state")
    if with_hdf5:
        (episode_dir / "ep_demo.hdf5").write_bytes(b"hdf5")
    return episode_dir


def _write_lerobot_dataset(
    root: Path,
    *,
    task: str,
    episode_mtimes: list[float],
    discarded: tuple[int, ...] = (),
) -> Path:
    dataset = root / f"robocasa_{task}_g1_3cam"
    (dataset / "meta").mkdir(parents=True)
    info = {
        "total_episodes": len(episode_mtimes),
        "chunks_size": 1000,
        "data_path": (
            "data/chunk-{episode_chunk:03d}/"
            "episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "discarded_episode_indices": list(discarded),
        "features": {
            "action.motion_token": {"dtype": "float64", "shape": [64]},
            "observation.images.ego_view": {"dtype": "video"},
        },
    }
    (dataset / "meta" / "info.json").write_text(json.dumps(info))
    with (dataset / "meta" / "episodes.jsonl").open("w") as file:
        for index in range(len(episode_mtimes)):
            file.write(
                json.dumps({"episode_index": index, "length": 2, "tasks": []})
                + "\n"
            )
    for index, mtime in enumerate(episode_mtimes):
        parquet = (
            dataset
            / "data"
            / "chunk-000"
            / f"episode_{index:06d}.parquet"
        )
        video = (
            dataset
            / "videos"
            / "chunk-000"
            / "observation.images.ego_view"
            / f"episode_{index:06d}.mp4"
        )
        parquet.parent.mkdir(parents=True, exist_ok=True)
        video.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_bytes(b"parquet")
        video.write_bytes(b"video")
        os.utime(parquet, (mtime, mtime))
    return dataset


def _write_replay_report(
    path: Path,
    *,
    task: str,
    statuses: dict[str, str],
    completed_at: str = "2026-09-01T12:00:00+08:00",
) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "success_mode": "any",
                "completed_at": completed_at,
                "results": [
                    {"task": task, "episode": episode, "status": status}
                    for episode, status in statuses.items()
                ],
            }
        )
    )
    return path


def test_latest_replay_result_supersedes_older_status_and_keeps_provenance(
    tmp_path,
):
    task = "TaskReplay"
    older = _write_replay_report(
        tmp_path / "older.json",
        task=task,
        statuses={"ep_1": "error"},
        completed_at="2026-09-01T11:00:00+08:00",
    )
    newer = _write_replay_report(
        tmp_path / "newer.json",
        task=task,
        statuses={"ep_1": "passed"},
        completed_at="2026-09-01T13:00:00+08:00",
    )

    statuses, provenance = load_replay_statuses([newer, older], task)

    assert statuses == {"ep_1": "passed"}
    assert [item["path"] for item in provenance] == [
        str(newer.resolve()),
        str(older.resolve()),
    ]
    assert all(item["sha256"] for item in provenance)
    assert [item["completed_at"] for item in provenance] == [
        "2026-09-01T13:00:00+08:00",
        "2026-09-01T11:00:00+08:00",
    ]


def test_replay_result_completion_time_precedence(tmp_path):
    task = "TaskReplay"
    report = tmp_path / "precedence.json"
    report.write_text(
        json.dumps(
            {
                "completed_at": "2026-09-01T15:00:00+08:00",
                "results": [
                    {
                        "task": task,
                        "episode": "ep_1",
                        "status": "failed",
                        "completed_at": "2026-09-01T14:00:00+08:00",
                        "isolation": {
                            "completed_at": "2026-09-01T11:00:00+08:00"
                        },
                    },
                    {
                        "task": task,
                        "episode": "ep_1",
                        "status": "passed",
                        "completed_at": "2026-09-01T12:00:00+08:00",
                    },
                ],
            }
        )
    )

    statuses, _provenance = load_replay_statuses([report], task)

    assert statuses == {"ep_1": "passed"}


def test_conflicting_statuses_only_abort_at_same_latest_time(tmp_path):
    task = "TaskReplay"
    first = _write_replay_report(
        tmp_path / "first.json",
        task=task,
        statuses={"ep_1": "failed"},
        completed_at="2026-09-01T13:00:00+08:00",
    )
    second = _write_replay_report(
        tmp_path / "second.json",
        task=task,
        statuses={"ep_1": "passed"},
        completed_at="2026-09-01T13:00:00+08:00",
    )

    with pytest.raises(AlignmentError, match="conflicting latest replay statuses"):
        load_replay_statuses([first, second], task)


def test_plan_keeps_only_uniquely_matched_replay_passed_episode(tmp_path):
    task = "TaskA"
    raw_root = tmp_path / "sonic_raw"
    _write_raw_episode(
        raw_root,
        task=task,
        episode="ep_1",
        end_timestamp="100_0",
        with_hdf5=False,
    )
    _write_raw_episode(
        raw_root,
        task=task,
        episode="ep_2",
        end_timestamp="200_0",
        with_hdf5=True,
    )
    _write_raw_episode(
        raw_root,
        task=task,
        episode="ep_3",
        end_timestamp="300_0",
        with_hdf5=True,
    )
    dataset = _write_lerobot_dataset(
        tmp_path, task=task, episode_mtimes=[103.0, 204.0, 305.0, 500.0]
    )
    report = _write_replay_report(
        tmp_path / "report.json",
        task=task,
        statuses={"ep_2": "passed", "ep_3": "failed"},
    )

    raw = discover_raw_episodes(raw_root, task)
    _info, lerobot = discover_lerobot_episodes(dataset)
    statuses, provenance = load_replay_statuses([report], task)
    matches, unmatched = align_by_completion_time(
        raw,
        lerobot,
        max_lag_seconds=15.0,
        ambiguity_margin_seconds=1.0,
    )
    plan = build_selection_plan(
        task=task,
        dataset=dataset,
        raw_episodes=raw,
        lerobot_episodes=lerobot,
        matches=matches,
        unmatched_lerobot=unmatched,
        replay_statuses=statuses,
        replay_provenance=provenance,
        max_lag_seconds=15.0,
        ambiguity_margin_seconds=1.0,
    )

    assert plan["kept_source_episode_indices"] == [1]
    assert plan["excluded_source_episode_indices"] == [0, 2, 3]
    assert plan["extra_discard_episode_indices"] == [0, 2, 3]
    assert plan["unmatched_lerobot_episode_indices"] == [3]


def test_existing_discard_is_not_reintroduced(tmp_path):
    task = "TaskB"
    raw_root = tmp_path / "sonic_raw"
    _write_raw_episode(
        raw_root,
        task=task,
        episode="ep_1",
        end_timestamp="100_0",
        with_hdf5=True,
    )
    _write_raw_episode(
        raw_root,
        task=task,
        episode="ep_2",
        end_timestamp="200_0",
        with_hdf5=True,
    )
    dataset = _write_lerobot_dataset(
        tmp_path, task=task, episode_mtimes=[102.0, 202.0], discarded=(0,)
    )
    report = _write_replay_report(
        tmp_path / "report.json",
        task=task,
        statuses={"ep_1": "passed", "ep_2": "passed"},
    )
    raw = discover_raw_episodes(raw_root, task)
    _info, lerobot = discover_lerobot_episodes(dataset)
    statuses, provenance = load_replay_statuses([report], task)
    matches, unmatched = align_by_completion_time(
        raw,
        lerobot,
        max_lag_seconds=15.0,
        ambiguity_margin_seconds=1.0,
    )

    plan = build_selection_plan(
        task=task,
        dataset=dataset,
        raw_episodes=raw,
        lerobot_episodes=lerobot,
        matches=matches,
        unmatched_lerobot=unmatched,
        replay_statuses=statuses,
        replay_provenance=provenance,
        max_lag_seconds=15.0,
        ambiguity_margin_seconds=1.0,
    )

    assert plan["kept_source_episode_indices"] == [1]
    assert plan["successful_but_source_discarded"] == [0]
    assert plan["extra_discard_episode_indices"] == []


def test_missing_replay_result_fails_closed(tmp_path):
    task = "TaskC"
    raw_root = tmp_path / "sonic_raw"
    _write_raw_episode(
        raw_root,
        task=task,
        episode="ep_1",
        end_timestamp="100_0",
        with_hdf5=True,
    )
    dataset = _write_lerobot_dataset(tmp_path, task=task, episode_mtimes=[102.0])
    raw = discover_raw_episodes(raw_root, task)
    _info, lerobot = discover_lerobot_episodes(dataset)
    matches, unmatched = align_by_completion_time(
        raw,
        lerobot,
        max_lag_seconds=15.0,
        ambiguity_margin_seconds=1.0,
    )

    with pytest.raises(AlignmentError, match="missing replay results"):
        build_selection_plan(
            task=task,
            dataset=dataset,
            raw_episodes=raw,
            lerobot_episodes=lerobot,
            matches=matches,
            unmatched_lerobot=unmatched,
            replay_statuses={},
            replay_provenance=[],
            max_lag_seconds=15.0,
            ambiguity_margin_seconds=1.0,
        )


def test_ambiguous_timestamp_match_is_rejected(tmp_path):
    task = "TaskD"
    raw_root = tmp_path / "sonic_raw"
    _write_raw_episode(
        raw_root,
        task=task,
        episode="ep_1",
        end_timestamp="100_0",
        with_hdf5=True,
    )
    _write_raw_episode(
        raw_root,
        task=task,
        episode="ep_2",
        end_timestamp="100_5",
        with_hdf5=True,
    )
    dataset = _write_lerobot_dataset(tmp_path, task=task, episode_mtimes=[102.0])
    raw = discover_raw_episodes(raw_root, task)
    _info, lerobot = discover_lerobot_episodes(dataset)

    with pytest.raises(AlignmentError, match="ambiguous timestamp match"):
        align_by_completion_time(
            raw,
            lerobot,
            max_lag_seconds=15.0,
            ambiguity_margin_seconds=1.0,
        )


def test_cleaner_python_path_preserves_virtualenv_symlink(tmp_path):
    base_python = tmp_path / "base-python"
    base_python.write_bytes(b"python")
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)

    selected = cleaner_python_path(venv_python)

    assert selected == venv_python.absolute()
    assert selected != venv_python.resolve()
