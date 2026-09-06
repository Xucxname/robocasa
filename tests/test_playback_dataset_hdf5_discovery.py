from robocasa.scripts.dataset_scripts.playback_dataset_hdf5 import (
    _discover_playback_datasets,
)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return str(path)


def test_discovers_reorganized_task_episode_datasets_in_stable_order(tmp_path):
    task_dir = tmp_path / "LoadDishwasher_sonic"
    ep_2 = _touch(task_dir / "ep_2" / "ep_demo.hdf5")
    ep_1 = _touch(task_dir / "ep_1" / "ep_demo.hdf5")

    assert _discover_playback_datasets(str(task_dir)) == [ep_1, ep_2]


def test_legacy_batch_aggregate_remains_supported(tmp_path):
    batch_dir = tmp_path / "2026-08-13_LoadDishwasher_sonic"
    aggregate = _touch(batch_dir / "demo.hdf5")

    assert _discover_playback_datasets(str(batch_dir)) == [aggregate]


def test_episode_datasets_take_precedence_over_aggregate_in_same_tree(tmp_path):
    batch_dir = tmp_path / "2026-08-13_LoadDishwasher_sonic"
    _touch(batch_dir / "demo.hdf5")
    episode = _touch(batch_dir / "episodes" / "ep_1" / "ep_demo.hdf5")

    assert _discover_playback_datasets(str(batch_dir)) == [episode]


def test_episode_precedence_applies_across_requested_tree(tmp_path):
    _touch(tmp_path / "legacy_batch" / "demo.hdf5")
    episode = _touch(
        tmp_path / "LoadDishwasher_sonic" / "ep_1" / "ep_demo.hdf5"
    )

    assert _discover_playback_datasets(str(tmp_path)) == [episode]


def test_completed_episode_video_is_skipped_without_falling_back_to_aggregate(
    tmp_path,
):
    task_dir = tmp_path / "LoadDishwasher_sonic"
    _touch(task_dir / "demo.hdf5")
    episode = task_dir / "ep_1" / "ep_demo.hdf5"
    _touch(episode)
    _touch(episode.with_suffix(".mp4"))

    assert _discover_playback_datasets(str(task_dir)) == []


def test_explicit_dataset_file_is_returned_unchanged(tmp_path):
    dataset = tmp_path / "demo.hdf5"
    _touch(dataset)
    _touch(dataset.with_suffix(".mp4"))

    assert _discover_playback_datasets(str(dataset)) == [str(dataset)]
