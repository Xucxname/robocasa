import sys

import h5py
import pytest

from robocasa.scripts.dataset_scripts import playback_dataset_hdf5
from robocasa.scripts.dataset_scripts.playback_dataset_hdf5 import (
    DEFAULT_VIDEO_CAMERA_NAMES,
    _resolve_render_image_names,
    get_playback_args,
)


def test_onscreen_playback_defaults_to_head_camera():
    assert _resolve_render_image_names(None, render=True) == [
        "robot0_head_camera"
    ]


def test_video_playback_keeps_three_camera_default():
    assert _resolve_render_image_names(None, render=False) == list(
        DEFAULT_VIDEO_CAMERA_NAMES
    )


def test_explicit_single_camera_is_preserved():
    assert _resolve_render_image_names("robot0_right_wrist_camera", render=True) == [
        "robot0_right_wrist_camera"
    ]


def test_onscreen_playback_rejects_multiple_explicit_cameras():
    with pytest.raises(ValueError, match="supports exactly one camera"):
        _resolve_render_image_names(
            ["robot0_head_camera", "robot0_left_wrist_camera"],
            render=True,
        )


def test_cli_onscreen_default_resolves_to_head_camera(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["playback_dataset_hdf5.py", "--dataset", "dummy.hdf5", "--render"],
    )

    args = get_playback_args()

    assert args.render_image_names == ["robot0_head_camera"]


def test_cli_rejects_multiple_onscreen_cameras(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "playback_dataset_hdf5.py",
            "--dataset",
            "dummy.hdf5",
            "--render",
            "--render_image_names",
            "robot0_head_camera",
            "robot0_left_wrist_camera",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        get_playback_args()

    assert "supports exactly one camera" in capsys.readouterr().err


def test_onscreen_camera_is_forwarded_to_environment(tmp_path, monkeypatch):
    dataset = tmp_path / "demo.hdf5"
    with h5py.File(dataset, "w") as hdf5_file:
        hdf5_file.create_group("data")

    env_kwargs = {}

    class FakeEnv:
        def close(self):
            pass

    def make_env(**kwargs):
        env_kwargs.update(kwargs)
        return FakeEnv()

    monkeypatch.setattr(
        playback_dataset_hdf5,
        "get_env_metadata_from_dataset",
        lambda dataset_path: {"env_name": "FakeEnv", "env_kwargs": {}},
    )
    monkeypatch.setattr(playback_dataset_hdf5.robosuite, "make", make_env)

    playback_dataset_hdf5.playback_dataset(
        dataset=str(dataset),
        use_actions=False,
        use_abs_actions=False,
        use_obs=False,
        filter_key=None,
        n=0,
        render=True,
        render_image_names=None,
        camera_height=64,
        camera_width=64,
        video_path=None,
        video_skip=1,
        extend_states=False,
        first=False,
        verbose=False,
    )

    assert env_kwargs["render_camera"] == "robot0_head_camera"
