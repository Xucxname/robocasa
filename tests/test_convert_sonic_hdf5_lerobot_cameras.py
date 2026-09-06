import json
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from robocasa.scripts.dataset_scripts.convert_sonic_hdf5_lerobot import (
    THREE_CAMERA_NAMES,
    _inject_virtual_wrist_cameras,
    _prepare_model_xml,
    _write_camera_metadata,
)
from robocasa.utils.camera_utils import get_robot_cam_configs


WRIST_CAMERA_NAMES = (
    "robot0_left_wrist_camera",
    "robot0_right_wrist_camera",
)


def _model_xml(*, include_left_camera=False):
    left_camera = (
        '<camera name="robot0_left_wrist_camera" pos="1 2 3" quat="1 0 0 0"/>'
        if include_left_camera
        else ""
    )
    return f"""
    <mujoco>
      <worldbody>
        <body name="robot0_torso_link">
          <camera name="robot0_head_camera"/>
        </body>
        <body name="robot0_left_wrist_yaw_link">{left_camera}</body>
        <body name="robot0_right_wrist_yaw_link"/>
      </worldbody>
    </mujoco>
    """


def _vector_string(values):
    return " ".join(str(value) for value in values)


def test_injected_wrist_cameras_match_canonical_sonic_g1_config():
    root = ET.fromstring(_inject_virtual_wrist_cameras(_model_xml()))
    canonical = get_robot_cam_configs("SonicG1")

    for camera_name in WRIST_CAMERA_NAMES:
        config = canonical[camera_name]
        body = root.find(f".//body[@name='{config['parent_body']}']")
        assert body is not None
        camera = body.find(f"camera[@name='{camera_name}']")
        assert camera is not None
        assert camera.attrib["mode"] == "fixed"
        assert camera.attrib["pos"] == _vector_string(config["pos"])
        assert camera.attrib["quat"] == _vector_string(config["quat"])
        for key, value in config["camera_attribs"].items():
            assert camera.attrib[key] == str(value)
        assert "euler" not in camera.attrib
        assert "fovy" not in camera.attrib


def test_injection_preserves_an_existing_camera():
    root = ET.fromstring(
        _inject_virtual_wrist_cameras(_model_xml(include_left_camera=True))
    )
    left = root.find(".//camera[@name='robot0_left_wrist_camera']")
    right = root.find(".//camera[@name='robot0_right_wrist_camera']")

    assert left is not None
    assert left.attrib["pos"] == "1 2 3"
    assert left.attrib["quat"] == "1 0 0 0"
    assert right is not None


def test_prepare_model_rejects_missing_wrist_cameras_when_injection_disabled():
    args = SimpleNamespace(
        camera_names=list(THREE_CAMERA_NAMES),
        inject_virtual_wrist_cameras=False,
    )

    with pytest.raises(ValueError, match="robot0_left_wrist_camera"):
        _prepare_model_xml(_model_xml(), args)


def test_camera_metadata_records_canonical_injected_extrinsics(tmp_path):
    args = SimpleNamespace(
        camera_names=list(THREE_CAMERA_NAMES),
        image_keys=["ego_view", "left_wrist", "right_wrist"],
        inject_virtual_wrist_cameras=True,
    )
    (tmp_path / "meta").mkdir()

    _write_camera_metadata(tmp_path, args)

    metadata = json.loads((tmp_path / "meta" / "cameras.json").read_text())
    canonical = get_robot_cam_configs("SonicG1")
    for image_key, camera_name in zip(args.image_keys[1:], WRIST_CAMERA_NAMES):
        entry = metadata[image_key]
        config = canonical[camera_name]
        assert entry["mujoco_body"] == config["parent_body"]
        assert entry["pos"] == _vector_string(config["pos"])
        assert entry["quat"] == _vector_string(config["quat"])
        assert "euler" not in entry
        assert "fovy" not in entry
