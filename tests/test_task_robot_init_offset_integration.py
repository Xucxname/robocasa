import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

import robocasa  # noqa: F401  Registers RoboCasa environments with robosuite.
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.scripts.collect_sonic_g1_demos import SONIC_CFG

from robocasa.utils import env_utils as EnvUtils


def _make_env(task_name):
    return robosuite.make(
        task_name,
        robots=["SonicG1"],
        controller_configs=load_composite_controller_config(
            controller=SONIC_CFG,
            robot="SonicG1",
        ),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        layout_ids=1,
        style_ids=1,
        control_freq=200,
        initialization_noise=None,
        seed=7,
    )


def _xml_pelvis_position(env):
    root = ET.fromstring(env.model.get_xml())
    pelvis = next(
        body for body in root.iter("body") if body.get("name") == "robot0_pelvis"
    )
    return np.fromstring(pelvis.attrib["pos"], sep=" ")


@pytest.mark.parametrize(
    ("task_name", "expected_offset"),
    [
        ("SlideDishwasherRack", (-0.50, 0.0)),
        ("PickPlaceDrawerToCounter", (0.0, -0.15)),
    ],
)
def test_tasks_apply_unified_offsets_before_the_sonic_root_is_written(
    task_name, expected_offset
):
    env = None
    try:
        env = _make_env(task_name)
        env.reset()
        assert env.sim.model._model.opt.solver == int(mujoco.mjtSolver.mjSOL_NEWTON)
        unshifted_anchor, base_ori = EnvUtils.compute_robot_base_placement_pose(
            env,
            ref_fixture=env.get_fixture(env.init_robot_base_ref),
        )
        expected_anchor = EnvUtils.apply_robot_init_offset(
            unshifted_anchor,
            base_ori,
            expected_offset,
        )

        np.testing.assert_allclose(
            env.init_robot_base_pos_anchor,
            expected_anchor,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            _xml_pelvis_position(env),
            env.init_robot_base_pos_anchor,
            atol=1e-7,
        )

        env.step(np.zeros(env.action_spec[0].shape))
        assert np.isfinite(env.sim.data.qpos).all()
        assert np.isfinite(env.sim.data.qvel).all()
        assert np.isfinite(env.sim.data.ctrl).all()
    finally:
        if env is not None:
            env.close()


def test_episode_metadata_restores_the_task_offset_after_the_registry_changes(
    monkeypatch,
):
    source = restored = None
    try:
        source = _make_env("PickPlaceDrawerToCounter")
        source.reset()
        metadata = source.get_ep_meta()
        monkeypatch.setitem(
            EnvUtils._TASK_ROBOT_INIT_OFFSETS,
            "PickPlaceDrawerToCounter",
            (0.0, 0.0),
        )
        assert metadata["robot_init_offset"] == [0.0, -0.15]

        restored = _make_env("PickPlaceDrawerToCounter")
        restored.set_ep_meta(metadata)
        restored.reset()

        np.testing.assert_allclose(
            restored.init_robot_base_pos_anchor,
            source.init_robot_base_pos_anchor,
            atol=1e-7,
        )
    finally:
        if source is not None:
            source.close()
        if restored is not None:
            restored.close()
