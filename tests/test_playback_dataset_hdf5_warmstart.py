from types import SimpleNamespace

import mujoco
import numpy as np

from robocasa.scripts.dataset_scripts.playback_dataset_hdf5 import (
    _restore_sonic_band_state,
    _restore_sonic_integration_state,
)


CONTACT_MODEL_XML = """
<mujoco>
  <option solver="PGS" iterations="5"/>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body name="pelvis" pos="0 0 0.09">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


class SonicWholeBodyController:
    def __init__(self):
        self.band_enabled = True

    def release_band(self):
        self.band_enabled = False


class FakeSim:
    def __init__(self, model, data):
        self.model = SimpleNamespace(_model=model)
        self.data = SimpleNamespace(_data=data)
        self.forward_calls = 0

    def forward(self):
        self.forward_calls += 1
        mujoco.mj_forward(self.model._model, self.data._data)


def test_integration_restore_preserves_recorded_warmstart_after_forward():
    model = mujoco.MjModel.from_xml_string(CONTACT_MODEL_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.nefc > 0

    expected_warmstart = np.arange(model.nv, dtype=float) + 123.0
    data.qacc_warmstart[:] = expected_warmstart
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    recorded_state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, recorded_state, spec)

    data.qacc_warmstart[:] = 0.0
    sim = FakeSim(model, data)
    env = SimpleNamespace(
        sim=sim,
        robots=[SimpleNamespace(composite_controller=SonicWholeBodyController())],
    )

    _restore_sonic_integration_state(env, recorded_state[None])

    assert sim.forward_calls == 1
    np.testing.assert_array_equal(data.qacc_warmstart, expected_warmstart)


def test_band_state_follows_recorded_pelvis_external_force():
    model = mujoco.MjModel.from_xml_string(CONTACT_MODEL_XML)
    data = mujoco.MjData(model)
    controller = SonicWholeBodyController()
    env = SimpleNamespace(
        sim=FakeSim(model, data),
        robots=[
            SimpleNamespace(
                composite_controller=controller,
                robot_model=SimpleNamespace(root_body="pelvis"),
            )
        ],
    )
    pelvis_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "pelvis",
    )

    controller.band_enabled = False
    data.xfrc_applied[pelvis_body_id, 2] = 10.0
    _restore_sonic_band_state(env)
    assert controller.band_enabled is True

    data.xfrc_applied[pelvis_body_id] = 0.0
    _restore_sonic_band_state(env)
    assert controller.band_enabled is False
