"""Teaching-handle button control: rising-edge detection + sync gating.

Covers the continuous-loop model where the top button toggles sync and the second
button toggles recording, driven off the controller's ``read_inputs()``.
"""

import numpy as np

from yam_abc_reproduce.teleop.loop import ControlLoop


class _FakeRobot:
    def get_observations(self):
        return {"joint_pos": np.zeros(6), "gripper_pos": np.zeros(1)}

    def command_joint_pos(self, pos):
        pass

    def stop(self):
        pass


class _FakeAgent:
    """A controller whose buttons/trigger we drive from the test."""

    def __init__(self):
        self.inputs = ([False, False], 0.0)
        self.acts = 0

    def act(self, obs):
        self.acts += 1
        return np.zeros(7)

    def engage(self):
        pass

    def read_inputs(self):
        return self.inputs

    def stop(self):
        pass


def _loop_with_agent():
    from yam_abc_reproduce.runtime import ArmUnit

    agent = _FakeAgent()
    loop = ControlLoop([ArmUnit("left", _FakeRobot(), agent)], cameras=[], control_hz=0)
    return loop, agent


def test_rising_edge_fires_once_per_press():
    loop, agent = _loop_with_agent()
    sync, rec = [], []
    loop.on_sync_button = lambda: sync.append(1)
    loop.on_record_button = lambda: rec.append(1)

    # Top button: press (held 2 ticks) -> release -> press again = 2 rising edges.
    for buttons in [[False, False], [True, False], [True, False], [False, False], [True, False]]:
        agent.inputs = (buttons, 0.0)
        loop._step()
    assert len(sync) == 2, "held button must not retrigger; only rising edges fire"

    # Second button: a single press = 1 edge.
    for buttons in [[False, False], [False, True], [False, True], [False, False]]:
        agent.inputs = (buttons, 0.0)
        loop._step()
    assert len(rec) == 1


def test_sync_gate_blocks_commanding():
    loop, agent = _loop_with_agent()
    loop._step()  # sync disabled by default
    assert agent.acts == 0, "follower must not be commanded while sync is disabled"
    loop.sync_enabled = True
    loop._step()
    assert agent.acts == 1, "follower is commanded once sync is enabled"


def test_trigger_and_buttons_tracked_for_status():
    loop, agent = _loop_with_agent()
    agent.inputs = ([True, False], 0.42)
    loop._step()
    assert loop.buttons == [True, False]
    assert abs(loop.trigger - 0.42) < 1e-9
