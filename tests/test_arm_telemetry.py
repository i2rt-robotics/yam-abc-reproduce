"""Per-arm live telemetry: ControlLoop.joint_snapshot feeds the GUI's arm panel.

The global ``buttons``/``trigger`` merge every controller (any leader's top button toggles
sync), so they can't say which leader moved. The snapshot keeps each leader's own reading,
which the panel renders.
"""

import numpy as np

from yam_abc_reproduce.runtime import ArmUnit
from yam_abc_reproduce.teleop.loop import ControlLoop


class _FakeRobot:
    """Follower whose observation is a constant, distinct per arm."""

    def __init__(self, fill: float):
        self.fill = fill

    def get_observations(self):
        return {"joint_pos": np.full(6, self.fill), "gripper_pos": np.array([self.fill])}

    def command_joint_pos(self, pos):
        pass

    def stop(self):
        pass


class _FakeAgent:
    """A controller whose inputs and leader angles the test drives.

    ``leader_raw`` returns ``(raw, raw * signs)`` like the real driver's
    ``leader_angles()``; ``signs=None`` models a leader with no raw readout.
    """

    def __init__(self, action_fill: float, raw_fill: float | None = None, signs=None):
        self.inputs = ([False, False], 0.0)
        self._action_fill = action_fill
        self._raw = None if raw_fill is None else np.full(7, raw_fill)
        self._signs = None if signs is None else np.asarray(signs, dtype=float)

    def act(self, obs):
        return np.full(7, self._action_fill)

    def engage(self, abort=None):
        pass

    def read_inputs(self):
        return self.inputs

    def leader_raw(self):
        if self._raw is None or self._signs is None:
            return None
        return self._raw, self._raw * self._signs

    def stop(self):
        pass


LEFT_SIGNS = [-1, 1, 1, 1, -1, -1, -1]
RIGHT_SIGNS = [-1, -1, -1, -1, -1, -1, 1]


def _two_arm_loop():
    left = _FakeAgent(0.1, raw_fill=0.5, signs=LEFT_SIGNS)
    right = _FakeAgent(0.2, raw_fill=0.25, signs=RIGHT_SIGNS)
    loop = ControlLoop(
        [
            ArmUnit("left", _FakeRobot(1.0), left),
            ArmUnit("right", _FakeRobot(2.0), right),
        ],
        cameras=[],
        control_hz=0,
    )
    return loop, left, right


def test_each_leader_keeps_its_own_buttons_and_trigger():
    loop, left, right = _two_arm_loop()
    left.inputs = ([True, False], 0.25)
    right.inputs = ([False, True], 0.75)
    loop._step()

    snap = loop.joint_snapshot()
    assert snap["left"]["leader_buttons"] == [True, False]
    assert snap["right"]["leader_buttons"] == [False, True]
    assert snap["left"]["leader_grip"] == 0.25
    assert snap["right"]["leader_grip"] == 0.75
    # The merged pair still ORs both leaders, and drives sync/record.
    assert loop.buttons == [True, True]


def test_follower_vector_is_per_arm_joints_plus_gripper():
    loop, _, _ = _two_arm_loop()
    loop._step()

    snap = loop.joint_snapshot()
    assert snap["left"]["follower"] == [1.0] * 7
    assert snap["right"]["follower"] == [2.0] * 7


def test_each_leader_reports_its_own_raw_and_sign_applied_angles():
    """The readout that makes a stuck trigger visible: per arm, raw angles off the bus and
    the same values through that side's sign table."""
    loop, _, _ = _two_arm_loop()
    loop._step()

    snap = loop.joint_snapshot()
    assert snap["left"]["leader_raw"] == [0.5] * 7
    assert snap["left"]["leader_cal"] == [0.5 * s for s in LEFT_SIGNS]
    assert snap["right"]["leader_raw"] == [0.25] * 7
    assert snap["right"]["leader_cal"] == [0.25 * s for s in RIGHT_SIGNS]


def test_a_leader_without_a_raw_readout_reports_none():
    """A motorized ``YamLeaderArm`` reads through i2rt already calibrated, so it has no raw
    encoder vector to show."""
    agent = _FakeAgent(0.1)  # no raw_fill/signs
    loop = ControlLoop([ArmUnit("left", _FakeRobot(1.0), agent)], cameras=[], control_hz=0)
    loop._step()

    snap = loop.joint_snapshot()
    assert snap["left"]["leader_raw"] is None
    assert snap["left"]["leader_cal"] is None
    # The follower half of the card still populates.
    assert snap["left"]["follower"] == [1.0] * 7


def test_action_is_reported_only_while_commanding():
    loop, _, _ = _two_arm_loop()

    loop._step()  # sync off by default: nothing is sent
    assert loop.joint_snapshot()["left"]["leader_cmd"] is None

    loop.sync_enabled = True
    loop._step()
    snap = loop.joint_snapshot()
    assert snap["left"]["leader_cmd"] == [0.1] * 7
    assert snap["right"]["leader_cmd"] == [0.2] * 7

    # Turning sync back off must clear it, or the panel keeps showing a stale command.
    loop.sync_enabled = False
    loop._step()
    assert loop.joint_snapshot()["left"]["leader_cmd"] is None
