"""Autonomy runs on the follower buses only.

A policy replaces the teleop leader, so a rollout needs one CAN bus per arm instead of
two. These cover the session states that come out of that: the followers-only go-live,
handing a live teleop session over to a policy, and the safety paths (E-STOP, the
e-stop latch) that no longer have a ControlLoop to route through.
"""

import numpy as np
import pytest

from yam_abc_reproduce.config import CameraConfig, RobotConfig, StationConfig
from yam_abc_reproduce.gui.session import CollectSession
from yam_abc_reproduce.runtime import ArmUnit
from yam_abc_reproduce.teleop.loop import ControlLoop


@pytest.fixture
def session(tmp_path):
    cfg = StationConfig(
        robot=RobotConfig(type="mock", num_arm_joints=6),
        cameras=[CameraConfig(name="top", type="mock", role="top", mode="mono")],
        control_hz=200.0,
        save_root=str(tmp_path / "episodes"),
    )
    s = CollectSession(cfg, mock=True)
    yield s
    s.reset_session()


# --- go-live ---------------------------------------------------------------


def test_deploy_go_live_builds_followers_without_a_teleop_stack(session):
    session.go_live(session.cfg, followers_only=True)

    assert session.live and session.followers_only
    assert [u.name for u in session.units] == ["left", "right"]
    assert all(u.agent is None for u in session.units)
    # No ControlLoop and no teleop recorder: engage_all() would ramp each follower to
    # wherever its leader sits, right before the policy takes over.
    assert session.loop is None and session.recorder is None
    assert session.status()["teleop_running"] is False


def test_teleop_go_live_still_builds_leaders_and_syncs(session):
    session.go_live(session.cfg)

    assert session.live and not session.followers_only
    assert all(u.agent is not None for u in session.units)
    assert session.loop is not None
    assert session.status()["teleop_running"] is True


def test_teleop_is_refused_until_the_session_is_rebuilt(session):
    session.go_live(session.cfg, followers_only=True)

    with pytest.raises(RuntimeError, match="Reset Session"):
        session.start_teleop()

    session.reset_session()
    session.start_teleop()
    assert session.live and not session.followers_only
    assert all(u.agent is not None for u in session.units)


def test_control_loop_refuses_leaderless_units():
    """The loop reaches the arms through their agents — a followers-only unit would
    make estop() raise partway through and leave the rest of the arms uncommanded."""
    from yam_abc_reproduce.robot.mock_robot import MockRobot

    units = [ArmUnit("left", MockRobot(num_arm_joints=6), None)]
    with pytest.raises(ValueError, match="followers-only"):
        ControlLoop(units, cameras=[], control_hz=0)


def test_preflight_covers_every_bus_it_is_about_to_open():
    """Both arms' buses are checked, not just robots[0]'s pair."""
    cfg = StationConfig(robot=RobotConfig())  # two yam arms + two lead arms
    assert CollectSession._needed_channels(cfg, followers_only=False) == [
        "can_left", "can_right", "can_lead_l", "can_lead_r",
    ]
    assert CollectSession._needed_channels(cfg, followers_only=True) == [
        "can_left", "can_right",
    ]


# --- safety ----------------------------------------------------------------


def test_estop_stops_every_arm_with_no_loops_left(session):
    """The hole this guards: after Stop, a followers-only session has no ControlLoop
    and no DeployLoop, but both arms are still energized holding the last pose."""
    session.go_live(session.cfg, followers_only=True)
    robots = [u.robot for u in session.units]
    assert session.loop is None and session.deploy_loop is None

    session.estop()

    assert all(r._stopped for r in robots)
    assert session.status()["estopped"] is True


def test_estop_latch_clears_when_the_session_re_arms(session):
    """Start Teleop is a documented E-STOP recovery, so the latch can't outlive it."""
    session.go_live(session.cfg, followers_only=True)
    session.estop()
    assert session.status()["estopped"] is True

    session.start_teleop(session.cfg)  # estop() cleared `live`, so this rebuilds
    assert session.status()["estopped"] is False


# --- teleop -> autonomy handover -------------------------------------------


class _FakeLeader:
    def __init__(self):
        self.stopped = False

    def get_state(self):
        return np.zeros(6), 0.0, [False, False]

    def stop(self):
        self.stopped = True


def _teleop_agent():
    from yam_abc_reproduce.robot.mock_robot import MockRobot
    from yam_abc_reproduce.robot.yam_adapter import YamTeleop

    follower = MockRobot(num_arm_joints=6)
    return YamTeleop(leader=_FakeLeader(), follower=follower), follower


def test_release_leader_closes_the_bus_and_leaves_the_follower_alone():
    agent, follower = _teleop_agent()
    leader = agent._leader

    assert agent.release_leader() is True
    assert leader.stopped
    assert not follower._stopped  # follower keeps holding torque


def test_a_released_agent_refuses_to_command_the_follower():
    """PassiveGelloLeader.stop() leaves its last sample in place, so an un-poisoned
    agent would happily snap the follower back to the handoff pose mid-rollout."""
    agent, follower = _teleop_agent()
    agent.release_leader()
    before = follower.get_joint_pos()

    for call in (lambda: agent.act({"joint_pos": np.zeros(6)}), agent.read_inputs, agent.engage):
        with pytest.raises(RuntimeError, match="released for autonomy"):
            call()
    assert np.array_equal(follower.get_joint_pos(), before)


def test_deploy_hands_a_live_teleop_session_over(session, monkeypatch):
    """Load & Run on a live teleop session drops the teleop stack and closes the
    leader buses; the followers are never torn down, so they keep holding torque."""
    session.go_live(session.cfg)  # teleop: leaders open
    followers = [u.robot for u in session.units]
    released = []
    for u in session.units:
        u.agent.release_leader = lambda name=u.name: (released.append(name), True)[1]

    _stub_policy_client(monkeypatch)
    session.start_deploy(host="127.0.0.1", port=8000, prompt="do it", record=False)
    try:
        assert released == ["left", "right"]
        assert session.followers_only and session.live
        assert session.loop is None and session.recorder is None
        assert [u.robot for u in session.units] == followers
        assert not any(f._stopped for f in followers)
    finally:
        session.stop_deploy()


def test_release_leader_reports_a_motorized_lead_arm_it_cannot_release():
    """A YamLeaderArm has no stop(): closing its i2rt chain would drop gravity comp
    and the arm would sag, so it keeps its bus — but the agent is still poisoned."""
    from yam_abc_reproduce.robot.mock_robot import MockRobot
    from yam_abc_reproduce.robot.yam_adapter import YamTeleop

    class _Motorized:
        def get_state(self):
            return np.zeros(6), 0.0, [False, False]

    agent = YamTeleop(leader=_Motorized(), follower=MockRobot(num_arm_joints=6))
    assert agent.release_leader() is False
    with pytest.raises(RuntimeError, match="released for autonomy"):
        agent.read_inputs()


# --- rollout ---------------------------------------------------------------


class _FakePolicyClient:
    """Stands in for WebsocketPolicyClient (which connects in __init__)."""

    def __init__(self, *a, **kw):
        self.metadata: dict = {}
        self.calls = 0

    def get_action(self, obs) -> np.ndarray:
        self.calls += 1
        return np.zeros(14, dtype=np.float32)

    def reset(self) -> None:
        pass


def _stub_policy_client(monkeypatch) -> None:
    monkeypatch.setattr(
        "yam_abc_reproduce.deploy.client.WebsocketPolicyClient", _FakePolicyClient
    )


def test_cold_deploy_goes_live_on_the_followers_only(session, monkeypatch):
    _stub_policy_client(monkeypatch)
    session.start_deploy(host="127.0.0.1", port=8000, prompt="do it", record=False)
    try:
        assert session.live and session.followers_only
        assert session.loop is None  # never started teleop on the way in
        assert all(u.agent is None for u in session.units)
        assert [u.name for u in session.units] == ["left", "right"]
    finally:
        session.stop_deploy()


def test_the_arms_are_homed_before_the_policy_takes_over(session, monkeypatch):
    _stub_policy_client(monkeypatch)
    home = [0.1] * 7 + [0.2] * 7
    session.go_live(session.cfg, followers_only=True)

    # Trace what each follower is commanded, in order.
    traces: dict[str, list] = {}
    for u in session.units:
        trace = traces[u.name] = []
        inner = u.robot.command_joint_pos
        u.robot.command_joint_pos = lambda pos, _t=trace, _i=inner: (
            _t.append(np.asarray(pos, dtype=float).copy()), _i(pos)
        )[1]

    session.start_deploy(
        host="127.0.0.1", port=8000, prompt="do it", record=False, home_pose=home
    )
    try:
        for name, want in (("left", home[:7]), ("right", home[7:])):
            trace = traces[name]
            reached = [i for i, c in enumerate(trace) if np.allclose(c, want, atol=1e-6)]
            assert reached, f"{name} was never commanded to the home pose"
            # ...and only then does the (all-zero) policy start commanding.
            assert np.allclose(trace[-1], 0.0, atol=1e-6)
            assert reached[0] < len(trace) - 1
    finally:
        session.stop_deploy()


def test_a_second_rollout_is_refused_while_one_is_running(session, monkeypatch):
    _stub_policy_client(monkeypatch)
    session.start_deploy(host="127.0.0.1", port=8000, prompt="do it", record=False)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            session.start_deploy(host="127.0.0.1", port=8000, prompt="do it", record=False)
    finally:
        session.stop_deploy()


def test_a_wrong_length_home_pose_is_refused_before_anything_moves(session, monkeypatch):
    _stub_policy_client(monkeypatch)
    with pytest.raises(ValueError, match=r"7 values but this station needs 14"):
        session.start_deploy(
            host="127.0.0.1", port=8000, prompt="do it", record=False, home_pose=[0.0] * 7
        )
    assert not session.live  # refused before CAN, the arm build and the gripper sweep


def test_the_arm_monitor_reads_from_the_rollout(session, monkeypatch):
    """With no ControlLoop in a deploy session the monitor would otherwise go blank."""
    _stub_policy_client(monkeypatch)
    session.start_deploy(host="127.0.0.1", port=8000, prompt="do it", record=False)
    try:
        arms = session.status()["arms"]
        assert sorted(arms) == ["left", "right"]
        row = arms["left"]
        assert len(row["follower"]) == 7 and len(row["leader_cmd"]) == 7
        assert row["leader_raw"] is None and row["leader_buttons"] is None
    finally:
        loop = session.deploy_loop
        session.stop_deploy()

    # Same keys as the teleop snapshot, so app.js renderArms needs no branch.
    from yam_abc_reproduce.robot.mock_robot import MockRobot, MockTeleop

    robot = MockRobot(num_arm_joints=6)
    teleop = ControlLoop(
        [ArmUnit("left", robot, MockTeleop(robot, num_arm_joints=6))], cameras=[], control_hz=0
    )
    assert set(row) == set(teleop.joint_snapshot()["left"])
    # Nothing is commanded after Stop, so the action row must not look live.
    assert loop.joint_snapshot()["left"]["leader_cmd"] is None


def test_deploy_home_pose_round_trips_from_the_station_yaml(tmp_path):
    import yaml

    from yam_abc_reproduce.config import apply_station_form, build_station_config

    pose = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0] * 2
    path = tmp_path / "station.yaml"
    path.write_text(yaml.safe_dump({"robot": {"type": "yam"}, "deploy_home_pose": pose}))

    cfg = build_station_config(path)
    assert cfg.deploy_home_pose == pose
    # The Station rail doesn't expose it, so editing the rail must not drop it.
    assert apply_station_form(cfg, {"task_name": "pp"}).deploy_home_pose == pose

