"""Per-device CAN channel selection: explicit override, type fallback, conflicts.

Each robot/controller opens its own ``can.interface.Bus``, so two devices sharing a bus
are refused up front rather than left to fight over it at runtime. No CAN or i2rt is
touched here.
"""

import sys
import types

import pytest
import yaml

from yam_abc_reproduce.config import (
    ControllerConfig,
    RobotConfig,
    RobotUnitConfig,
    StationConfig,
    apply_station_form,
    build_station_config,
    controller_channel_for,
    robot_channel_for,
)
from yam_abc_reproduce.runtime import build_arm_units


def test_channel_falls_back_to_the_type_default_when_unset():
    assert robot_channel_for(RobotUnitConfig("yam_right")) == "can_right"
    assert controller_channel_for(ControllerConfig("mobile_gello_right")) == "can_lead_r"


def test_explicit_channel_wins_over_the_type_default():
    assert robot_channel_for(RobotUnitConfig("yam_left", channel="can3")) == "can3"
    assert controller_channel_for(ControllerConfig("passive_gello_left", channel="can7")) == "can7"


def test_blank_channel_is_treated_as_unset():
    """The rail sends "" for an untouched field, which means derive from type."""
    assert robot_channel_for(RobotUnitConfig("yam_left", channel="  ")) == "can_left"


def test_station_yaml_parses_per_device_channels(tmp_path):
    path = tmp_path / "station.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "robot": {
                    "type": "yam",
                    "robots": [{"type": "yam_left", "gripper": "flexible_4310", "channel": "can5"}],
                    "controllers": [
                        {"type": "passive_gello_left", "controls": "yam_left", "channel": "can6"}
                    ],
                }
            }
        )
    )
    cfg = build_station_config(path)
    (rb,) = cfg.robot.robots
    (ct,) = cfg.robot.controllers
    assert robot_channel_for(rb) == "can5"
    assert controller_channel_for(ct) == "can6"


def test_station_yaml_rejects_two_devices_on_one_bus(tmp_path):
    path = tmp_path / "station.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "robot": {
                    "type": "yam",
                    "robots": [
                        {"type": "yam_left", "channel": "can0"},
                        {"type": "yam_right", "channel": "can0"},
                    ],
                }
            }
        )
    )
    with pytest.raises(ValueError, match="more than one device"):
        build_station_config(path)


def test_a_default_collision_is_caught_too():
    """An explicit channel colliding with another device's derived default also fails."""
    with pytest.raises(ValueError, match="more than one device"):
        apply_station_form(
            StationConfig(),
            {
                "robots": [{"type": "yam_left"}, {"type": "yam_right", "channel": "can_left"}],
                "controllers": [],
            },
        )


def test_rail_round_trips_a_channel_and_a_cleared_channel():
    base = StationConfig(
        robot=RobotConfig(
            robots=[RobotUnitConfig("yam_left", "flexible_4310", channel="can9")],
            controllers=[ControllerConfig("passive_gello_left", "yam_left", channel="can8")],
        )
    )
    cfg = apply_station_form(
        base,
        {
            "robots": [{"type": "yam_left", "gripper": "flexible_4310", "channel": "can4"}],
            # Clearing the field returns this leader to its type-derived bus.
            "controllers": [{"type": "passive_gello_left", "controls": "yam_left", "channel": ""}],
        },
    )
    (rb,) = cfg.robot.robots
    (ct,) = cfg.robot.controllers
    assert rb.channel == "can4"
    assert ct.channel is None
    assert controller_channel_for(ct) == "can_lead_l"


def _stub_adapter(monkeypatch) -> dict[str, str]:
    """Stub out the i2rt boundary and return the dict recording which bus each
    device driver was opened on."""
    opened: dict[str, str] = {}

    def fake_robot(**kw):
        opened["follower"] = kw["channel"]
        return types.SimpleNamespace(stop=lambda: None, gripper_limits=lambda: None)

    def fake_leader(**kw):
        opened["leader"] = kw["channel"]
        return types.SimpleNamespace(stop=lambda: None)

    stub = types.ModuleType("yam_abc_reproduce.robot.yam_adapter")
    stub.YamRobot = fake_robot
    stub.YamLeaderArm = fake_leader
    stub.YamTeleop = lambda **kw: types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "yam_abc_reproduce.robot.yam_adapter", stub)
    return opened


def _one_arm_station() -> StationConfig:
    return StationConfig(
        robot=RobotConfig(
            type="yam",
            robots=[RobotUnitConfig("yam_left", "flexible_4310", channel="can2")],
            controllers=[ControllerConfig("yam_lead_left", "yam_left", channel="can3")],
        )
    )


def test_runtime_opens_the_configured_channels(monkeypatch):
    """The follower and leader drivers get the resolved bus, not the type default."""
    opened = _stub_adapter(monkeypatch)
    assert [u.name for u in build_arm_units(_one_arm_station())] == ["left"]
    assert opened == {"follower": "can2", "leader": "can3"}


def test_followers_only_opens_no_leader_bus(monkeypatch):
    """Autonomy commands the followers, so a followers-only build opens one bus per
    arm and leaves the leaders (and their buses) alone."""
    opened = _stub_adapter(monkeypatch)
    (unit,) = build_arm_units(_one_arm_station(), followers_only=True)
    assert opened == {"follower": "can2"}  # no "leader" key: never constructed
    assert unit.agent is None


def test_followers_only_needs_no_controller_at_all(monkeypatch):
    """A station with the leaders unplugged (and unconfigured) can still deploy."""
    opened = _stub_adapter(monkeypatch)
    cfg = StationConfig(
        robot=RobotConfig(type="yam", robots=[RobotUnitConfig("yam_right")], controllers=[])
    )
    (unit,) = build_arm_units(cfg, followers_only=True)
    assert opened == {"follower": "can_right"}
    assert unit.agent is None


def test_followers_only_on_the_mock_path():
    cfg = StationConfig(robot=RobotConfig(type="mock", num_arm_joints=6))
    units = build_arm_units(cfg, mock=True, followers_only=True)
    assert [u.name for u in units] == ["left", "right"]
    assert [u.agent for u in units] == [None, None]
