"""Per-arm follower gripper limits: config parsing, rail preservation, runtime wiring.

Unpinned, i2rt re-measures a flexible_4310's [closed, open] range per arm on every build
by stalling the motor, and the span can come out collapsed. Pinning it in the station
config skips that. No CAN or i2rt is touched here.
"""

import logging
import sys
import types

import yaml

from yam_abc_reproduce.config import (
    ControllerConfig,
    RobotConfig,
    RobotUnitConfig,
    StationConfig,
    apply_station_form,
    build_station_config,
)
from yam_abc_reproduce.runtime import _report_gripper_travel, build_arm_units

LEFT_LIMITS = [0.0, 6.4]
RIGHT_LIMITS = [0.1, 6.2]


def test_unpinned_by_default_so_i2rt_still_auto_calibrates():
    assert RobotUnitConfig("yam_left", "flexible_4310").gripper_limits is None


def test_station_yaml_parses_per_robot_gripper_limits(tmp_path):
    path = tmp_path / "station.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "robot": {
                    "type": "yam",
                    "robots": [
                        {"type": "yam_left", "gripper": "flexible_4310", "gripper_limits": LEFT_LIMITS},
                        {"type": "yam_right", "gripper": "flexible_4310"},
                    ],
                }
            }
        )
    )
    left, right = build_station_config(path).robot.robots
    assert left.gripper_limits == LEFT_LIMITS
    assert right.gripper_limits is None


def test_station_form_preserves_a_pin_it_cannot_edit():
    base = StationConfig(
        robot=RobotConfig(robots=[RobotUnitConfig("yam_left", "flexible_4310", LEFT_LIMITS)])
    )
    cfg = apply_station_form(base, {"robots": [{"type": "yam_left", "gripper": "flexible_4310"}]})
    (rb,) = cfg.robot.robots
    assert rb.gripper_limits == LEFT_LIMITS


def test_station_form_drops_the_pin_when_the_gripper_type_changes():
    """A range measured on one gripper maps the trigger onto the wrong travel on another."""
    base = StationConfig(
        robot=RobotConfig(robots=[RobotUnitConfig("yam_left", "flexible_4310", LEFT_LIMITS)])
    )
    cfg = apply_station_form(base, {"robots": [{"type": "yam_left", "gripper": "linear_4310"}]})
    (rb,) = cfg.robot.robots
    assert rb.gripper_limits is None


def test_a_collapsed_travel_range_warns(caplog):
    """A near-zero span silently maps the whole normalized trigger onto one motor position."""
    robot = RobotUnitConfig("yam_left", "flexible_4310")
    with caplog.at_level(logging.WARNING):
        _report_gripper_travel("left", robot, [0.0, 0.08])
    assert "barely" in caplog.text
    assert "0.080 rad" in caplog.text


def test_a_healthy_travel_range_does_not_warn(caplog):
    robot = RobotUnitConfig("yam_left", "flexible_4310")
    with caplog.at_level(logging.WARNING):
        _report_gripper_travel("left", robot, [0.0, 6.5])
    assert caplog.text == ""


def test_an_absent_range_does_not_warn(caplog):
    """A gripper type with no limits (no_gripper) has nothing to check."""
    with caplog.at_level(logging.WARNING):
        _report_gripper_travel("left", RobotUnitConfig("yam_left"), None)
    assert caplog.text == ""


def test_runtime_hands_each_follower_its_own_limits(monkeypatch):
    """Each follower is built with its own arm's range; an unpinned arm still gets None
    (auto-calibrate)."""
    built: dict[str, list[float] | None] = {}

    def fake_robot(**kw):
        built[kw["channel"]] = kw["gripper_limits"]
        return types.SimpleNamespace(stop=lambda: None, gripper_limits=lambda: kw["gripper_limits"])

    stub = types.ModuleType("yam_abc_reproduce.robot.yam_adapter")
    stub.YamRobot = fake_robot
    stub.YamLeaderArm = lambda **kw: types.SimpleNamespace(stop=lambda: None)
    stub.YamTeleop = lambda **kw: types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "yam_abc_reproduce.robot.yam_adapter", stub)

    cfg = StationConfig(
        robot=RobotConfig(
            type="yam",
            robots=[
                RobotUnitConfig("yam_left", "flexible_4310", LEFT_LIMITS),
                RobotUnitConfig("yam_right", "flexible_4310"),
            ],
            controllers=[
                ControllerConfig("yam_lead_left", "yam_left"),
                ControllerConfig("yam_lead_right", "yam_right"),
            ],
        )
    )
    assert [u.name for u in build_arm_units(cfg)] == ["left", "right"]
    assert built == {"can_left": LEFT_LIMITS, "can_right": None}
