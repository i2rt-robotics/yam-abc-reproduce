"""Leader joint-sign resolution: mobile GELLO per-side tables, station fallback, overrides."""

import sys
import types

import yaml

from yam_abc_reproduce.config import (
    CONTROLLER_TYPE_OPTIONS,
    ControllerConfig,
    RobotConfig,
    RobotUnitConfig,
    StationConfig,
    apply_station_form,
    build_station_config,
    controller_channel,
    gello_variant,
    is_passive_gello,
    leader_joint_signs_for,
)
from yam_abc_reproduce.robot import passive_gello
from yam_abc_reproduce.runtime import build_arm_units

# Literal copies of the fleet driver's tables (lab42
# common/hardware/src/common_hardware/robots/passive_gello.py), spelled out rather than
# imported from config so the assertions catch drift.
STATION_SIGNS = [-1, -1, 1, 1, -1, -1, 1]
MOBILE_LEFT_SIGNS = [-1, 1, 1, 1, -1, -1, -1]
MOBILE_RIGHT_SIGNS = [-1, -1, -1, -1, -1, -1, 1]


def test_mobile_gello_is_a_selectable_passive_leader_on_the_leader_buses():
    for side, channel in (("left", "can_lead_l"), ("right", "can_lead_r")):
        ctype = f"mobile_gello_{side}"
        assert ctype in CONTROLLER_TYPE_OPTIONS
        assert is_passive_gello(ctype)
        assert controller_channel(ctype) == channel


def test_mobile_gello_sides_carry_their_own_signs():
    left = ControllerConfig(type="mobile_gello_left", controls="yam_left")
    right = ControllerConfig(type="mobile_gello_right", controls="yam_right")
    # One station-wide list can't express two tables, so the variant table wins.
    assert leader_joint_signs_for(left, STATION_SIGNS) == MOBILE_LEFT_SIGNS
    assert leader_joint_signs_for(right, STATION_SIGNS) == MOBILE_RIGHT_SIGNS
    assert left.joint_signs is right.joint_signs is None
    # The sides disagree on the gripper element, which is safe only because the trigger
    # normalizes by |angle| (see test_gripper_normalization).
    assert MOBILE_LEFT_SIGNS[-1] != MOBILE_RIGHT_SIGNS[-1]


def test_gello_variant_maps_controller_types_to_type_and_side():
    assert gello_variant("mobile_gello_left") == ("mobile", "left")
    assert gello_variant("mobile_gello_right") == ("mobile", "right")
    # A desk GELLO's sides share one table, so it carries no side.
    assert gello_variant("passive_gello_left") == ("regular", None)
    assert gello_variant("passive_gello_right") == ("regular", None)


def test_desk_gello_prefers_the_station_signs_over_the_regular_table():
    desk = ControllerConfig(type="passive_gello_left", controls="yam_left")
    tuned = [1, 1, 1, 1, 1, 1, 1]
    assert leader_joint_signs_for(desk, tuned) == tuned
    # With nothing configured, fall back to the fleet's regular table, not all +1.
    assert leader_joint_signs_for(desk, None) == STATION_SIGNS


def test_controller_joint_signs_win_over_the_variant_table():
    tuned = [1, 1, 1, 1, 1, 1, -1]
    ctrl = ControllerConfig(type="mobile_gello_left", controls="yam_left", joint_signs=tuned)
    assert leader_joint_signs_for(ctrl, STATION_SIGNS) == tuned


def test_station_yaml_parses_per_controller_joint_signs(tmp_path):
    tuned = [1, -1, 1, -1, 1, -1, 1]
    path = tmp_path / "station.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "robot": {
                    "type": "yam",
                    "leader_joint_signs": STATION_SIGNS,
                    "robots": [{"type": "yam_left", "gripper": "flexible_4310"}],
                    "controllers": [
                        {
                            "type": "mobile_gello_left",
                            "controls": "yam_left",
                            "joint_signs": tuned,
                        }
                    ],
                }
            }
        )
    )
    cfg = build_station_config(path)
    (ctrl,) = cfg.robot.controllers
    assert leader_joint_signs_for(ctrl, cfg.robot.leader_joint_signs) == tuned


def test_station_form_preserves_a_controller_override_it_cannot_edit():
    tuned = [1, -1, 1, -1, 1, -1, 1]
    base = StationConfig(
        robot=RobotConfig(
            controllers=[
                ControllerConfig(type="mobile_gello_left", controls="yam_left", joint_signs=tuned)
            ]
        )
    )
    cfg = apply_station_form(
        base, {"controllers": [{"type": "mobile_gello_left", "controls": "yam_left"}]}
    )
    (ctrl,) = cfg.robot.controllers
    assert ctrl.joint_signs == tuned


def test_runtime_hands_each_leader_its_resolved_signs(monkeypatch):
    """Each leader is built with its own side's signs. Drivers are stubbed, so no CAN/i2rt."""
    built: dict[str, list[int]] = {}
    variants: dict[str, tuple[str, str | None]] = {}

    class FakeLeader:
        def __init__(self, channel, num_arm_joints, gripper_config, joint_signs, gello_type, side):
            built[channel] = joint_signs
            variants[channel] = (gello_type, side)

    stub = types.ModuleType("yam_abc_reproduce.robot.yam_adapter")
    stub.YamRobot = lambda **kw: types.SimpleNamespace(
        stop=lambda: None, gripper_limits=lambda: kw.get("gripper_limits")
    )
    stub.YamLeaderArm = lambda **kw: types.SimpleNamespace(stop=lambda: None)
    stub.YamTeleop = lambda **kw: types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "yam_abc_reproduce.robot.yam_adapter", stub)
    monkeypatch.setattr(passive_gello, "PassiveGelloLeader", FakeLeader)

    cfg = StationConfig(
        robot=RobotConfig(
            type="yam",
            leader_joint_signs=STATION_SIGNS,
            robots=[
                RobotUnitConfig("yam_left", "flexible_4310"),
                RobotUnitConfig("yam_right", "flexible_4310"),
            ],
            controllers=[
                ControllerConfig("mobile_gello_left", "yam_left"),
                ControllerConfig("mobile_gello_right", "yam_right"),
            ],
        )
    )
    assert [u.name for u in build_arm_units(cfg)] == ["left", "right"]
    assert built == {"can_lead_l": MOBILE_LEFT_SIGNS, "can_lead_r": MOBILE_RIGHT_SIGNS}
    assert variants == {"can_lead_l": ("mobile", "left"), "can_lead_r": ("mobile", "right")}
