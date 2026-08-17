"""Factories that wire a StationConfig into robot, teleop, and cameras."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .camera.interface import CameraDriver, CameraMode
from .camera.mock_camera import MockCamera
from .config import (
    RobotUnitConfig,
    StationConfig,
    controller_channel_for,
    gello_variant,
    is_passive_gello,
    leader_joint_signs_for,
    robot_channel_for,
)
from .robot.interface import RobotInterface, TeleopAgent
from .robot.mock_robot import MockRobot, MockTeleop


@dataclass
class ArmUnit:
    """One driven teleop pair: a follower robot + its teleop agent (the agent
    holds the leader). ``name`` is the schema prefix used when recording.

    ``agent`` is None on a followers-only build (see ``build_arm_units``): autonomy
    drives the follower from a policy, so no leader is opened. ``DeployLoop`` uses
    only ``robot``/``name``; ``ControlLoop`` requires the agent."""

    name: str
    robot: RobotInterface
    agent: TeleopAgent | None


def _arm_name(robot_type: str) -> str:
    """Schema prefix from a robot type: yam_left -> left, yam_right -> right."""
    return robot_type.replace("yam_lead_", "").replace("yam_", "") or "left"


# An unpinned gripper's travel is re-measured on every build by stalling the motor (see
# ``RobotUnitConfig`` in config.py). The linear/flexible units have a ~6.57 rad motor
# stroke, so a span this far below it means the sweep stalled early.
_MIN_PLAUSIBLE_GRIPPER_SPAN = 1.0


def _report_gripper_travel(name: str, robot: RobotUnitConfig, limits: list[float] | None) -> None:
    """Log the ``[closed, open]`` range a follower's gripper ended up with, and warn if it
    collapsed: the normalized ``[0, 1]`` gripper maps onto exactly this span, so a collapse
    disables the trigger with nothing else logged."""
    source = "pinned from config" if robot.gripper_limits else "auto-detected this run"
    logging.info("%s follower gripper (%s): limits=%s (%s)", name, robot.gripper, limits, source)
    if limits is None or len(limits) != 2:
        return
    span = abs(limits[1] - limits[0])
    if span < _MIN_PLAUSIBLE_GRIPPER_SPAN:
        logging.warning(
            "%s follower gripper travel is only %.3f rad (%s) — the trigger will barely "
            "move the jaws. Clear the jaws and rebuild to re-measure, or pin a known-good "
            "`gripper_limits` for this arm; see docs/hardware.md.",
            name, span, source,
        )


def build_arm_units(
    cfg: StationConfig, mock: bool = False, followers_only: bool = False
) -> list[ArmUnit]:
    """Build one ArmUnit per configured robot (+ the controller bound to it).

    Multi-arm: every ``cfg.robot.robots`` entry becomes a driven pair. Each
    device's type maps to its CAN bus; leader->follower is an identity map (both are YAM arms).

    ``followers_only`` skips the leaders entirely (``agent`` is None), so an
    autonomous rollout opens only the follower buses -- one per arm instead of two.
    """
    n = cfg.robot.num_arm_joints
    robots = cfg.robot.robots or [RobotUnitConfig()]

    if mock or cfg.robot.type == "mock":
        units = []
        for r in robots:
            robot = MockRobot(num_arm_joints=n)
            agent = None if followers_only else MockTeleop(robot, num_arm_joints=n)
            units.append(ArmUnit(_arm_name(r.type), robot, agent))
        return units

    # Hardware path: the single i2rt boundary.
    from .robot.yam_adapter import YamLeaderArm, YamRobot, YamTeleop

    def _build_leader(controller):
        """Passive GELLO (desk or mobile) -> motorless encoder arm (PassiveGelloLeader);
        yam_lead_* -> motorized YAM arm read through i2rt (YamLeaderArm)."""
        channel = controller_channel_for(controller)
        if is_passive_gello(controller.type):
            from .robot.passive_gello import PassiveGelloLeader

            grip = cfg.robot.leader_gripper
            gello_type, side = gello_variant(controller.type)
            # Home zero lives in each encoder's EEPROM (calibrate_gello.py zero)
            return PassiveGelloLeader(
                channel=channel,
                num_arm_joints=n,
                gripper_config=tuple(grip) if grip else (n, 0.7, 0.0),
                joint_signs=leader_joint_signs_for(controller, cfg.robot.leader_joint_signs),
                gello_type=gello_type,
                side=side,
            )
        return YamLeaderArm(
            channel=channel,
            num_arm_joints=n,
            arm_type=cfg.robot.arm_type,
            gripper_type=cfg.robot.leader_gripper_type,
            ee_mass=cfg.robot.ee_mass,
            bilateral_kp=cfg.robot.bilateral_kp,
        )

    # Build each pair, tearing down anything already opened if a later build fails
    units: list[ArmUnit] = []
    opened: list = []  # every device with a stop(), in open order
    for r in robots:
        controller = None if followers_only else cfg.robot.controller_for(r)
        try:
            if r.gripper_limits is None:
                # i2rt re-measures an unpinned gripper by driving the jaws into both
                # stops. Say so first: on a cold Load & Run this sweep is the arm's
                # first motion, and _report_gripper_travel only logs once it's done.
                logging.info(
                    "%s follower gripper (%s): measuring travel — the jaws will move to "
                    "both stops. Pin `gripper_limits` in the station config to skip this.",
                    _arm_name(r.type), r.gripper,
                )
            follower = YamRobot(
                channel=robot_channel_for(r),
                gripper_type=r.gripper,
                num_arm_joints=n,
                arm_type=cfg.robot.arm_type,
                ee_mass=cfg.robot.ee_mass,
                gripper_limits=r.gripper_limits,
            )
            opened.append(follower)
            _report_gripper_travel(_arm_name(r.type), r, follower.gripper_limits())
            leader = None if controller is None else _build_leader(controller)
            if leader is not None:
                opened.append(leader)
        except Exception as exc:
            for d in reversed(opened):
                try:
                    d.stop()
                except Exception:
                    pass
            devices = f"follower={r.type}->{robot_channel_for(r)}"
            if controller is not None:
                devices += f", leader={controller.type}->{controller_channel_for(controller)}"
            raise RuntimeError(
                f"failed to build arm {_arm_name(r.type)!r} ({devices}): {exc}"
            ) from exc
        agent = None
        if leader is not None:
            # Passive GELLO has no motors: force bilateral off regardless of config.
            kp = 0.0 if is_passive_gello(controller.type) else cfg.robot.bilateral_kp
            agent = YamTeleop(leader=leader, follower=follower, bilateral_kp=kp)
        units.append(ArmUnit(_arm_name(r.type), follower, agent))
    return units


def build_cameras_from_config(cfg: StationConfig, mock: bool = False) -> list[CameraDriver]:
    if mock or cfg.robot.type == "mock":
        cams = cfg.cameras or [
            None
        ]  # default to a single mock top camera if none configured
        out: list[CameraDriver] = []
        for i, c in enumerate(cams):
            if c is None:
                out.append(MockCamera(name="top", role="top"))
            else:
                mode = CameraMode(c.mode)
                out.append(
                    MockCamera(name=c.name, role=c.role, mode=mode, width=c.width, height=c.height)
                )
        return out

    from .camera.registry import build_cameras

    return build_cameras(cfg.cameras)
