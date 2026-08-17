"""The ONLY module that imports i2rt. Everything else uses the protocols in
``interface.py``.

Boundary conventions (YAM-ABC-Reproduce side):
  * joint vector = ``[arm_0 ... arm_{n-1}, gripper]``
  * arm joints are radians (pass-through to/from i2rt)
  * gripper is normalized ``[0, 1]`` (0 = closed, 1 = open).

NOTE The follower gripper unit (normalized vs raw) depends on i2rt's gripper driver and should be verified on the
bench. The defaults here are placeholders.
"""

from __future__ import annotations

import time
from threading import Event

import numpy as np

from .interface import RobotInterface, TeleopAgent


def _normalize(raw: float, lo: float, hi: float) -> float:
    if hi == lo:
        return float(np.clip(raw, 0.0, 1.0))
    return float(np.clip((raw - lo) / (hi - lo), 0.0, 1.0))


def _denormalize(norm: float, lo: float, hi: float) -> float:
    return float(lo + np.clip(norm, 0.0, 1.0) * (hi - lo))


def _build_yam(
    channel: str,
    arm_type: str,
    gripper_type: str,
    ee_mass: float | None,
    gripper_limits: list[float] | None = None,
):
    """Construct an i2rt YAM robot, converting our string config to i2rt enums.

    ``gripper_limits`` becomes i2rt's ``gripper_limits_override``, which pins the gripper's
    ``[closed, open]`` motor range and sets ``gripper_needs_cal = False``. Left None, a
    gripper declaring ``needs_calibration`` re-measures the range on every construction.
    See ``RobotUnitConfig.gripper_limits``.
    """
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import ArmType, GripperType

    return get_yam_robot(
        channel=channel,
        arm_type=ArmType.from_string_name(arm_type),
        gripper_type=GripperType.from_string_name(gripper_type),
        ee_mass=ee_mass,
        gripper_limits_override=(
            None if gripper_limits is None else np.asarray(gripper_limits, dtype=float)
        ),
    )


class YamRobot(RobotInterface):
    """Follower YAM arm wrapped from i2rt's ``get_yam_robot``."""

    def __init__(
        self,
        channel: str,
        gripper_type: str = "linear_4310",
        num_arm_joints: int = 6,
        arm_type: str = "yam",
        ee_mass: float | None = None,
        gripper_limits: list[float] | None = None,
        gripper_raw_open: float = 1.0,
        gripper_raw_closed: float = 0.0,
    ):
        self._robot = _build_yam(channel, arm_type, gripper_type, ee_mass, gripper_limits)
        self._n = num_arm_joints
        self._g_open = gripper_raw_open
        self._g_closed = gripper_raw_closed

    def num_dofs(self) -> int:
        return self._n + 1

    def gripper_limits(self) -> list[float] | None:
        """The ``[closed, open]`` gripper motor range i2rt is using: the pinned config
        value, or what its auto-calibration measured this run. Our normalized ``[0, 1]``
        gripper spans exactly this range, so a collapsed one means the jaws barely move."""
        limits = self._robot.get_robot_info().get("gripper_limits")
        return None if limits is None else [float(x) for x in limits]

    def get_joint_pos(self) -> np.ndarray:
        q = np.asarray(self._robot.get_joint_pos(), dtype=np.float64).reshape(-1)
        arm = q[: self._n]
        grip = _normalize(q[self._n], self._g_closed, self._g_open)
        return np.concatenate([arm, [grip]])

    def command_joint_pos(self, pos: np.ndarray) -> None:
        pos = np.asarray(pos, dtype=np.float64).reshape(-1)
        arm = pos[: self._n]
        grip_raw = _denormalize(pos[self._n], self._g_closed, self._g_open)
        self._robot.command_joint_pos(np.concatenate([arm, [grip_raw]]))

    def get_observations(self) -> dict[str, np.ndarray]:
        full = self.get_joint_pos()
        return {"joint_pos": full[: self._n].copy(), "gripper_pos": full[self._n :].copy()}

    def stop(self) -> None:
        # Hold current pose (cease following). Used by E-STOP: the arm must stay
        # put (not go limp) so anything held isn't dropped.
        try:
            self._robot.command_joint_pos(self._robot.get_joint_pos())
        except Exception:
            pass

    def relax(self) -> None:
        # Zero torque: the arm goes limp (kp/kd/commands -> 0). Reversible -- a
        # later command_joint_pos + rebuilt kp/kd re-energizes it. Used when
        # tearing the session down so the arm is safe to handle / re-home.
        try:
            self._robot.zero_torque_mode()
        except Exception:
            pass

    def power_off(self) -> dict:
        """Send i2rt's motor-off frame to every follower joint and gripper.

        This deliberately differs from :meth:`stop`: ``stop`` holds the last
        pose, whereas this method disables each motor after the control thread
        has stopped.  The caller must therefore ensure the arm is supported.
        """
        chain = self._robot.motor_chain
        motor_interface = chain.motor_interface
        disabled, failed = [], {}
        for spec in chain.motor_list:
            motor_id = int(spec[0])
            try:
                motor_interface.motor_off(motor_id)
                disabled.append(motor_id)
            except Exception as exc:  # noqa: BLE001
                failed[str(motor_id)] = str(exc)
        # Stop i2rt's background command thread only after every motor-off frame
        # has been attempted, so no later position command can re-enable a joint.
        try:
            chain.close()
        except Exception as exc:  # noqa: BLE001
            failed["transport"] = str(exc)
        return {"disabled": disabled, "failed": failed}


class YamLeaderArm:
    """Passive YAM lead arm read through i2rt.

    Backdrivability: with ``bilateral_kp == 0`` the PD gains are zeroed so the arm
    floats on i2rt's gravity compensation (exactly as i2rt's bimanual_lead_follower
    leader does). With ``bilateral_kp > 0`` the leader is given proportional gains
    and can be commanded toward the follower pose for force feedback.
    """

    def __init__(
        self,
        channel: str,
        num_arm_joints: int = 6,
        arm_type: str = "yam",
        gripper_type: str = "yam_teaching_handle",
        ee_mass: float | None = None,
        bilateral_kp: float = 0.0,
    ):
        self._robot = _build_yam(channel, arm_type, gripper_type, ee_mass)
        self._n = num_arm_joints
        # Remember the arm's native kp so bilateral scaling is relative to it.
        self._native_kp = np.asarray(getattr(self._robot, "_kp", np.zeros(self._n)), dtype=float)
        kp = self._native_kp * bilateral_kp if bilateral_kp > 0 else np.zeros(self._n)
        self._robot.update_kp_kd(kp=kp, kd=np.zeros(self._n))

    def get_state(self) -> tuple[np.ndarray, float, list[bool]]:
        """One same-bus read -> (arm_joints, gripper_norm, [top, second] buttons).

        Mirrors i2rt minimum_gello's ``YAMLeaderRobot.get_info``: the teaching-handle
        trigger is ``1 - position`` and ``io_inputs`` are the two button bits."""
        arm = np.asarray(self._robot.get_joint_pos(), dtype=np.float64).reshape(-1)[: self._n]
        enc = self._robot.motor_chain.get_same_bus_device_states()[0]
        gripper = float(np.clip(1.0 - enc.position, 0.0, 1.0))
        # Teaching-handle button polarity varies between units (some idle high,
        # some idle low). Learn the idle level from the first read (assumes no
        # button held during startup) and report "pressed" as deviation from it.
        raw = [bool(b) for b in enc.io_inputs]
        if not hasattr(self, "_btn_idle") or self._btn_idle is None or len(self._btn_idle) != len(raw):
            self._btn_idle = raw
        buttons = [r != i for r, i in zip(raw, self._btn_idle)]
        return arm, gripper, buttons

    def command_arm(self, arm_joints: np.ndarray) -> None:
        """Command the leader arm joints (bilateral force feedback only)."""
        self._robot.command_joint_pos(np.asarray(arm_joints, dtype=np.float64).reshape(-1))


class YamTeleop(TeleopAgent):
    """Identity map the lead arm onto the follower and apply it each tick.

    Leader and follower are the same YAM geometry, so the mapping is 1:1.
    """

    def __init__(self, leader: YamLeaderArm, follower: YamRobot, bilateral_kp: float = 0.0):
        self._leader = leader
        self._follower = follower
        self._bilateral_kp = bilateral_kp
        self._n = follower.num_dofs() - 1
        # Leader state cached by read_inputs() so act() reads the bus only once/tick.
        self._cached: tuple[np.ndarray, float] | None = None

    def _read_leader(self) -> tuple[np.ndarray, float, list[bool]]:
        arm, gripper, buttons = self._require_leader().get_state()
        self._cached = (arm, gripper)
        return arm, gripper, buttons

    def _require_leader(self):
        if self._leader is None:
            raise RuntimeError(
                "this arm's leader was released for autonomy — Reset Session and "
                "Start Teleop to rebuild it"
            )
        return self._leader

    def _target(self) -> np.ndarray:
        if self._cached is None:
            arm, gripper, _ = self._read_leader()
        else:
            arm, gripper = self._cached
        return np.concatenate([arm, [float(np.clip(gripper, 0.0, 1.0))]])

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        cmd = self._target()  # identity: follower target == leader pose
        self._follower.command_joint_pos(cmd)
        if self._bilateral_kp > 0:
            # Push the leader toward the follower's current arm pose (haptics).
            self._leader.command_arm(np.asarray(obs["joint_pos"], dtype=np.float64).reshape(-1))
        self._cached = None
        return cmd

    def read_inputs(self) -> tuple[list[bool], float] | None:
        _, gripper, buttons = self._read_leader()
        return buttons, gripper

    def leader_raw(self) -> tuple[np.ndarray, np.ndarray] | None:
        """``(raw, cal)`` leader joint angles in radians for the live readout, or None for
        a leader without a raw readout (``YamLeaderArm`` reports through i2rt, already
        calibrated)."""
        angles = getattr(self._leader, "leader_angles", None)
        return angles() if callable(angles) else None

    def engage(self, abort: Event | None = None, steps: int = 33, dt: float = 0.03) -> None:
        """Ease the follower from its current pose to the leader's pose before live
        tracking, so it never snaps. Blocks ~steps*dt s (~1 s). Aborts early when
        ``abort`` (the loop's E-STOP event) is set, so a stop interrupts the ramp
        instead of it commanding the follower all the way to the leader."""
        arm, gripper, _ = self._require_leader().get_state()
        target = np.concatenate([arm, [float(np.clip(gripper, 0.0, 1.0))]])
        start = np.asarray(self._follower.get_joint_pos(), dtype=np.float64).reshape(-1)
        for i in range(1, steps + 1):
            if abort is not None and abort.is_set():
                return
            alpha = i / steps
            self._follower.command_joint_pos(start * (1.0 - alpha) + target * alpha)
            time.sleep(dt)

    def release_leader(self) -> bool:
        """Drop the leader's CAN bus while the follower stays energized, so an
        autonomous rollout runs on the follower buses only.

        Returns whether the bus was actually closed: a motorized ``YamLeaderArm`` has
        no ``stop()`` -- closing its i2rt chain would drop gravity compensation and the
        arm would sag -- so it keeps its bus. A passive GELLO has no torque to lose.

        Either way the agent is poisoned, because ``PassiveGelloLeader.stop()`` leaves
        its last sample in place: without this, ``get_state()`` would keep returning a
        frozen pose and a later ``act()`` would snap the follower to wherever the leader
        was at handoff.
        """
        leader, self._leader, self._cached = self._leader, None, None
        leader_stop = getattr(leader, "stop", None)
        if not callable(leader_stop):
            return False
        leader_stop()
        return True

    def stop(self) -> None:
        self._follower.stop()
        # Release the leader too (e.g. a passive GELLO's CAN reader thread/bus).
        leader_stop = getattr(self._leader, "stop", None)
        if callable(leader_stop):
            try:
                leader_stop()
            except Exception:
                pass
