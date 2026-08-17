"""Mock robot + teleop for hardware-free development, tests, and CI.

``MockRobot`` holds an internal joint state that ``command_joint_pos`` writes
to. ``MockTeleop`` synthesizes a smooth, deterministic trajectory so recorded
episodes contain real motion to inspect.
"""

from __future__ import annotations

import math

import numpy as np

from .interface import RobotInterface, TeleopAgent


class MockRobot(RobotInterface):
    def __init__(self, num_arm_joints: int = 6):
        self._n = num_arm_joints
        self._pos = np.zeros(num_arm_joints + 1, dtype=np.float64)  # last = gripper
        self._stopped = False

    def num_dofs(self) -> int:
        return self._n + 1

    def get_joint_pos(self) -> np.ndarray:
        return self._pos.copy()

    def command_joint_pos(self, pos: np.ndarray) -> None:
        if self._stopped:
            return
        pos = np.asarray(pos, dtype=np.float64).reshape(-1)
        if pos.shape[0] != self._n + 1:
            raise ValueError(f"expected {self._n + 1} dofs, got {pos.shape[0]}")
        self._pos = pos.copy()
        self._pos[-1] = float(np.clip(self._pos[-1], 0.0, 1.0))

    def get_observations(self) -> dict[str, np.ndarray]:
        return {
            "joint_pos": self._pos[: self._n].copy(),
            "gripper_pos": self._pos[self._n :].copy(),
        }

    def stop(self) -> None:
        self._stopped = True


class MockTeleop(TeleopAgent):
    """Generates a deterministic sinusoidal command and applies it to a robot."""

    def __init__(self, robot: RobotInterface, num_arm_joints: int = 6):
        self._robot = robot
        self._n = num_arm_joints
        self._t = 0

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        phase = self._t * 0.05
        arm = 0.3 * np.sin(phase + np.arange(self._n) * 0.5)
        gripper = 0.5 * (1.0 + math.sin(phase))  # sweeps [0, 1]
        cmd = np.concatenate([arm, [gripper]]).astype(np.float64)
        self._robot.command_joint_pos(cmd)
        self._t += 1
        return cmd

    def engage(self, abort=None) -> None:
        # No physical leader to sync to; the synthesized trajectory starts at rest.
        pass

    def read_inputs(self) -> tuple[list[bool], float] | None:
        # The mock controller has no physical buttons; sync/record are GUI-driven.
        return None

    def leader_raw(self) -> None:
        # No encoders to read raw angles from; the trajectory is synthesized.
        return None

    def stop(self) -> None:
        self._robot.stop()
