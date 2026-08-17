"""YAM-ABC-Reproduce-owned robot protocols.

Convention: a joint vector is ``[arm_joint_0 ... arm_joint_{n-1}, gripper]``
where the trailing gripper element is normalized to ``[0, 1]`` (0 = closed,
1 = open).
"""

from __future__ import annotations

from threading import Event
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RobotInterface(Protocol):
    def num_dofs(self) -> int:
        """Total DOFs including the gripper (arm joints + 1)."""
        ...

    def get_joint_pos(self) -> np.ndarray:
        """Full joint vector incl. normalized gripper as the last element."""
        ...

    def command_joint_pos(self, pos: np.ndarray) -> None:
        """Command a full joint vector (last element = normalized gripper)."""
        ...

    def get_observations(self) -> dict[str, np.ndarray]:
        """Structured observation, e.g. {"joint_pos": (n,), "gripper_pos": (1,)}."""
        ...

    def relax(self) -> None:
        """Release the arm to zero torque (limp). Default: same as stop()."""
        self.stop()

    def stop(self) -> None:
        """Hard stop: cease commanding and release/hold safely. Idempotent."""
        ...


@runtime_checkable
class TeleopAgent(Protocol):
    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Produce the follower command for this tick and apply it.

        Returns the commanded full joint vector (for recording as the action).
        """
        ...

    def engage(self, abort: Event | None = None) -> None:
        """Synchronously ease the follower to the leader's current pose before
        live tracking begins, so it never snaps. Called once on Start Teleop,
        before the control loop starts. If ``abort`` is given, poll it each step
        and return early when set (E-STOP), so the ramp can't keep commanding after
        a stop. No-op for agents that need no ramp.
        """
        ...

    def read_inputs(self) -> tuple[list[bool], float] | None:
        """Latest ``(button_states, trigger)`` from the controller, read without
        commanding the follower. ``button_states`` is ``[top, second]``; ``trigger``
        is the normalized gripper trigger in ``[0, 1]``. Returns ``None`` for
        controllers with no physical buttons."""
        ...

    def stop(self) -> None:
        ...
