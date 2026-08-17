"""Passive-GELLO leader driver.

A passive GELLO is a fully un-powered leader arm: every joint (arm joints + the
gripper trigger) is a magnetic encoder that *broadcasts* its angle on the CAN bus
(``0x50F`` report frames, one per device id) — there are **no motors**. 

We read the broadcast with i2rt's own ``PassiveJointEncoder``. A background thread keeps the latest
report per device; ``get_state()`` assembles the joint vector, and derives the normalized gripper and buttons.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Sequence

import numpy as np

# Encoder counts -> radians. The passive encoders report a signed 16-bit count over
# one 4096-count revolution (same constant i2rt uses in PassiveEncoderReader).
_COUNTS_PER_REV = 4096
_COUNTS_TO_RAD = 2.0 * math.pi / _COUNTS_PER_REV

# Trigger travel (rad) that maps to fully-closed, capped against the measured
# ``closed_rad``. Mirrors GRIPPER_RANGE in the fleet's own passive_gello driver.
GRIPPER_RANGE = 0.67


def _normalize_gripper(angle: float, closed_rad: float) -> float:
    """Trigger angle (rad) -> ``[0, 1]``, 1 = released/open, 0 = fully squeezed/closed.

    Normalizes by distance from zero, as the fleet's driver does, so the desk GELLO's
    negative-on-squeeze trigger and a mobile one's positive trigger map to the same curve,
    and the gripper's ``joint_signs`` entry has no effect (a signed mapping would pin one
    of them to a constant). ``closed_rad`` is the measured full-squeeze angle; only its
    magnitude is used, capped at ``GRIPPER_RANGE`` so a slightly short squeeze reads closed.
    """
    span = min(abs(closed_rad), GRIPPER_RANGE)
    if span == 0:
        return 0.0
    delta = min(abs(angle), span)
    return 1.0 - delta / span


class PassiveGelloLeader:
    """Leader backed by passive CAN joint encoders.

    Args:
        channel: CAN interface name (e.g. ``can_lead_l``).
        num_arm_joints: number of arm joints (encoder device ids ``0 .. n-1``).
        gripper_config: ``(device_id, closed_rad, open_rad)`` for the gripper trigger
            encoder. ``closed_rad`` sets the travel (see ``_normalize_gripper``); released
            is always 0, so ``open_rad`` is unused and kept only for config compatibility.
        joint_signs: per-joint direction that matches the follower's joint frame,
            applied as ``angle = sign * raw_rad``. Length ``num_arm_joints + 1``
            (arm joints then gripper). Default: ``+1`` for every joint. The home
            zero is stored in each encoder's EEPROM (``calibrate_gello.py zero``),
            not in software, so there is no offset term. The gripper entry is inert
            in ``get_state``: normalization keys off ``|angle|``.
        gello_type: ``"regular"`` (desk) or ``"mobile"``; ``side`` is ``"left"`` or
            ``"right"`` on a mobile rig. Identification only; the sign table is picked
            upstream by ``config.leader_joint_signs_for``.
        button_device: device id whose discrete inputs supply the ``[top, second]``
            buttons. Defaults to the gripper device.
        bitrate / bustype: CAN link params (mirrors i2rt's socketcan usage).
        report_freq: per-device broadcast rate (Hz) to ensure on connect. A GELLO
            left in passive mode (``report_freq=0`` in EEPROM) is silent until polled;
            we set this so it streams ``0x50F`` for passive reading. Set 0 to skip.
        ready_timeout: seconds to wait for every joint to report at least once
            before ``get_state`` will answer; raises a clear error otherwise.
    """

    def __init__(
        self,
        channel: str,
        num_arm_joints: int = 6,
        gripper_config: Sequence[float] = (6, 0.7, 0.0),
        joint_signs: Sequence[int] | None = None,
        gello_type: str = "regular",
        side: str | None = None,
        button_device: int | None = None,
        bitrate: int = 1_000_000,
        bustype: str = "socketcan",
        report_freq: int = 200,
        ready_timeout: float = 3.0,
    ) -> None:
        import can
        from i2rt.utils.encoder_manager import EncoderCanID, PassiveJointEncoder

        self.channel = channel
        self.gello_type = gello_type
        self.side = side
        self._n = num_arm_joints
        self._njoints = num_arm_joints + 1  # arm joints + gripper
        self._grip_device = int(gripper_config[0])
        self._grip_closed = float(gripper_config[1])
        self._button_device = self._grip_device if button_device is None else int(button_device)
        # Arm encoder device ids are 0 .. n-1; the gripper has its own id.
        self._arm_devices = list(range(self._n))
        self._needed = set(self._arm_devices) | {self._grip_device, self._button_device}

        self._signs = (
            np.ones(self._njoints)
            if joint_signs is None
            else np.asarray(joint_signs, dtype=float).reshape(-1)
        )
        if self._signs.shape[0] != self._njoints:
            raise ValueError(
                f"joint_signs must have length {self._njoints} "
                f"(arm joints + gripper), got {self._signs.shape[0]}"
            )

        self._report_id = int(EncoderCanID.REPORT)
        self._bus = can.interface.Bus(interface=bustype, channel=channel, bitrate=bitrate)
        self._encoder = PassiveJointEncoder(self._bus)  # sets CAN filters for 0x50E/0F/10

        # Ensure the encoders are broadcasting (a GELLO reset to passive mode is silent
        # until polled). Skip the write if it's already streaming, to avoid needless
        # EEPROM churn.
        if report_freq > 0 and not self._encoder.wait_for_report(timeout=0.3):
            self._encoder.set_report_frequency(report_freq)
            time.sleep(0.3)

        # latest raw report per device id: {device: (position_counts, velocity, inputs)}
        self._latest: dict[int, tuple[int, int, int]] = {}
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()
        self._thread = threading.Thread(
            target=self._reader_loop, name=f"passive_gello_{channel}", daemon=True
        )
        self._thread.start()
        self._wait_ready(ready_timeout)

    # --- background reader -------------------------------------------------
    def _reader_loop(self) -> None:
        while self._running.is_set():
            try:
                reports = self._encoder.wait_for_report(timeout=0.05)
            except Exception:
                # A transient bus hiccup shouldn't kill the reader; the ready/
                # freshness checks surface a persistent outage instead.
                time.sleep(0.01)
                continue
            if not reports:
                continue
            with self._lock:
                for r in reports:
                    self._latest[r.device] = (r.position, r.velocity, r.inputs)

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                seen = set(self._latest)
            if self._needed <= seen:
                return
            time.sleep(0.02)
        with self._lock:
            seen = sorted(self._latest)
        missing = sorted(self._needed - set(seen))
        self.stop()
        raise RuntimeError(
            f"passive GELLO on {self.channel!r} not reporting joints {missing} "
            f"(saw device ids {seen}). Is the leader powered and on this bus? "
            f"Expected encoders 0..{self._n - 1} (arm) + {self._grip_device} (gripper)."
        )

    # --- leader interface -------------------------------------------------
    def _joint_angle(self, device: int, joint_index: int) -> tuple[float, float]:
        pos_counts, vel_counts, _ = self._latest[device]
        raw = pos_counts * _COUNTS_TO_RAD
        angle = self._signs[joint_index] * raw
        return angle, vel_counts * _COUNTS_TO_RAD

    def get_state(self) -> tuple[np.ndarray, float, list[bool]]:
        """One snapshot -> (arm_joints[rad], gripper_norm[0..1], [top, second] buttons)."""
        with self._lock:
            missing = self._needed - set(self._latest)
            if missing:
                raise RuntimeError(
                    f"passive GELLO on {self.channel!r} stopped reporting joints "
                    f"{sorted(missing)} (leader unpowered or bus dropped?)"
                )
            arm = np.array(
                [self._joint_angle(dev, i)[0] for i, dev in enumerate(self._arm_devices)],
                dtype=np.float64,
            )
            grip_angle, _ = self._joint_angle(self._grip_device, self._n)
            inputs = self._latest[self._button_device][2]

        gripper = _normalize_gripper(grip_angle, self._grip_closed)
        buttons = [bool(inputs & 0x01), bool(inputs & 0x02)]
        return arm, gripper, buttons

    def raw_angles(self) -> tuple[np.ndarray, float, list[bool]]:
        """Uncalibrated readings for calibration: (arm_raw_rad, gripper_raw_rad,
        buttons) with NO sign applied. ``get_state`` = sign*raw."""
        with self._lock:
            missing = self._needed - set(self._latest)
            if missing:
                raise RuntimeError(
                    f"passive GELLO on {self.channel!r} not reporting joints {sorted(missing)}"
                )
            arm = np.array(
                [self._latest[dev][0] * _COUNTS_TO_RAD for dev in self._arm_devices],
                dtype=np.float64,
            )
            grip = self._latest[self._grip_device][0] * _COUNTS_TO_RAD
            inputs = self._latest[self._button_device][2]
        return arm, grip, [bool(inputs & 0x01), bool(inputs & 0x02)]

    def leader_angles(self) -> tuple[np.ndarray, np.ndarray]:
        """``(raw, cal)`` joint vectors in radians, each ``[arm_0..arm_{n-1}, gripper]``.

        ``cal`` is ``sign * raw``. Unlike ``get_state``, the gripper element is an encoder
        angle, not normalized. Reads only the report cache, so it is safe to call every
        control tick."""
        arm_raw, grip_raw, _ = self.raw_angles()
        raw = np.concatenate([arm_raw, [grip_raw]])
        return raw, raw * self._signs

    def command_arm(self, arm_joints: np.ndarray) -> None:
        """No-op: a passive GELLO has no motors, so bilateral force feedback is
        impossible. Present only to satisfy the leader interface."""
        return None

    def hardware_zero(self) -> list[int]:
        """Set the current pose as the zero in every joint encoder's EEPROM, gripper
        included (i2rt ``reset_zero_position``). Returns the device ids that were zeroed.

        Hold the leader at the pose you want mapped to the follower's home, with the
        trigger fully released: the gripper normalizes by distance from zero (see
        ``_normalize_gripper``), so zero must be the released position, or the trigger
        reads a constant offset and can stick at one end."""
        devices = [*self._arm_devices, self._grip_device]
        for dev in devices:
            self._encoder.reset_zero_position(device=dev)
            time.sleep(0.05)
        return devices

    def stop(self) -> None:
        self._running.clear()
        t = getattr(self, "_thread", None)
        if t is not None and t.is_alive():
            t.join(timeout=0.5)
        try:
            self._bus.shutdown()
        except Exception:
            pass
