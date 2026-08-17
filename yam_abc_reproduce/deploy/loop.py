"""DeployLoop: the fixed-rate read -> infer -> command cycle for autonomy.

Structural twin of ``yam_abc_reproduce.teleop.loop.ControlLoop``, but the per-tick
command comes from a policy server instead of a teleop leader. It reuses the
same camera-worker plumbing and the same ``RobotInterface``/``ArmUnit`` the
teleop path uses, so hardware wiring and camera setup are identical between
data collection and deployment.

Safety: on start it ramps every follower from its current pose to the policy's
first predicted action over a short window, so the arm never snaps. ``estop()``
stops commanding and hard-stops every arm immediately.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np

from ..camera.interface import CameraDriver, CameraFrame
from ..camera.worker import CameraWorker
from ..runtime import ArmUnit
from . import contract
from .client import PolicyClient


class DeployLoop:
    def __init__(
        self,
        units: list[ArmUnit],
        cameras: list[CameraDriver],
        client: PolicyClient,
        prompt: str,
        control_hz: float = 30.0,
        max_joint_speed: float = 1.5,
        on_frame: Callable[[str, CameraFrame], None] | None = None,
    ) -> None:
        self.units = units
        self.cameras = cameras  # drivers: name->role + image_keys (mirrors ControlLoop)
        self.client = client
        self.prompt = prompt
        # Share already-running workers (GUI) or wrap raw drivers (CLI), exactly
        # like ControlLoop, so a camera is never opened twice.
        self._own_cameras = not (cameras and all(isinstance(c, CameraWorker) for c in cameras))
        self._workers = [
            c if isinstance(c, CameraWorker) else CameraWorker(c, on_frame=on_frame)
            for c in cameras
        ]
        self._arm_dims = [u.robot.num_dofs() for u in units]
        self.dt = 1.0 / control_hz if control_hz > 0 else 0.0
        # Safety clamp, specified as a max joint SPEED (rad/s) so the limit stays
        # the same if control_hz changes; applied per step as speed * dt.
        # <=0 disables. Gripper dims are not clamped.
        self.max_joint_speed = float(max_joint_speed or 0.0)
        self.max_joint_step = self.max_joint_speed * self.dt
        self._last_clamp_warn = 0.0
        self._running = False
        self._estopped = False
        self._thread: threading.Thread | None = None
        self.actual_hz = 0.0
        self._last_action: dict[str, np.ndarray] = {}
        self._last_obs: dict[str, dict] = {}  # cached per-arm reads, for joint_snapshot()
        self._recorder = None  # optional EpisodeRecorder to log the rollout
        self.last_error: str | None = None  # surfaces a rollout-thread crash to the GUI

    def attach_recorder(self, recorder) -> None:
        """Log each rollout step (policy action + follower obs + frames) to an
        EpisodeRecorder, exactly like ControlLoop does for teleop."""
        self._recorder = recorder

    @property
    def estopped(self) -> bool:
        return self._estopped

    @property
    def running(self) -> bool:
        """True while the threaded rollout is active (for the GUI status feed)."""
        return self._running

    def set_prompt(self, prompt: str) -> None:
        """Swap the instruction mid-run (e.g. from the GUI); re-queries next tick."""
        self.prompt = prompt
        self.client.reset()

    # --- camera lifecycle (only when we own the cameras) -------------------
    def _start_cameras(self) -> None:
        if self._own_cameras:
            for w in self._workers:
                w.start()

    def _stop_cameras(self) -> None:
        if self._own_cameras:
            for w in self._workers:
                w.stop()

    # --- one iteration -----------------------------------------------------
    def _read(self) -> tuple[dict[str, CameraFrame], list[dict]]:
        frames: dict[str, CameraFrame] = {}
        for w in self._workers:
            fr = w.read()
            if fr is not None:
                frames[w.name] = fr
        per_arm_obs = [u.robot.get_observations() for u in self.units]
        # Cached here rather than in _step() so the GUI's arm monitor stays live during
        # move_to_home() and the ramp too, not just once the rollout is running.
        self._last_obs = {u.name: per_arm_obs[i] for i, u in enumerate(self.units)}
        return frames, per_arm_obs

    def _command(self, cmds: list[np.ndarray]) -> None:
        for u, cmd in zip(self.units, cmds):
            u.robot.command_joint_pos(cmd)
            self._last_action[u.name] = cmd

    def _clamp_cmds(self, cmds: list[np.ndarray], per_arm_obs: list[dict]) -> list[np.ndarray]:
        """Clip each arm-joint command to +-max_joint_step from the current joint pose
        (truncate, not refuse) and warn at most once/second. Gripper dim left untouched."""
        step = self.max_joint_step
        out, hit = [], False
        for cmd, o in zip(cmds, per_arm_obs):
            cmd = np.asarray(cmd, dtype=np.float32).copy()
            jp = np.asarray(o["joint_pos"], dtype=np.float32).reshape(-1)
            n = min(jp.shape[0], cmd.shape[0])  # arm joints; gripper (if any) stays as-is
            clipped = np.clip(cmd[:n], jp - step, jp + step)
            if not np.allclose(clipped, cmd[:n], atol=1e-6):
                hit = True
            cmd[:n] = clipped
            out.append(cmd)
        if hit:
            now = time.time()
            if now - self._last_clamp_warn > 1.0:
                self._last_clamp_warn = now
                print(f"[yam-abc] action clamped to +-{step} rad/step "
                      f"({self.max_joint_speed} rad/s) — the policy "
                      f"commanded a large jump (likely out of distribution)", flush=True)
        return out

    def joint_snapshot(self) -> dict:
        """Per-unit view of the last read — no CAN access. Same shape as
        ``ControlLoop.joint_snapshot`` so the GUI's arm monitor renders a rollout
        unchanged: ``follower`` is the observed pose and ``leader_cmd`` the commanded
        policy action. The leader rows are always None — autonomy has no leader.
        """
        def _vec(a):
            return None if a is None else [round(float(x), 4) for x in list(a)]

        out: dict = {}
        for u in self.units:
            o = self._last_obs.get(u.name) or {}
            jp, gp = o.get("joint_pos"), o.get("gripper_pos")
            out[u.name] = {
                "follower": (None if jp is None else _vec(jp) + (_vec(gp) or [])),
                "leader_cmd": _vec(self._last_action.get(u.name)),
                "leader_buttons": None,
                "leader_grip": None,
                "leader_raw": None,
                "leader_cal": None,
            }
        return out

    def validate_state_dim(self) -> None:
        """At startup, compare the server's advertised state_dim to what this station
        produces and raise a clear message on mismatch."""
        expected = (getattr(self.client, "metadata", None) or {}).get("state_dim")
        if not expected:
            return  # server didn't advertise it — nothing to check
        _, per_arm_obs = self._read()
        actual = int(contract.build_state(per_arm_obs).shape[0])
        if actual != expected:
            raise ValueError(
                f"state-dim mismatch: the policy server expects {expected}-D state but this "
                f"station produces {actual}-D from {len(self.units)} arm(s). Likely the model "
                f"is {'single' if expected == 7 else expected // 7}-arm while the station is "
                f"configured for {len(self.units)}. Match the arm count in configs/station_yam.yaml."
            )

    def _step(self) -> None:
        frames, per_arm_obs = self._read()
        obs = contract.build_observation(per_arm_obs, frames, self.cameras, self.prompt)
        action_row = self.client.get_action(obs)
        if self._estopped:
            return
        cmds = contract.split_action(action_row, self._arm_dims)
        if self.max_joint_step > 0:
            cmds = self._clamp_cmds(cmds, per_arm_obs)
        self._command(cmds)
        # Log the rollout (same schema as teleop collection) if a recorder is on.
        rec = self._recorder
        if rec is not None and rec.is_recording:
            rec.tick(
                {u.name: cmds[i] for i, u in enumerate(self.units)},
                {u.name: per_arm_obs[i] for i, u in enumerate(self.units)},
                frames,
            )

    def _sleep_to_rate(self, t0: float) -> None:
        elapsed = time.time() - t0
        if self.dt > 0 and elapsed < self.dt:
            time.sleep(self.dt - elapsed)
        total = time.time() - t0
        if total > 0:
            inst = 1.0 / total
            self.actual_hz = inst if self.actual_hz == 0.0 else 0.8 * self.actual_hz + 0.2 * inst

    # --- safety ramp -------------------------------------------------------
    def ramp_to_first_action(self, ramp_seconds: float = 1.0) -> None:
        """Ease every follower from its current pose to the policy's first
        predicted action so it never snaps. Queries the server once, then
        linearly interpolates over ``ramp_seconds`` at the loop rate."""
        frames, per_arm_obs = self._read()
        obs = contract.build_observation(per_arm_obs, frames, self.cameras, self.prompt)
        target = np.asarray(self.client.get_action(obs), dtype=np.float32)
        starts = [contract.arm_state(o) for o in per_arm_obs]
        start = np.concatenate(starts)
        # Reset so the real loop re-queries from the true first frame.
        self.client.reset()
        steps = max(1, int(ramp_seconds / self.dt)) if self.dt > 0 else 1
        for i in range(1, steps + 1):
            if self._estopped:
                return
            t0 = time.time()
            alpha = i / steps
            row = start * (1.0 - alpha) + target * alpha
            self._command(contract.split_action(row, self._arm_dims))
            self._sleep_to_rate(t0)

    def move_to_home(self, home_pose, seconds: float = 2.0) -> None:
        """Ramp the followers from their current pose to a fixed home pose (e.g. the
        demonstrations' start pose) BEFORE the policy takes over, so the first
        observation is in-distribution. Not recorded (doesn't call the recorder)."""
        frames, per_arm_obs = self._read()
        start = np.concatenate([contract.arm_state(o) for o in per_arm_obs]).astype(np.float32)
        target = np.asarray(home_pose, dtype=np.float32).reshape(-1)
        if target.shape != start.shape:
            raise ValueError(f"home_pose must have shape {start.shape}, got {target.shape}")
        steps = max(1, int(seconds / self.dt)) if self.dt > 0 else 1
        for i in range(1, steps + 1):
            if self._estopped:
                return
            t0 = time.time()
            a = i / steps
            row = start * (1.0 - a) + target * a
            self._command(contract.split_action(row, self._arm_dims))
            self._sleep_to_rate(t0)

    # --- run modes ---------------------------------------------------------
    def run_for(self, seconds: float, ramp_seconds: float = 1.0, home_pose=None) -> None:
        """Synchronous rollout for ``seconds`` (CLI path). ``home_pose`` ramps the
        followers there first, after the cameras are up so the arm monitor and the
        recorder see the same stream the rollout will."""
        self._start_cameras()
        try:
            self.validate_state_dim()
            if home_pose:
                self.move_to_home(home_pose)
            print("[yam-abc] first inference: server is JIT-compiling, can take 1-2 min...", flush=True)
            self.ramp_to_first_action(ramp_seconds)
            end = time.time() + seconds
            while time.time() < end and not self._estopped:
                t0 = time.time()
                self._step()
                self._sleep_to_rate(t0)
        finally:
            self._stop_cameras()

    def start(self, ramp_seconds: float = 1.0) -> None:
        """Threaded rollout (GUI path)."""
        if self._running:
            return
        self._start_cameras()
        self.validate_state_dim()
        print("[yam-abc] first inference: server is JIT-compiling, can take 1-2 min "
              "(arm eases to the start pose then pauses — not a hang)...", flush=True)
        self.ramp_to_first_action(ramp_seconds)
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # Capture any rollout-thread crash so the GUI status feed can show why it stopped.
        try:
            while self._running and not self._estopped:
                t0 = time.time()
                self._step()
                self._sleep_to_rate(t0)
        except Exception as e:  # noqa: BLE001
            # Stop commanding (no estop/home) — the arm holds at its last pose. Record why.
            self.last_error = (
                f"{type(e).__name__}: {e} — arm HOLDING at last commanded pose "
                f"(not homed). E-STOP to release, or Stop + re-Start once the server is back."
            )
            self._running = False

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # Nothing is being commanded any more; drop the cached action so the GUI's
        # arm monitor doesn't keep showing a live-looking one (ControlLoop does the same).
        self._last_action = {}
        self._stop_cameras()

    def estop(self) -> None:
        """Safety stop: cease commanding and hard-stop every arm."""
        self._estopped = True
        self._running = False
        for u in self.units:
            u.robot.stop()
        # Join only after the arms are stopped, so the rollout thread can't slip one
        # more command in between (it bails on _estopped at the top of _step).
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._last_action = {}
