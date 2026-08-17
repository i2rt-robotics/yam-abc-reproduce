"""CollectSession: the one stateful live object behind the Collect tab.

Owns the ControlLoop, the EpisodeRecorder, and the CameraHub, and implements the
continuous-loop + sync-gate model:

  * "Go live" (first Start Teleop): apply the edited config, bring the CAN buses
    up (hardware), build the devices, start the loop, and enable sync.
  * Once live, the loop runs continuously so the teaching-handle buttons stay
    readable. The **top button** and the GUI toggle both flip ``sync_enabled``
    (follower mirrors or not); the **second button** and the GUI toggle both
    start/stop recording. State is reported via ``status()`` so the GUI mirrors
    the hardware buttons and vice versa.

Hardware isn't touched until go-live, so the operator can edit the Station rail
first. The mock path has no CAN/buttons — sync/record are GUI-driven only.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from ..camera.worker import CameraWorker
from ..config import (
    ControllerConfig,
    StationConfig,
    controller_channel_for,
    gello_variant,
    is_passive_gello,
    leader_joint_signs_for,
    robot_channel_for,
)
from ..data.recorder import EpisodeRecorder, task_slug
from ..data.schema import WRITE_COMPLETE_FLAG
from ..robot.can_bus import check_can_up, reset_can_buses
from ..runtime import build_arm_units, build_cameras_from_config
from ..teleop.loop import ControlLoop
from .camera_hub import CameraHub


class CollectSession:
    def __init__(self, cfg: StationConfig, mock: bool = False):
        self.mock = mock
        self.cfg = cfg
        # Restore the last task name so a GUI restart pre-fills the Task field with what
        # you were collecting, instead of the station yaml's default.
        last = self._load_last_task()
        if last:
            self.cfg.task_name = last
        self.hub = CameraHub()
        self.live = False
        # Live for autonomy: followers only, no leaders and no teleop loop. Set by a
        # deploy go-live and by _release_leaders(); teleop can't resume until a rebuild.
        self.followers_only = False
        self.powered_off = False
        # E-STOP latch at session level. The teleop ControlLoop has its own, but a
        # followers-only session has no loop to hold one, so the GUI would show
        # "idle" after an E-STOP. Cleared by go_live() and reset_session().
        self._estopped = False
        self.last_can_msg = ""
        # Serializes sync transitions so a hardware button and a GUI click can't
        # both run engage() at once.
        self._sync_lock = threading.Lock()
        # Cameras can be connected (preview) without teleop; robots/loop are built
        # at go-live. Nothing here touches hardware.
        self.units: list = []
        self.loop = self.recorder = None
        self.deploy_loop = None  # autonomous policy rollout (Deploy tab)
        self.deploy_recorder = None  # logs the rollout to a separate save path
        self.workers: list = []
        self.cameras_connected = False
        self._cam_cfg: dict = {}
        self._cam_sig: list = []

    # --- last-task persistence (survive a GUI restart) --------------------
    def _last_task_path(self) -> Path:
        return Path(self.cfg.save_root).parent / ".last_task"

    def _load_last_task(self) -> str | None:
        try:
            return self._last_task_path().read_text().strip() or None
        except OSError:
            return None

    def _save_last_task(self, name: str) -> None:
        name = (name or "").strip()
        if not name:
            return
        try:
            p = self._last_task_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(name)
        except OSError:
            pass

    # --- cameras (independent of teleop; feed previews) -------------------
    @staticmethod
    def _camera_sig(cfg: StationConfig) -> list:
        return [(c.name, c.type, c.role, c.serial, c.mode) for c in cfg.cameras]

    def connect_cameras(self, cfg: StationConfig) -> None:
        """Open the cameras and start their capture threads (feeding previews),
        independent of teleop/CAN. Re-opens only if the camera config changed, so
        a later go-live reuses already-previewing cameras. A device opens once."""
        if self.cameras_connected and self._camera_sig(cfg) == self._cam_sig:
            self.cfg = cfg
            return
        if self.live:
            # Reopening would stop the workers the running loop reads from (below).
            raise RuntimeError("stop teleop or the rollout before changing the camera set")
        if self.loop is not None:
            self.loop.stop()  # the loop shares these workers; stop before swapping
        self._disconnect_cameras()
        self.cfg = cfg
        drivers = build_cameras_from_config(cfg, mock=self.mock)
        self.workers = [CameraWorker(d, on_frame=self.hub.update) for d in drivers]
        for w in self.workers:
            w.start()
        self._cam_cfg = {c.name: c for c in cfg.cameras}
        self._cam_sig = self._camera_sig(cfg)
        self.cameras_connected = True

    def _disconnect_cameras(self) -> None:
        for w in self.workers:
            w.stop()
        self.workers = []
        self.cameras_connected = False

    # --- device stack -----------------------------------------------------
    @staticmethod
    def _needed_channels(cfg: StationConfig, followers_only: bool) -> list[str]:
        """The CAN buses go-live is about to open, mirroring ``build_arm_units``: one
        per follower, plus the controller bound to each follower unless followers-only.
        Iterating ``cfg.robot.controllers`` instead would also list a controller that
        isn't bound to any robot and so never gets opened."""
        robots = cfg.robot.robots or [cfg.robot.active_robot()]
        needed = [robot_channel_for(r) for r in robots]
        if not followers_only:
            needed += [controller_channel_for(cfg.robot.controller_for(r)) for r in robots]
        return needed

    def _build_teleop_stack(self, cfg: StationConfig) -> None:
        """Build the teleop loop and recorder over the already-built units and the
        already-connected cameras, and wire the button callbacks."""
        # Pass the running workers so the loop shares them (doesn't reopen devices).
        self.loop = ControlLoop(self.units, self.workers, cfg.control_hz)
        self.recorder = EpisodeRecorder(
            cfg.save_root, cfg, self.workers, arm_names=[u.name for u in self.units]
        )
        self.loop.attach_recorder(self.recorder)
        self.loop.on_sync_button = self.toggle_sync
        self.loop.on_record_button = self.toggle_record

    def go_live(self, cfg: StationConfig, followers_only: bool = False) -> None:
        """Apply ``cfg``, bring CAN up (hardware), build the robots over the connected
        cameras, and (unless ``followers_only``) start the teleop loop with sync on.

        ``followers_only`` is the autonomy bring-up: it opens one bus per follower
        instead of two, and deliberately builds no teleop loop -- ``enable_sync``'s
        ``engage_all()`` would ramp every follower to wherever its leader is sitting,
        which is a real arm motion nobody asked for right before a policy takes over.
        """
        if self.loop is not None:
            self.loop.stop()
        if not self.mock:
            ok, msg = reset_can_buses()
            self.last_can_msg = msg
            # Pre-flight: every bus we're about to open must be up before i2rt opens
            # it, else fail with a clear message instead of deep inside the driver.
            needed = self._needed_channels(cfg, followers_only)
            down = check_can_up(needed)
            if down:
                raise RuntimeError(
                    f"CAN interface(s) not up: {', '.join(down)}. "
                    f"Run scripts/setup_can_sudoers.sh once, check the arm is powered, "
                    f"and that udev names match (expected {needed})."
                )
        self.connect_cameras(cfg)  # reuse preview cameras if already up
        try:
            self.cfg = cfg
            self.units = build_arm_units(cfg, mock=self.mock, followers_only=followers_only)
            self.followers_only = followers_only
            if not followers_only:
                self._build_teleop_stack(cfg)
            self.live = True
            self.powered_off = False
            self._estopped = False  # re-armed: Start Teleop is a documented E-STOP recovery
            if not followers_only:
                self.enable_sync()
                self.loop.start()
        except Exception:
            # Release the half-built units on failure so a retry can rebuild. Cameras stay.
            self._teardown_units()
            self.live = False
            raise

    def _release_leaders(self) -> None:
        """Close the leader CAN buses while the followers stay energized, so an
        autonomous rollout runs on the follower buses only.

        The agents are kept, not dropped: E-STOP and Power Off Arms reach the hardware
        through ``self.units``, and a motorized lead arm keeps its bus (releasing it
        would drop gravity comp and the arm would sag). ``release_leader`` poisons the
        agent either way, so nothing can command a follower from a frozen leader reading.
        """
        for u in self.units:
            release = getattr(u.agent, "release_leader", None)
            if not callable(release):
                continue
            try:
                if not release():
                    logging.warning(
                        "%s leader is a motorized lead arm and keeps its CAN bus — it "
                        "needs gravity compensation to stay up. Only the passive-GELLO "
                        "leaders are released for autonomy.", u.name,
                    )
            except Exception:  # noqa: BLE001 — a stuck leader must not block the rollout
                logging.exception("failed to release the %s leader", u.name)
        self.followers_only = True

    def _teardown_units(self) -> None:
        """Stop the teleop loop and release the robot units (drop i2rt handles); leave
        cameras. Used by go_live's failure path and reset_session."""
        if self.loop is not None:
            try:
                self.loop.stop()
            except Exception:  # noqa: BLE001
                pass
            self.loop = None
        self.recorder = None
        # Close the leader buses too, or the GELLO reader thread and its socket outlive
        # the session and the next go-live opens a second socket on the same bus.
        self._release_leaders()
        for u in self.units:
            # Relax to zero torque first (arm goes limp, safe to handle / re-home),
            # then drop the i2rt handle. stop() would instead hold the pose.
            for meth in ("relax", "stop"):
                try:
                    getattr(u.robot, meth)()
                    break
                except Exception:  # noqa: BLE001
                    continue
        self.units = []
        self.followers_only = False

    # --- teleop / sync ----------------------------------------------------
    def start_teleop(self, cfg: StationConfig | None = None) -> None:
        """Go live on the first call (applying ``cfg``); afterwards just re-enable
        sync. The physical top button toggles sync once live."""
        if self.live and self.followers_only:
            # No leaders and no teleop loop in this session — re-enabling sync would
            # silently do nothing. The leaders come back only on a rebuild.
            raise RuntimeError(
                "session is live for autonomy (follower buses only) — Stop the rollout "
                "and click Reset Session before starting teleop"
            )
        if not self.live:
            self.go_live(cfg or self.cfg)
        else:
            self.enable_sync()

    def stop_teleop(self) -> None:
        # Disable sync but keep the loop running so the buttons stay live.
        self.disable_sync()

    def enable_sync(self) -> None:
        with self._sync_lock:
            # Never (re)engage on an estopped loop — it's a latch; recovery is a
            # full rebuild via go_live(), not enable_sync().
            if self.loop is None or self.loop.sync_enabled or self.loop.estopped:
                return
            # Ease every follower to its leader before mirroring so it never snaps
            # (i2rt slow_move pattern); no-op on the mock.
            self.loop.engage_all()
            # If E-STOP fired during the ramp, engage_all() aborted — don't (re)enable
            # sync, or the loop would start mirroring right after a stop.
            if not self.loop.estopped:
                self.loop.sync_enabled = True

    def disable_sync(self) -> None:
        with self._sync_lock:
            if self.loop is not None:
                self.loop.sync_enabled = False

    def toggle_sync(self) -> None:
        if self.loop is not None and self.loop.sync_enabled:
            self.disable_sync()
        else:
            self.enable_sync()

    def zero_gello(self, side: str) -> dict:
        """Hardware-zero a passive-GELLO leader's joint encoders and gripper at its current
        pose (writes each encoder's EEPROM via i2rt ``reset_zero_position``). Hold the leader
        at the pose that should map to the follower's home, with the trigger released.

        Safe to run while live: sync is disabled first so the instantaneous
        reported-angle change can't jerk the follower, and the zero (being in
        hardware EEPROM) takes effect immediately for the running loop — no rebuild.
        Returns ``{ok, message, ...}`` so the GUI can show the outcome.
        """
        if self.mock:
            return {"ok": False, "message": "mock mode: no hardware to zero"}
        if side not in ("left", "right"):
            return {"ok": False, "message": f"unknown side {side!r} (expected left/right)"}

        sync_was_on = bool(self.loop and self.loop.sync_enabled)
        if sync_was_on:
            self.disable_sync()

        from ..robot.passive_gello import PassiveGelloLeader

        cfg = self.cfg
        # Use the leader this station has on that side, so its own bus and signs apply: a
        # mobile GELLO or an assigned channel isn't reachable via the passive_gello_* defaults.
        controller = next(
            (c for c in cfg.robot.controllers if is_passive_gello(c.type) and c.type.endswith(side)),
            ControllerConfig(type=f"passive_gello_{side}"),
        )
        channel = controller_channel_for(controller)
        n = cfg.robot.num_arm_joints
        grip = cfg.robot.leader_gripper
        gello_type, gello_side = gello_variant(controller.type)
        try:
            lead = PassiveGelloLeader(
                channel=channel,
                num_arm_joints=n,
                gripper_config=tuple(grip) if grip else (n, 0.7, 0.0),
                joint_signs=leader_joint_signs_for(controller, cfg.robot.leader_joint_signs),
                gello_type=gello_type,
                side=gello_side,
            )
        except Exception as e:  # noqa: BLE001 — surface a readable reason to the UI
            return {
                "ok": False,
                "message": f"{type(e).__name__}: {e}",
                "sync_disabled": sync_was_on,
            }
        try:
            devices = lead.hardware_zero()
        finally:
            lead.stop()
        return {
            "ok": True,
            "message": (
                f"zeroed {channel} encoders {devices} (arm joints + gripper) at current pose"
            ),
            "devices": devices,
            "sync_disabled": sync_was_on,
        }

    # --- recording --------------------------------------------------------
    def start_recording(self, task_name: str) -> str:
        if self.recorder is None:
            raise RuntimeError("not live — Start Teleop before recording")
        if not (task_name or "").strip():
            raise ValueError("set a Task name before recording")
        self._save_last_task(task_name)  # remember for the next GUI restart
        return str(self.recorder.start(task_name))

    def stop_recording(self) -> dict:
        if self.recorder is None:
            raise RuntimeError("not recording — no active session")
        path = self.recorder.stop()
        if path is None:  # empty episode was discarded (e.g. phantom button tap)
            return {"path": None, "frames": 0}
        return {"path": str(path), "frames": self.recorder.frame_count}

    def toggle_record(self) -> None:
        """Second-button / GUI toggle. Uses the rail's Task name; refuses to start a
        nameless recording (an empty task poisons the dataset + training prompt)."""
        if self.recorder is None:
            return
        if self.recorder.is_recording:
            self.stop_recording()
        elif (self.cfg.task_name or "").strip():
            self.start_recording(self.cfg.task_name)
        else:
            print("[yam-abc] recording ignored — set a Task name in the Station rail first", flush=True)

    # --- safety -----------------------------------------------------------
    # --- autonomous deployment -------------------------------------------
    def start_deploy(
        self,
        host: str,
        port: int,
        prompt: str,
        cfg: "StationConfig | None" = None,
        open_loop_horizon: int = 15,
        record: bool = True,
        save_root: str = "data/rollouts",
        home_pose: list | None = None,
        rtc: bool = False,
        rtc_prefix_length: int = 4,   # P: frozen prefix rows
        rtc_action_horizon: int = 15,  # H: rows streamed per chunk (= ABC execute_chunk_dim)
        rtc_lead_steps: int = 4,       # L: re-query when L rows remain (= ABC rtc_inference_lead_steps)
        max_joint_speed: float = 1.5,   # safety: max arm-joint speed (rad/s); <=0 off
    ) -> None:
        """Drive the followers from a policy server. Reuses the previewing cameras.
        Connects to a server at host:port (127.0.0.1 for same-machine). When ``record``
        (default), logs the rollout as one episode to ``save_root`` (separate from
        teleop collection) using the same schema/writer.

        Autonomy needs the follower buses only, so a cold start opens just those (one
        per arm, no leaders); a session already live from teleop hands the arms over by
        closing the leader buses, leaving the followers energized throughout."""
        from ..data.recorder import EpisodeRecorder
        from ..deploy.client import WebsocketPolicyClient
        from ..deploy.loop import DeployLoop

        cfg = cfg or self.cfg
        if self.deploy_loop is not None:
            raise RuntimeError("a rollout is already running — press Stop first")
        # Check the home pose against this station before anything moves: move_to_home
        # would otherwise only raise after the CAN reset, both arm builds and the
        # gripper sweep.
        if home_pose:
            expected = len(cfg.robot.robots or [cfg.robot.active_robot()]) * (
                cfg.robot.num_arm_joints + 1
            )
            if len(home_pose) != expected:
                raise ValueError(
                    f"home pose has {len(home_pose)} values but this station needs "
                    f"{expected} ([joints..., gripper] per arm, in robots order)"
                )
        if not self.live:
            self.go_live(cfg, followers_only=True)  # CAN up + followers + cameras
        elif not self.followers_only:
            # Live from teleop: stop mirroring, drop the teleop stack, and hand the
            # arms over by releasing the leaders. The followers keep holding torque.
            self.disable_sync()
            if self.loop is not None:
                self.loop.stop()  # shared camera workers stay running
            self.loop = self.recorder = None
            self._release_leaders()

        client = WebsocketPolicyClient(host=host, port=int(port),
                                       open_loop_horizon=int(open_loop_horizon),
                                       rtc=rtc, rtc_prefix_length=rtc_prefix_length,
                                       rtc_action_horizon=rtc_action_horizon,
                                       rtc_lead_steps=rtc_lead_steps)
        self.deploy_loop = DeployLoop(
            self.units, self.workers, client, prompt=prompt, control_hz=self.cfg.control_hz,
            max_joint_speed=max_joint_speed,
        )
        # Catch a station/model arm-count mismatch before the arms move for a rollout
        # that can't run (start() re-checks; it's a read, not a command).
        self.deploy_loop.validate_state_dim()
        # Move to the demos' start pose BEFORE the policy runs so the first obs is
        # in-distribution (done before recording so the homing motion isn't logged).
        if home_pose:
            self.deploy_loop.move_to_home(home_pose)
        if record:
            self.deploy_recorder = EpisodeRecorder(
                save_root, cfg, self.workers, arm_names=[u.name for u in self.units]
            )
            self.deploy_recorder.start(task_name=(prompt or "deploy"))
            self.deploy_loop.attach_recorder(self.deploy_recorder)
        self.deploy_loop.start()  # ramps to the first action, then runs threaded

    def stop_deploy(self) -> dict:
        """Stop the rollout and, if recording, save the episode. Returns the saved
        path + frame count (empty if not recording)."""
        if self.deploy_loop is not None:
            self.deploy_loop.stop()
            self.deploy_loop = None
        out: dict = {}
        if self.deploy_recorder is not None:
            if self.deploy_recorder.is_recording:
                path = self.deploy_recorder.stop()
                out = {"path": str(path), "frames": self.deploy_recorder.frame_count}
            self.deploy_recorder = None
        return out

    def set_deploy_prompt(self, prompt: str) -> None:
        if self.deploy_loop is not None:
            self.deploy_loop.set_prompt(prompt)

    def estop(self) -> None:
        self._estopped = True
        if self.deploy_loop is not None:
            self.deploy_loop.estop()
            self.deploy_loop = None
        if self.deploy_recorder is not None:
            self.deploy_recorder.abort()  # discard the partial rollout episode
            self.deploy_recorder = None
        if self.loop is not None:
            self.loop.estop()
        if self.recorder is not None:
            # Discard any half-recorded episode so E-STOP never leaves a partial,
            # uncompleted folder behind.
            self.recorder.abort()
        # Hard-stop every arm directly as well. A followers-only session that has already
        # stopped its rollout has no loop and no deploy_loop left to route this through,
        # and the arms are still energized holding the last commanded pose.
        for u in self.units:
            try:
                u.robot.stop()
            except Exception:  # noqa: BLE001 — one bad arm must not skip the others
                logging.exception("E-STOP: failed to stop the %s arm", u.name)
        # Leave the live state so E-STOP is recoverable. Setting live=False makes start_teleop() re-arm.
        self.live = False
        self.followers_only = False

    def power_off_arms(self) -> dict:
        """Stop control, then physically disable every follower motor.

        This is intentionally a separate, explicit action from E-STOP: the arm
        loses holding torque and can fall or move freely once it succeeds.
        """
        if self.deploy_loop is not None:
            # stop() joins the rollout thread; estop() alone doesn't wait for it, and a
            # command slipping in between the per-joint motor_off frames below would
            # re-enable a joint.
            try:
                self.deploy_loop.stop()
            except Exception:  # noqa: BLE001
                pass
        self.estop()
        if self.loop is not None:
            try:
                self.loop.stop()
            except Exception:  # noqa: BLE001
                pass
            self.loop = None
        self.recorder = None
        arms = {}
        for unit in self.units:
            try:
                power_off = getattr(unit.robot, "power_off", None)
                if not callable(power_off):
                    raise RuntimeError("this robot driver has no power_off method")
                arms[unit.name] = power_off()
            except Exception as exc:  # noqa: BLE001
                arms[unit.name] = {"disabled": [], "failed": {"driver": str(exc)}}
        self.units = []
        self.live = False
        self.powered_off = bool(arms) and all(not result["failed"] for result in arms.values())
        return {"ok": self.powered_off, "arms": arms}

    def reset_session(self) -> dict:
        """Tear down loops/recorders/units, reset the CAN buses, and clear live/estop so
        the next Start Teleop rebuilds from scratch. Cameras are kept. Used to recover from
        E-STOP or a failed go-live without restarting the GUI."""
        for obj, meth in ((self.deploy_loop, "stop"), (self.deploy_recorder, "abort"),
                          (self.recorder, "abort")):
            if obj is not None:
                try:
                    getattr(obj, meth)()
                except Exception:  # noqa: BLE001
                    pass
        self.deploy_loop = self.deploy_recorder = None
        # Stop the teleop loop + release the robot units (drop i2rt handles) before the reset.
        self._teardown_units()
        can_msg = "(mock)"
        if not self.mock:
            try:
                _, can_msg = reset_can_buses()
                self.last_can_msg = can_msg
            except Exception as e:  # noqa: BLE001
                can_msg = f"CAN reset failed: {e}"
        self.live = False
        self._estopped = False
        return {"ok": True, "can": can_msg}

    # --- queries ----------------------------------------------------------
    def list_episodes(self, task_name: str | None = None) -> list[str]:
        """Completed episodes under the CURRENT task's folder
        (``save_root/<task_slug>/<ep>`` with a ``write_complete.flag``).

        Counts the recorder's task when one is active/known — the GUI form task
        only reaches ``self.cfg`` on Start Teleop, so counting cfg alone shows 0
        for episodes just recorded under a different (or post-restart) name."""
        task = task_name or getattr(self.recorder, "task_name", None) or self.cfg.task_name
        task_dir = Path(self.cfg.save_root) / task_slug(task)
        if not task_dir.exists():
            return []
        return sorted(str(p) for p in task_dir.iterdir() if (p / WRITE_COMPLETE_FLAG).exists())

    def camera_descriptors(self) -> list[dict]:
        """One row per video stream (a stereo cam yields two), enriched with the
        configured device specs so the GUI can render the config rail and build
        per-eye preview URLs without a second request."""
        out = []
        for cam in self.workers:
            c = self._cam_cfg.get(cam.name)
            for k in cam.image_keys():
                eye = None if k == "rgb" else k
                out.append(
                    {
                        "name": cam.name,
                        "role": cam.role,
                        "mode": cam.mode.value,
                        "eye": eye,
                        "type": (c.type if c else "mock"),
                        "width": (c.width if c else 0),
                        "height": (c.height if c else 0),
                        "fps": (c.fps if c else max(1, int(round(self.cfg.control_hz)))),
                    }
                )
        return out

    def status(self) -> dict:
        loop, rec, dep = self.loop, self.recorder, self.deploy_loop
        return {
            "live": self.live,
            # Live for autonomy: no leaders, no teleop loop. The GUI greys out the
            # teleop/record controls, which would otherwise 409 on click.
            "followers_only": self.followers_only,
            # "teleop_running" now means sync is enabled (follower mirroring).
            "teleop_running": bool(loop and loop.sync_enabled),
            "recording": bool(rec and rec.is_recording),
            "estopped": self._estopped or bool(loop and loop.estopped),
            "powered_off": self.powered_off,
            "current_task": (rec.task_name if rec else None),
            "episodes_done": len(self.list_episodes()),
            # Live teaching-handle inputs for the GUI indicators.
            "buttons": (list(loop.buttons) if loop else [False, False]),
            "trigger": (round(loop.trigger, 3) if loop else 0.0),
            "cameras": self.camera_descriptors(),
            # Per-unit follower joint vector + leader command, for calibration/debug.
            # A rollout reports its own (same shape, no leader rows) — keyed on the loop
            # existing, not on `deploying`, which is still False while homing/ramping.
            "arms": (dep.joint_snapshot() if dep else (loop.joint_snapshot() if loop else {})),
            # Unit roster + most recent teleop-loop error, for the status badge.
            "units": [u.name for u in self.units],
            "last_error": (loop.last_error if loop else None),
            # Autonomous-deploy state for the Deploy tab.
            "deploying": bool(self.deploy_loop and self.deploy_loop.running),
            "deploy_hz": (round(self.deploy_loop.actual_hz, 1) if self.deploy_loop else 0.0),
            "deploy_error": (self.deploy_loop.last_error if self.deploy_loop else None),
            "deploy_recording": bool(self.deploy_recorder and self.deploy_recorder.is_recording),
            "deploy_frames": (self.deploy_recorder.frame_count if self.deploy_recorder else 0),
        }
