"""EpisodeRecorder: buffers per-key data during an episode, then flushes it
through a format writer and writes the completeness flag last.
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from ..camera.interface import CameraDriver, CameraFrame
from ..config import StationConfig
from .formats import get_writer
from .schema import (
    SCHEMA_VERSION,
    WRITE_COMPLETE_FLAG,
    CameraMeta,
    EpisodeMeta,
    action_gripper_key,
    action_joint_key,
    cam_image_key,
    cam_timestamp_key,
    gripper_pos_key,
    joint_pos_key,
)


def task_slug(task_name: str | None) -> str:
    """Filesystem-safe task folder name. Falls back to 'untitled'."""
    s = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in (task_name or "").strip().lower())
    s = "_".join(filter(None, s.split("_")))  # collapse repeats
    return s or "untitled"


class EpisodeRecorder:
    def __init__(
        self,
        save_root: str | Path,
        station: StationConfig,
        cameras: list[CameraDriver],
        arm_names: list[str],
        writer_name: str | None = None,
    ):
        self.save_root = Path(save_root)
        self.station = station
        self.cameras = cameras
        self.arms = list(arm_names)  # one schema prefix per driven arm
        self.n = station.robot.num_arm_joints
        self.writer_name = writer_name or station.data_format
        self.writer = get_writer(self.writer_name)
        self.is_recording = False
        self.frame_count = 0
        self.task_name: str | None = None
        self._buf: dict[str, list] | None = None
        self._dir: Path | None = None
        # Guards start/tick/stop/abort: the control-loop thread calls tick()
        # concurrently with the GUI thread calling start()/stop().
        self._lock = threading.Lock()

    def start(self, task_name: str) -> Path:
        with self._lock:
            if self.is_recording:
                raise RuntimeError("already recording")
            ts = time.strftime("%Y%m%d_%H%M%S")
            # Group episodes on disk by task slug so different tasks stay separate
            # (dataset: data/episodes/<task>/<ep>; rollout: data/rollouts/<policy>/<task>/<ep>)
            # and the converters/Review can target one task at a time.
            self._dir = self.save_root / task_slug(task_name) / f"{ts}_{uuid.uuid4().hex[:8]}"
            self._dir.mkdir(parents=True, exist_ok=True)
            self._buf = {}
            for arm in self.arms:
                self._buf[joint_pos_key(arm)] = []
                self._buf[gripper_pos_key(arm)] = []
                self._buf[action_joint_key(arm)] = []
                self._buf[action_gripper_key(arm)] = []
            for cam in self.cameras:
                for k in cam.image_keys():
                    self._buf[cam_image_key(cam.role, k)] = []
                self._buf[cam_timestamp_key(cam.role)] = []
            self.task_name = task_name
            self.frame_count = 0
            self.is_recording = True
            return self._dir

    def tick(
        self,
        actions: dict[str, np.ndarray],
        obs: dict[str, dict[str, np.ndarray]],
        frames: dict[str, CameraFrame],
    ) -> None:
        """Record one step. ``actions``/``obs`` are keyed by arm name; the camera
        streams are shared across arms."""
        with self._lock:
            if not self.is_recording or self._buf is None:
                return
            for arm in self.arms:
                action = np.asarray(actions[arm], dtype=np.float64).reshape(-1)
                o = obs[arm]
                self._buf[joint_pos_key(arm)].append(
                    np.asarray(o["joint_pos"], dtype=np.float64)
                )
                self._buf[gripper_pos_key(arm)].append(
                    np.asarray(o["gripper_pos"], dtype=np.float64)
                )
                self._buf[action_joint_key(arm)].append(action[: self.n].copy())
                self._buf[action_gripper_key(arm)].append(action[self.n : self.n + 1].copy())
            for cam in self.cameras:
                fr = frames[cam.name]
                for k in cam.image_keys():
                    # Copy: camera drivers may reuse one buffer across reads, so
                    # storing the reference would alias every frame to the latest.
                    self._buf[cam_image_key(cam.role, k)].append(
                        np.array(fr.images[k], copy=True)
                    )
                self._buf[cam_timestamp_key(cam.role)].append(float(fr.timestamp_ms))
            self.frame_count += 1

    def stop(self) -> Path:
        with self._lock:
            if not self.is_recording or self._buf is None or self._dir is None:
                raise RuntimeError("not recording")
            # Freeze recording first so any in-flight tick() is a no-op.
            self.is_recording = False
            buf, out_dir = self._buf, self._dir
            # Capture inside the lock, before a concurrent start() can reset them.
            frame_count, task_name = self.frame_count, self.task_name

        out: dict = {}
        for key, val in buf.items():
            if "-images-" in key:
                out[key] = val  # list of frames
            elif key.endswith("-timestamp"):
                out[key] = np.asarray(val, dtype=np.float64)
            else:
                out[key] = np.stack(val) if val else np.empty((0,))

        meta = EpisodeMeta(
            yam_abc_reproduce_version=_yam_abc_reproduce_version(),
            schema_version=SCHEMA_VERSION,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            task_name=task_name or "",
            arm_names=self.arms,
            num_arm_joints=self.n,
            control_hz=self.station.control_hz,
            cameras=self._camera_metas(out),
            num_frames=frame_count,
        )
        if frame_count == 0:
            shutil.rmtree(out_dir, ignore_errors=True)
            print(f"[yam-abc] discarded empty episode {out_dir.name} (0 frames)", flush=True)
            return None
        self.writer.write_episode(out_dir, meta, out)
        # Completeness marker, written last so partial folders are detectable.
        (out_dir / WRITE_COMPLETE_FLAG).write_text("")
        return out_dir

    def abort(self) -> Path | None:
        """Discard the in-progress episode without writing a completeness flag.

        Used on E-STOP: the half-recorded directory is removed so it can't be
        mistaken for a finished episode."""
        with self._lock:
            if not self.is_recording:
                return None
            self.is_recording = False
            out_dir = self._dir
            self._buf = None
            self._dir = None
        if out_dir is not None and out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        return out_dir

    def _camera_metas(self, out: dict) -> list[CameraMeta]:
        cfg_by_name = {c.name: c for c in self.station.cameras}
        metas = []
        for cam in self.cameras:
            keys = cam.image_keys()
            sample = out.get(cam_image_key(cam.role, keys[0]), [])
            h, w = (sample[0].shape[:2] if sample else (0, 0))
            c = cfg_by_name.get(cam.name)
            metas.append(
                CameraMeta(
                    name=cam.name,
                    type=(c.type if c else "mock"),
                    role=cam.role,
                    mode=cam.mode.value,
                    image_keys=keys,
                    width=int(w),
                    height=int(h),
                    fps=(c.fps if c else max(1, int(round(self.station.control_hz)))),
                )
            )
        return metas


def _yam_abc_reproduce_version() -> str:
    from .. import __version__

    return __version__
