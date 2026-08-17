"""Convert YAM-ABC-Reproduce default episodes into a LeRobot dataset.

Implements the DatasetWriter role (begin / add_episode / finalize). Dataset
writing and video encoding are delegated to LeRobot itself (``LeRobotDataset``);
the only non-trivial logic here is mapping our schema to LeRobot features and
aligning per-camera streams onto a single reference timeline.

Feature mapping:
    <role>-images-<key>  -> observation.images.<role>_<key>   (video)
    <arm>-joint_pos + <arm>-gripper_pos       -> observation.state  (concat)
    action-<arm>-joint + action-<arm>-gripper -> action             (concat)
"""

from __future__ import annotations

import numpy as np

from .. import codec
from ..schema import (
    EpisodeMeta,
    action_gripper_key,
    action_joint_key,
    cam_image_key,
    cam_timestamp_key,
    gripper_pos_key,
    joint_pos_key,
)
from .base import register


def _lerobot_image_name(role: str, img_key: str) -> str:
    return f"observation.images.{role}_{img_key}"


def _nearest_indices(ref_ts: np.ndarray, src_ts: np.ndarray) -> np.ndarray:
    """Index into src for each ref timestamp (nearest-neighbor sample-and-hold).

    ``src_ts`` must be sorted ascending (capture timestamps are monotonic). Uses
    searchsorted + a neighbor compare — O(n log m) time, O(n) memory — instead of
    materializing the full (n_ref x n_src) difference matrix. Ties resolve to the
    earlier index, matching argmin's first-minimum behavior."""
    src_ts = np.asarray(src_ts)
    if src_ts.shape[0] <= 1:
        return np.zeros(len(ref_ts), dtype=int)
    idx = np.clip(np.searchsorted(src_ts, ref_ts), 1, src_ts.shape[0] - 1)
    prefer_left = (ref_ts - src_ts[idx - 1]) <= (src_ts[idx] - ref_ts)
    return idx - prefer_left.astype(int)


class LeRobotFormat:
    def __init__(self):
        self._repo_id = None
        self._out = None
        self._ds = None
        self._fps = 30

    def begin(self, repo_id: str, out: str | None = None) -> None:
        self._repo_id = repo_id
        self._out = out

    def _build_features(self, meta: EpisodeMeta) -> dict:
        feats: dict = {}
        for cam in meta.cameras:
            for k in cam.image_keys:
                feats[_lerobot_image_name(cam.role, k)] = {
                    "dtype": "video",
                    "shape": (cam.height, cam.width, 3),
                    "names": ["height", "width", "channels"],
                }
        # Concatenate all arms: [<arm0 joints..> <arm0 gripper> <arm1 joints..> ..].
        arms = meta.arm_names or ["left"]
        names: list[str] = []
        for a in arms:
            names += [f"{a}_joint_{i}" for i in range(meta.num_arm_joints)] + [f"{a}_gripper"]
        state_dim = len(names)
        feats["observation.state"] = {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": names,
        }
        feats["action"] = {"dtype": "float32", "shape": (state_dim,), "names": names}
        return feats

    @staticmethod
    def _rgb_encoder():
        """Hand LeRobot a hardware encoder only once one has really been proven to work.

        LeRobot's own ``vcodec="auto"`` is not usable here: it selects on
        ``get_codec(name) is not None`` (``lerobot/datasets/pyav_utils.py``), a descriptor
        lookup that answers yes to ``h264_nvenc`` on any box whose FFmpeg was *built* with
        NVENC -- GPU-less ones included, where the encode then dies mid-conversion with
        ``Operation not permitted: avcodec_open2(h264_nvenc)``. Our probe opens the encoder
        for real, so pass its answer explicitly.

        With no hardware we return None and leave LeRobot on its libsvtav1 default, which is
        a better software codec than anything we would substitute.

        ``crf`` is retuned for NVENC because LeRobot feeds it straight through as an H.264
        ``qp`` (``configs/video.py:get_codec_options``), and its default of 30 is an AV1
        quality number -- as a qp it is visibly worse. 23 is the rough H.264 equivalent.

        The parameter is newer than our ``lerobot>=0.5`` floor (an openpi sync pins 0.5.1,
        which has no ``rgb_encoder``), so when it is missing we let LeRobot default.
        """
        try:
            from lerobot.configs.video import RGBEncoderConfig
        except ImportError:
            return None
        chosen = codec.encoder()
        if chosen == "libx264":
            return None
        return RGBEncoderConfig(vcodec=chosen, crf=23)

    def _ensure_dataset(self, meta: EpisodeMeta) -> None:
        if self._ds is not None:
            return
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._fps = max(1, int(round(meta.control_hz)))
        encoder = self._rgb_encoder()
        self._ds = LeRobotDataset.create(
            repo_id=self._repo_id,
            fps=self._fps,
            features=self._build_features(meta),
            root=self._out,
            use_videos=True,
            **({"rgb_encoder": encoder} if encoder is not None else {}),
        )

    def add_episode(self, meta: EpisodeMeta, buffers: dict) -> None:
        self._ensure_dataset(meta)
        arms = meta.arm_names or ["left"]

        # Reference timeline = first camera role; align everything else to it.
        ref_role = meta.cameras[0].role if meta.cameras else None
        ref_ts = (
            buffers[cam_timestamp_key(ref_role)] if ref_role else None
        )
        n_ref = len(ref_ts) if ref_ts is not None else len(buffers[joint_pos_key(arms[0])])

        # Concat every arm's [joints, gripper] into one state/action vector.
        state_parts, action_parts = [], []
        for a in arms:
            state_parts += [buffers[joint_pos_key(a)], buffers[gripper_pos_key(a)]]
            action_parts += [buffers[action_joint_key(a)], buffers[action_gripper_key(a)]]
        state = np.concatenate(state_parts, axis=1).astype(np.float32)
        action = np.concatenate(action_parts, axis=1).astype(np.float32)

        # Per-camera frame index for each reference step (identity when synced).
        cam_idx: dict[str, np.ndarray] = {}
        for cam in meta.cameras:
            ts = buffers[cam_timestamp_key(cam.role)]
            if ref_ts is not None and len(ts) != n_ref:
                cam_idx[cam.role] = _nearest_indices(ref_ts, ts)
            else:
                cam_idx[cam.role] = np.arange(n_ref)

        for i in range(n_ref):
            frame: dict = {
                "observation.state": state[i],
                "action": action[i],
                "task": meta.task_name or "task",
            }
            for cam in meta.cameras:
                j = int(cam_idx[cam.role][i])
                for k in cam.image_keys:
                    frame[_lerobot_image_name(cam.role, k)] = buffers[
                        cam_image_key(cam.role, k)
                    ][j]
            self._ds.add_frame(frame)
        self._ds.save_episode()

    def finalize(self) -> None:
        # LeRobotDataset persists incrementally; nothing else required.
        pass


register("lerobot", LeRobotFormat)
