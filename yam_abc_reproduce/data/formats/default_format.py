"""The canonical YAM-ABC-Reproduce on-disk format (source of truth for all conversions).

Layout of one episode folder::

    <episode>/
      metadata.json
      <arm>-joint_pos.npy        (N, num_arm_joints)
      <arm>-gripper_pos.npy      (N, 1)
      action-<arm>-joint.npy     (N, num_arm_joints)
      action-<arm>-gripper.npy   (N, 1)
      <role>-images-<key>.mp4    one per camera image stream (stereo -> two mp4s)
      <role>-timestamp.npy       (N,) capture timestamps in ms
      write_complete.flag        written last, by the recorder

``buffers`` value types: image keys -> list[HxWx3 frames]; ``*-timestamp`` ->
(N,) float array; state/action keys -> (N, d) array.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..schema import EpisodeMeta
from ..video import read_mp4, write_mp4


def _is_image_key(key: str) -> bool:
    return "-images-" in key


def _fps_for_role(meta: EpisodeMeta, role: str) -> int:
    for c in meta.cameras:
        if c.role == role:
            return int(c.fps)
    return max(1, int(round(meta.control_hz)))


class DefaultFormat:
    # --- EpisodeWriter -----------------------------------------------------
    def write_episode(self, episode_dir, meta: EpisodeMeta, buffers: dict) -> None:
        episode_dir = Path(episode_dir)
        episode_dir.mkdir(parents=True, exist_ok=True)
        for key, val in buffers.items():
            if _is_image_key(key):
                role = key.split("-images-")[0]
                write_mp4(episode_dir / f"{key}.mp4", list(val), fps=_fps_for_role(meta, role))
            else:
                np.save(episode_dir / f"{key}.npy", np.asarray(val))
        meta.to_json(episode_dir / "metadata.json")

    # --- FormatReader ------------------------------------------------------
    def read_episode(self, episode_dir) -> tuple[EpisodeMeta, dict]:
        episode_dir = Path(episode_dir)
        meta = EpisodeMeta.from_json(episode_dir / "metadata.json")
        buffers: dict = {}
        for npy in sorted(episode_dir.glob("*.npy")):
            buffers[npy.stem] = np.load(npy)
        for mp4 in sorted(episode_dir.glob("*-images-*.mp4")):
            buffers[mp4.stem] = read_mp4(mp4)
        return meta, buffers
