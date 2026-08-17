"""decxin stereo USB camera: one side-by-side frame -> left/right eyes.

``split_side_by_side`` is a free function so the split logic is unit-testable
without any hardware.
"""

from __future__ import annotations

import numpy as np

from .base_v4l2 import V4L2Camera
from .interface import CameraFrame, CameraMode


def split_side_by_side(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a wide HxWx3 frame into (left, right), each H x (W//2) x 3."""
    w = frame.shape[1]
    half = w // 2
    left = np.ascontiguousarray(frame[:, :half])
    right = np.ascontiguousarray(frame[:, half : 2 * half])
    return left, right


class DecxinStereo(V4L2Camera):
    mode = CameraMode.STEREO

    def image_keys(self) -> list[str]:
        return ["left", "right"]

    def read(self) -> CameraFrame:
        rgb, ts = self._grab_rgb()
        left, right = split_side_by_side(rgb)
        return CameraFrame(images={"left": left, "right": right}, timestamp_ms=ts)
