"""decxin monocular USB camera."""

from __future__ import annotations

from .base_v4l2 import V4L2Camera
from .interface import CameraFrame, CameraMode


class DecxinMono(V4L2Camera):
    mode = CameraMode.MONO

    def image_keys(self) -> list[str]:
        return ["rgb"]

    def read(self) -> CameraFrame:
        rgb, ts = self._grab_rgb()
        return CameraFrame(images={"rgb": rgb}, timestamp_ms=ts)
