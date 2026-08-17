"""Intel RealSense driver (RGB, optional depth)."""

from __future__ import annotations

import time

import numpy as np

from .interface import CameraFrame, CameraMode


class RealSense:
    mode = CameraMode.MONO

    def __init__(
        self,
        name: str,
        role: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial: str | None = None,
        enable_depth: bool = False,
    ):
        import pyrealsense2 as rs

        self.name = name
        self.role = role
        self._enable_depth = enable_depth

        self._pipe = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        if enable_depth:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self._pipe.start(cfg)
        self._depth_scale = (
            profile.get_device().first_depth_sensor().get_depth_scale()
            if enable_depth
            else None
        )

    def image_keys(self) -> list[str]:
        return ["rgb"]

    def read(self) -> CameraFrame:
        frames = self._pipe.wait_for_frames()
        color = frames.get_color_frame()
        rgb = np.asanyarray(color.get_data())
        depth = None
        if self._enable_depth:
            d = frames.get_depth_frame()
            if d:
                depth = np.asanyarray(d.get_data())
        return CameraFrame(
            images={"rgb": rgb},
            timestamp_ms=time.time() * 1000.0,
            depth=depth,
            depth_scale=self._depth_scale,
        )

    def stop(self) -> None:
        try:
            self._pipe.stop()
        except Exception:
            pass
