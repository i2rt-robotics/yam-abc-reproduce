"""Synthetic camera for tests, CI, and running the GUI without hardware."""

from __future__ import annotations

import time

import numpy as np

from .interface import CameraDriver, CameraFrame, CameraMode


class MockCamera(CameraDriver):
    def __init__(
        self,
        name: str = "top",
        role: str = "top",
        mode: CameraMode = CameraMode.MONO,
        width: int = 640,
        height: int = 480,
    ):
        self.name = name
        self.role = role
        self.mode = mode
        self._w = width
        self._h = height
        self._t = 0

    def image_keys(self) -> list[str]:
        return ["rgb"] if self.mode is CameraMode.MONO else ["left", "right"]

    def _frame(self, shift: int) -> np.ndarray:
        img = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        x = (self._t * 4 + shift) % self._w
        img[:, max(0, x - 10) : x + 10] = (0, 200, 255)  # a moving vertical bar
        return img

    def read(self) -> CameraFrame:
        if self.mode is CameraMode.MONO:
            images = {"rgb": self._frame(0)}
        else:
            images = {"left": self._frame(0), "right": self._frame(self._w // 4)}
        self._t += 1
        return CameraFrame(images=images, timestamp_ms=time.time() * 1000.0)

    def stop(self) -> None:
        pass
