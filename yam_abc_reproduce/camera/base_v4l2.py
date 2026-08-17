"""Shared V4L2/OpenCV capture + serial-based device discovery.

decxin cameras are UVC devices, so both the mono and stereo drivers build on this.
Serial-based discovery keeps device selection stable across reboots / USB re-enumeration.
"""

from __future__ import annotations

import time

import numpy as np


def find_device_by_serial(serial: str) -> str:
    """Resolve a /dev/videoN path from a USB serial (ID_SERIAL_SHORT) via pyudev."""
    import pyudev

    ctx = pyudev.Context()
    for dev in ctx.list_devices(subsystem="video4linux"):
        if dev.get("ID_SERIAL_SHORT") == serial or dev.get("ID_SERIAL") == serial:
            node = dev.device_node
            if node:
                return node
    raise RuntimeError(f"no v4l2 device with serial {serial!r}")


class V4L2Camera:
    """OpenCV ``VideoCapture`` wrapper. Subclasses set ``mode`` + ``read``/``image_keys``."""

    def __init__(
        self,
        name: str,
        role: str,
        width: int,
        height: int,
        fps: int,
        device: int | str | None = None,
        serial: str | None = None,
        fourcc: str = "MJPG",
    ):
        import cv2

        self.name = name
        self.role = role
        self._w = width
        self._h = height
        self._fps = fps

        if device is None and serial:
            device = find_device_by_serial(serial)
        if device is None:
            device = 0

        self._cap = cv2.VideoCapture(device)
        if fourcc:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"failed to open camera {name!r} (device={device})")

    def _grab_rgb(self) -> tuple[np.ndarray, float]:
        import cv2

        ok, bgr = self._cap.read()
        ts = time.time() * 1000.0
        if not ok:
            raise RuntimeError(f"camera {self.name!r} read failed")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), ts

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
