"""Shared V4L2/OpenCV capture + serial-based device discovery.

decxin cameras are UVC devices, so both the mono and stereo drivers build on this.
Serial-based discovery keeps device selection stable across reboots / USB re-enumeration.
"""

from __future__ import annotations

import logging
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

        # Pin the V4L2 backend. Left to CAP_ANY, OpenCV takes whichever backend opens the
        # device first, and for a /dev/videoN *path* -- what find_device_by_serial returns --
        # that is regularly FFMPEG or GStreamer, whose CAP_PROP_* setters below are accepted
        # and then ignored. The camera streams its own default mode instead, so the width /
        # height / fps from cameras.yaml go missing with no error anywhere. Only the V4L2
        # backend actually applies them.
        self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"failed to open camera {name!r} (device={device}) on the V4L2 backend -- "
                f"check the node exists and is a UVC device (`v4l2-ctl -d {device} --all`)"
            )
        # FOURCC first, and not just for tidiness: the pixel format decides which
        # width/height/fps combinations the driver will offer at all. MJPG is what lets
        # these cameras do full resolution above ~5 fps; raw YUYV cannot.
        if fourcc:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        # V4L2 negotiates down to the nearest mode it supports rather than failing, so a
        # request the camera cannot meet is otherwise invisible. Read back what we actually
        # got. fps is advisory on some UVC drivers and reads back 0 -- don't cry wolf there.
        got_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        got_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        got_fps = int(round(self._cap.get(cv2.CAP_PROP_FPS)))
        if (got_w, got_h) != (width, height) or (got_fps and got_fps != fps):
            logging.warning(
                "camera %r: asked for %dx%d@%dfps, driver gave %dx%d@%sfps -- "
                "`v4l2-ctl -d %s --list-formats-ext` lists the modes it supports",
                name,
                width,
                height,
                fps,
                got_w,
                got_h,
                got_fps or "?",
                device,
            )

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
