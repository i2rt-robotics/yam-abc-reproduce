"""Holds only the latest frame per camera and serves cheap JPEG previews.

The control loop pushes every frame here via ``update``; the hot loop never
pays encoding cost. ``preview_jpeg`` encodes on demand at the request rate.
"""

from __future__ import annotations

import threading

import numpy as np

from ..camera.interface import CameraFrame


class CameraHub:
    def __init__(self):
        self._latest: dict[str, CameraFrame] = {}
        self._lock = threading.Lock()

    def update(self, name: str, frame: CameraFrame) -> None:
        with self._lock:
            self._latest[name] = frame

    def names(self) -> list[str]:
        with self._lock:
            return list(self._latest)

    def preview_jpeg(self, name: str, eye: str | None = None, max_width: int = 320) -> bytes | None:
        import cv2

        with self._lock:
            frame = self._latest.get(name)
        if frame is None:
            return None

        images = frame.images
        if eye and eye in images:
            img = images[eye]
        elif "rgb" in images:
            img = images["rgb"]
        else:
            img = next(iter(images.values()))

        img = np.asarray(img)
        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / w
            img = cv2.resize(img, (max_width, int(h * scale)))
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr)
        return buf.tobytes() if ok else None
