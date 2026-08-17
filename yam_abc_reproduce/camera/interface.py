"""Camera abstraction that treats mono and stereo uniformly.

The single ``images`` dict is the key design move: a mono camera yields
``{"rgb": ...}`` and a stereo camera yields ``{"left": ..., "right": ...}``.
The recorder and every format writer iterate ``frame.images.items()`` and never
special-case stereo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np


class CameraMode(Enum):
    MONO = "mono"
    STEREO = "stereo"


@dataclass
class CameraFrame:
    images: dict[str, np.ndarray]  # mono: {"rgb"}; stereo: {"left","right"}
    timestamp_ms: float
    depth: np.ndarray | None = None
    depth_scale: float | None = None
    meta: dict = field(default_factory=dict)


@runtime_checkable
class CameraDriver(Protocol):
    name: str
    role: str  # "top" | "left" | "right" | "wrist"
    mode: CameraMode

    def read(self) -> CameraFrame:
        ...

    def image_keys(self) -> list[str]:
        """Keys present in ``CameraFrame.images``; drives schema + LeRobot features."""
        ...

    def stop(self) -> None:
        ...
