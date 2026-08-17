"""Deploy-side image resize for the ABC policy server.

ABC's training cache is built by ``export_mcap.py`` (in the abc repo), which
resizes every camera frame to 224x224 with ffmpeg::

    scale=224:224:force_original_aspect_ratio=decrease:flags=bicubic,
    pad=224:224:(ow-iw)/2:(oh-ih)/2

i.e. a bicubic, aspect-preserving downscale + centered zero-pad.
"""

from __future__ import annotations

import numpy as np


def letterbox_224(img_hwc, target_h: int = 224, target_w: int = 224) -> np.ndarray:
    """Aspect-preserving bicubic downscale + centered zero-pad to (target_h, target_w) uint8.

    Geometry matches ffmpeg ``force_original_aspect_ratio=decrease`` + centered
    ``pad`` (and abc_minimal.resize_with_pad): ratio = max(w/W, h/H), floor
    sizing, centered zero pad. Interpolation is bicubic to match ffmpeg
    ``flags=bicubic``. Idempotent on an image already at the target size.
    """
    import cv2

    a = np.asarray(img_hwc)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    h, w = a.shape[:2]
    if (h, w) == (target_h, target_w):
        return a
    ratio = max(w / target_w, h / target_h)
    new_h = max(1, int(h / ratio)) 
    new_w = max(1, int(w / ratio))
    resized = cv2.resize(a, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    pad_h0 = (target_h - new_h) // 2
    pad_h1 = target_h - new_h - pad_h0
    pad_w0 = (target_w - new_w) // 2
    pad_w1 = target_w - new_w - pad_w0
    out = cv2.copyMakeBorder(
        resized, pad_h0, pad_h1, pad_w0, pad_w1,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )
    return out.astype(np.uint8)
