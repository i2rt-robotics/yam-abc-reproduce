"""M3: stereo side-by-side split yields equal-width left/right eyes."""

import numpy as np

from yam_abc_reproduce.camera.decxin_stereo import split_side_by_side


def test_split_side_by_side():
    h, w = 48, 64  # wide frame
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, : w // 2] = 10  # left half marked
    frame[:, w // 2 :] = 200  # right half marked

    left, right = split_side_by_side(frame)
    assert left.shape == (h, w // 2, 3)
    assert right.shape == (h, w // 2, 3)
    assert int(left.mean()) == 10
    assert int(right.mean()) == 200


def test_split_odd_width():
    frame = np.zeros((10, 65, 3), dtype=np.uint8)
    left, right = split_side_by_side(frame)
    assert left.shape[1] == right.shape[1] == 32  # 65 // 2, dropping the odd column
