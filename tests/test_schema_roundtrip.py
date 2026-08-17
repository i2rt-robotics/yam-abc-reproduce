"""M1: default-format write -> read returns equivalent data."""

import numpy as np

from yam_abc_reproduce.data.formats import get_reader
from yam_abc_reproduce.data.schema import action_joint_key, cam_image_key, joint_pos_key
from yam_abc_reproduce.teleop.loop import ControlLoop


def test_default_roundtrip(mock_station, mock_devices):
    units, cameras = mock_devices
    loop = ControlLoop(units, cameras, control_hz=mock_station.control_hz)
    path = loop.run_for(
        seconds=0.25,
        record_task="t",
        save_root=mock_station.save_root,
        station=mock_station,
    )

    meta, buffers = get_reader("default").read_episode(path)
    n = meta.num_frames

    # State/action arrays survive the round trip.
    assert buffers[joint_pos_key("left")].shape == (n, 6)
    assert buffers[action_joint_key("left")].shape == (n, 6)

    # Decoded video has one frame per step, correct shape, for mono and stereo.
    top = buffers[cam_image_key("top", "rgb")]
    assert len(top) == n
    assert top[0].ndim == 3 and top[0].shape[2] == 3
    assert len(buffers[cam_image_key("front", "left")]) == n
    assert len(buffers[cam_image_key("front", "right")]) == n
    # Gripper action stays within the normalized range.
    g = buffers["action-left-gripper"]
    assert np.all((g >= 0.0) & (g <= 1.0))
