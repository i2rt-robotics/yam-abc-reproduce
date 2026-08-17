"""M1: a full episode is recorded to disk from mock devices."""

from pathlib import Path

import numpy as np

from yam_abc_reproduce.data.schema import (
    WRITE_COMPLETE_FLAG,
    EpisodeMeta,
    cam_image_key,
    cam_timestamp_key,
)
from yam_abc_reproduce.teleop.loop import ControlLoop


def test_records_complete_episode(mock_station, mock_devices):
    units, cameras = mock_devices
    loop = ControlLoop(units, cameras, control_hz=mock_station.control_hz)
    path = loop.run_for(
        seconds=0.3,
        record_task="pick_and_place",
        save_root=mock_station.save_root,
        station=mock_station,
    )
    ep = Path(path)

    # Completeness marker present.
    assert (ep / WRITE_COMPLETE_FLAG).exists()

    meta = EpisodeMeta.from_json(ep / "metadata.json")
    n = meta.num_frames
    assert n > 0
    assert meta.task_name == "pick_and_place"

    # State/action arrays line up with the frame count.
    assert np.load(ep / "left-joint_pos.npy").shape == (n, 6)
    assert np.load(ep / "action-left-gripper.npy").shape == (n, 1)

    # Mono camera -> one mp4; stereo camera -> two mp4s; timestamps monotonic.
    assert (ep / f"{cam_image_key('top', 'rgb')}.mp4").exists()
    assert (ep / f"{cam_image_key('front', 'left')}.mp4").exists()
    assert (ep / f"{cam_image_key('front', 'right')}.mp4").exists()
    ts = np.load(ep / f"{cam_timestamp_key('top')}.npy")
    assert ts.shape == (n,)
    assert np.all(np.diff(ts) >= 0)
