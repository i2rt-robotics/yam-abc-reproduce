"""M2: default episode -> LeRobot dataset, reloaded and checked.

Skipped automatically where ``lerobot`` isn't installed (keeps core CI light);
run in an env with the ``lerobot`` extra to exercise it.
"""

import numpy as np
import pytest

from yam_abc_reproduce.data.formats import convert_episode
from yam_abc_reproduce.teleop.loop import ControlLoop

pytest.importorskip("lerobot")


def test_default_to_lerobot(mock_station, mock_devices, tmp_path):
    units, cameras = mock_devices
    loop = ControlLoop(units, cameras, control_hz=mock_station.control_hz)
    ep = loop.run_for(
        seconds=0.25,
        record_task="pick_and_place",
        save_root=mock_station.save_root,
        station=mock_station,
    )

    out = str(tmp_path / "lerobot_ds")
    convert_episode(ep, to="lerobot", repo_id="yam_abc_reproduce/test", out=out)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="yam_abc_reproduce/test", root=out)
    assert ds.num_episodes == 1
    assert ds.num_frames > 0
    # Both mono and stereo image features exist.
    feats = set(ds.features)
    assert "observation.images.top_rgb" in feats
    assert "observation.images.front_left" in feats
    assert "observation.images.front_right" in feats
    assert ds.features["observation.state"]["shape"][0] == 7
    item = ds[0]
    assert np.asarray(item["action"]).shape[0] == 7
