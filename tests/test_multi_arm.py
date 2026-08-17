"""Multi-arm: both configured arms are driven and recorded (bimanual)."""

from pathlib import Path

import numpy as np

from yam_abc_reproduce.camera.interface import CameraMode
from yam_abc_reproduce.camera.mock_camera import MockCamera
from yam_abc_reproduce.config import RobotConfig, StationConfig
from yam_abc_reproduce.data.schema import EpisodeMeta
from yam_abc_reproduce.robot.mock_robot import MockRobot, MockTeleop
from yam_abc_reproduce.runtime import ArmUnit, build_arm_units
from yam_abc_reproduce.teleop.loop import ControlLoop


def test_build_arm_units_defaults_to_two_arms():
    cfg = StationConfig(robot=RobotConfig(type="mock", num_arm_joints=6))
    units = build_arm_units(cfg, mock=True)
    assert [u.name for u in units] == ["left", "right"]


def test_records_both_arms(tmp_path):
    station = StationConfig(
        robot=RobotConfig(type="mock", num_arm_joints=6),
        cameras=[],
        control_hz=200.0,
        save_root=str(tmp_path / "episodes"),
        task_name="pp",
    )
    units = []
    for name in ("left", "right"):
        r = MockRobot(num_arm_joints=6)
        units.append(ArmUnit(name, r, MockTeleop(r, num_arm_joints=6)))
    cameras = [MockCamera(name="top", role="top", mode=CameraMode.MONO)]

    loop = ControlLoop(units, cameras, control_hz=station.control_hz)
    path = Path(
        loop.run_for(
            seconds=0.2, record_task="pp", save_root=station.save_root, station=station
        )
    )

    meta = EpisodeMeta.from_json(path / "metadata.json")
    assert meta.arm_names == ["left", "right"]
    n = meta.num_frames
    assert n > 0

    # Both arms produce full state + action arrays, index-aligned to N.
    for arm in ("left", "right"):
        assert np.load(path / f"{arm}-joint_pos.npy").shape == (n, 6)
        assert np.load(path / f"{arm}-gripper_pos.npy").shape == (n, 1)
        assert np.load(path / f"action-{arm}-joint.npy").shape == (n, 6)
        assert np.load(path / f"action-{arm}-gripper.npy").shape == (n, 1)
