import pytest

from yam_abc_reproduce.camera.interface import CameraMode
from yam_abc_reproduce.camera.mock_camera import MockCamera
from yam_abc_reproduce.config import CameraConfig, RobotConfig, StationConfig
from yam_abc_reproduce.robot.mock_robot import MockRobot, MockTeleop
from yam_abc_reproduce.runtime import ArmUnit


@pytest.fixture
def mock_station(tmp_path):
    return StationConfig(
        robot=RobotConfig(type="mock", arm_name="left", num_arm_joints=6),
        cameras=[
            CameraConfig(name="top", type="mock", role="top", mode="mono"),
            CameraConfig(name="front", type="mock", role="front", mode="stereo", width=640),
        ],
        control_hz=200.0,
        save_root=str(tmp_path / "episodes"),
        task_name="pick_and_place",
        data_format="default",
    )


def make_units(names=("left",)):
    """Build one mock ArmUnit per name (each with its own MockRobot/MockTeleop)."""
    units = []
    for name in names:
        robot = MockRobot(num_arm_joints=6)
        units.append(ArmUnit(name, robot, MockTeleop(robot, num_arm_joints=6)))
    return units


@pytest.fixture
def mock_devices():
    """Single-arm mock: (units, cameras)."""
    cameras = [
        MockCamera(name="top", role="top", mode=CameraMode.MONO),
        MockCamera(name="front", role="front", mode=CameraMode.STEREO, width=640),
    ]
    return make_units(("left",)), cameras
