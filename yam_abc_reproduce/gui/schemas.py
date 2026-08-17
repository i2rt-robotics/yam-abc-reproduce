"""Request models for the GUI API."""

from __future__ import annotations

from pydantic import BaseModel


class StartRecording(BaseModel):
    task_name: str = "pick_and_place"


class ZeroGello(BaseModel):
    side: str = "left"  # "left" | "right" — which passive-GELLO leader to zero


class RobotForm(BaseModel):
    type: str = "yam_left"
    gripper: str = "linear_4310"
    channel: str | None = None  # blank/None = derive the CAN bus from `type`


class ControllerForm(BaseModel):
    type: str = "yam_lead_left"
    controls: str = "yam_left"
    channel: str | None = None  # blank/None = derive the CAN bus from `type`


class CameraForm(BaseModel):
    name: str
    type: str = "realsense"
    role: str | None = None
    serial: str | None = None


class StationForm(BaseModel):
    """Operator-editable Station rail settings, applied on Start Teleop."""

    controllers: list[ControllerForm] = []
    robots: list[RobotForm] = []
    cameras: list[CameraForm] = []
    data_format: str = "default"
    save_root: str = "data/episodes"
    task_name: str = ""


class CreateJob(BaseModel):
    kind: str  # "train" | "deploy"
    params: dict = {}


class DeployStart(BaseModel):
    """Start the in-session policy client against a server at host:port
    (127.0.0.1 for same-machine; the GPU box IP for a remote server)."""

    host: str = "127.0.0.1"
    port: int = 8000
    prompt: str = ""
    open_loop_horizon: int = 15               # non-RTC: rows executed per chunk before re-query (= ABC execute_chunk_dim)
    record: bool = True                       # log the rollout as an episode
    save_root: str = "data/rollouts/pi0"      # rollouts: data/rollouts/<policy>/<task>/<ep>
    home_pose: list[float] | None = None      # move here before the policy runs (demo start pose)
    rtc: bool = False                         # real-time chunking (abc: prefix-conditioned)
    # RTC knobs, mirroring ABC eval: P frozen-prefix rows, H rows streamed per
    # chunk (= execute_chunk_dim), re-query when L rows remain (= rtc_inference_lead_steps).
    rtc_prefix_length: int = 4                # P
    rtc_action_horizon: int = 15              # H
    rtc_lead_steps: int = 4                   # L
    # Safety: cap each arm joint to +-this speed in rad/s (truncate + warn), so an
    # OOD policy can't command a fast large jump. Speed-based so the limit is
    # control_hz-invariant. <=0 disables.
    max_joint_speed: float = 1.5
