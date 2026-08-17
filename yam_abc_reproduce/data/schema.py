"""YAM-ABC-Reproduce data schema: key constants, key builders, and episode metadata.

Per-step state/action keys (single arm, default name "left"):
    <arm>-joint_pos        follower arm joint angles            (num_arm_joints,)
    <arm>-gripper_pos      follower gripper, normalized [0, 1]  (1,)
    action-<arm>-joint     commanded arm joint angles           (num_arm_joints,)
    action-<arm>-gripper   commanded gripper, normalized [0, 1] (1,)

Per-camera keys (role in {top, left, right, wrist}, img_key in {rgb, left, right}):
    <role>-images-<img_key>   video stream
    <role>-timestamp          per-frame capture timestamps (ms)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
WRITE_COMPLETE_FLAG = "write_complete.flag"


# --- key builders -----------------------------------------------------------
def joint_pos_key(arm: str = "left") -> str:
    return f"{arm}-joint_pos"


def gripper_pos_key(arm: str = "left") -> str:
    return f"{arm}-gripper_pos"


def action_joint_key(arm: str = "left") -> str:
    return f"action-{arm}-joint"


def action_gripper_key(arm: str = "left") -> str:
    return f"action-{arm}-gripper"


def cam_image_key(role: str, img_key: str) -> str:
    return f"{role}-images-{img_key}"


def cam_timestamp_key(role: str) -> str:
    return f"{role}-timestamp"


def arm_state_keys(arm: str = "left") -> list[str]:
    return [joint_pos_key(arm), gripper_pos_key(arm)]


def arm_action_keys(arm: str = "left") -> list[str]:
    return [action_joint_key(arm), action_gripper_key(arm)]


# --- metadata ---------------------------------------------------------------
@dataclass
class CameraMeta:
    name: str
    type: str
    role: str
    mode: str  # "mono" | "stereo"
    image_keys: list[str]
    width: int
    height: int
    fps: int
    depth_scale: float | None = None


@dataclass
class EpisodeMeta:
    yam_abc_reproduce_version: str
    schema_version: int
    created_at: str  # ISO timestamp, stamped by the recorder
    task_name: str
    arm_names: list[str]  # schema prefixes, e.g. ["left"] or ["left", "right"]
    num_arm_joints: int
    control_hz: float
    cameras: list[CameraMeta] = field(default_factory=list)
    num_frames: int = 0
    git_commit: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> EpisodeMeta:
        with open(path) as f:
            data = json.load(f)
        cams = [CameraMeta(**c) for c in data.pop("cameras", [])]
        # Back-compat: legacy single-arm episodes stored ``arm_name``.
        if "arm_name" in data and "arm_names" not in data:
            data["arm_names"] = [data.pop("arm_name")]
        data.pop("arm_name", None)
        known = {f.name for f in fields(cls)}
        return cls(cameras=cams, **{k: v for k, v in data.items() if k in known})
