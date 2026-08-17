"""The single client<->server contract for unified policy deployment.

Every policy server (openpi, molmoact, abc) speaks the same observation and
action shape, so YAM-ABC-Reproduce ships ONE client. Embodiment-specific transforms
(camera resize/reorder, normalization, action-space conversion) live on the
SERVER side, inside each backend's adapter -- the client stays identical no
matter which policy it talks to.

Observation dict (client -> server), msgpack-numpy encoded:

    {
      "images": {<camera_role>: (H, W, 3) uint8 RGB, ...},   # e.g. top/left/right/wrist
      "state":  (S,) float32,   # per-arm [joint_0..joint_{n-1}, gripper] concatenated
      "prompt": str,            # natural-language instruction
    }

Response dict (server -> client):

    {"actions": (H, S) float32}   # H-step action chunk, same per-arm layout as `state`

The state/action layout mirrors ``yam_abc_reproduce.data.schema`` exactly: for each
driven arm (in ``units`` order -- i.e. the order ``build_arm_units`` returns,
which is the order the recorder wrote training data) we concatenate
``joint_pos`` (num_arm_joints) then ``gripper_pos`` (1, normalized [0, 1]). So
a two-arm YAM is 14-D:

    [left joints(6), left gripper(1), right joints(6), right gripper(1)]

Because the layout is derived from the same ``units`` list at record time and
deploy time, training data and live inference agree by construction -- there is
no hand-maintained index map to drift.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class _CameraLike(Protocol):
    """The subset of ``CameraDriver`` this module reads (name/role/keys)."""

    name: str
    role: str

    def image_keys(self) -> list[str]: ...


def arm_state(obs: dict[str, np.ndarray]) -> np.ndarray:
    """One arm's state slice ``[joint_0..joint_{n-1}, gripper]`` from a robot
    observation (``{"joint_pos": (n,), "gripper_pos": (g,)}``)."""
    joint = np.asarray(obs["joint_pos"], dtype=np.float32).reshape(-1)
    grip = np.asarray(obs["gripper_pos"], dtype=np.float32).reshape(-1)
    return np.concatenate([joint, grip])


def build_state(per_arm_obs: list[dict[str, np.ndarray]]) -> np.ndarray:
    """Full ``(S,)`` state vector, arms concatenated in ``units`` order."""
    return np.concatenate([arm_state(o) for o in per_arm_obs]).astype(np.float32)


def _primary_image(frame_images: dict[str, np.ndarray], image_keys: list[str]) -> np.ndarray:
    """Pick the RGB stream a mono/stereo camera contributes to the policy.

    Mono cameras expose ``{"rgb"}``; stereo cameras expose ``{"left","right"}``.
    Policies trained on the station consume one RGB view per role, so prefer
    ``rgb`` then ``left`` -- matching the convention the recorder/LeRobot
    converter use for the primary stream.
    """
    for k in ("rgb", "left", "right"):
        if k in frame_images:
            return np.asarray(frame_images[k])
    # Fall back to the camera's first declared key.
    k = image_keys[0]
    return np.asarray(frame_images[k])


def build_images(
    frames: dict[str, Any],
    cameras: list[_CameraLike],
) -> dict[str, np.ndarray]:
    """Map camera *role* -> one HxWx3 uint8 RGB image.

    ``frames`` is keyed by camera *name* (as produced by the control loop's
    camera workers); ``cameras`` provides the name->role mapping and declared
    image keys. Cameras with no frame this tick are skipped.
    """
    images: dict[str, np.ndarray] = {}
    for cam in cameras:
        fr = frames.get(cam.name)
        if fr is None:
            continue
        img = _primary_image(fr.images, cam.image_keys())
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        images[cam.role] = img
    return images


def build_observation(
    per_arm_obs: list[dict[str, np.ndarray]],
    frames: dict[str, Any],
    cameras: list[_CameraLike],
    prompt: str,
) -> dict[str, Any]:
    """Assemble the contract observation dict from live robot + camera reads."""
    return {
        "images": build_images(frames, cameras),
        "state": build_state(per_arm_obs),
        "prompt": prompt,
    }


def split_action(action_row: np.ndarray, arm_dims: list[int]) -> list[np.ndarray]:
    """Split one ``(S,)`` action row into per-arm command vectors.

    ``arm_dims[i]`` is arm ``i``'s total DOF incl. gripper (``robot.num_dofs()``).
    Each returned vector is ``[arm_joints..., gripper]`` ready for
    ``RobotInterface.command_joint_pos``.
    """
    row = np.asarray(action_row, dtype=np.float32).reshape(-1)
    total = int(sum(arm_dims))
    if row.shape[0] != total:
        raise ValueError(
            f"action row has {row.shape[0]} dims but arms need {total} "
            f"({arm_dims}); check that the served model matches this embodiment"
        )
    out: list[np.ndarray] = []
    i = 0
    for d in arm_dims:
        out.append(row[i : i + d].copy())
        i += d
    return out
