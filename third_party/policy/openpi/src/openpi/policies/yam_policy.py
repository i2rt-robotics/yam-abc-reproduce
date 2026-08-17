"""Policy transforms for YAM LeRobot datasets.

YAM episodes are converted to LeRobot v3.0 with:
  observation.state              (14,) = [L: 6 joints + 1 gripper, R: 6 joints + 1 gripper]
  action                         (14,) same layout (leader-derived command)
  observation.images.top_rgb     third-person (overhead)
  observation.images.left_rgb    left view   -> left wrist slot
  observation.images.right_rgb   right view  -> right wrist slot
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

YAM_ACTION_DIM = 14  # 2 arms x (6 joints + 1 gripper)


def make_yam_example() -> dict:
    """Random input example (matches the repacked key names)."""
    return {
        "observation/state": np.random.rand(YAM_ACTION_DIM),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # LeRobot stores video as (C, H, W)
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    """Convert repacked YAM inputs into the model's expected format (train + inference)."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base = _parse_image(data["observation/image"])
        left = _parse_image(data["observation/left_wrist"])
        right = _parse_image(data["observation/right_wrist"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": left,
                "right_wrist_0_rgb": right,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:  # only present during training
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    """Strip model action padding back to YAM's real dims (inference only).

    ``action_dim`` is 7 per arm (6 joints + 1 gripper): 14 bimanual, 7 single-arm."""

    action_dim: int = YAM_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
