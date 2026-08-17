from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from launch_scripts.data_constants import (
    SO100_SO101_MOLMOACT2,
    YAM_BIMANUAL_MOLMOACT2,
)

TAG_METADATA_BY_TAG: Dict[str, Dict[str, object]] = {}
LEROBOT_TAG_PREFIX = "lerobot:"
DEFAULT_TAG_ACTION_HORIZON = 50
DEFAULT_TAG_N_ACTION_STEPS = 25
RawMixtureEntry = Tuple[str, List[object], float]
MixtureBuilder = Callable[[], Tuple[List[RawMixtureEntry], Dict[str, Dict[str, object]]]]


def reset_tag_metadata() -> None:
    TAG_METADATA_BY_TAG.clear()


def is_lerobot_tag(tag: object) -> bool:
    return isinstance(tag, str) and tag.startswith(LEROBOT_TAG_PREFIX)


def strip_lerobot_tag_prefix(tag: str) -> str:
    normalized = str(tag).strip()
    if normalized.startswith(LEROBOT_TAG_PREFIX):
        normalized = normalized[len(LEROBOT_TAG_PREFIX):].strip()
    if not normalized:
        raise ValueError("LeRobot tag names must be non-empty after removing prefix.")
    return normalized


def _make_tag_metadata(
    *,
    action_key: str,
    state_keys: List[str],
    camera_keys: Optional[List[str]] = None,
    camera_keys_alternative: Optional[List[str]] = None,
    normalize_gripper: bool,
    action_dim: int = 32,
    action_horizon: int = DEFAULT_TAG_ACTION_HORIZON,
    n_action_steps: int = DEFAULT_TAG_N_ACTION_STEPS,
    setup_type: str = "",
    control_mode: str = "",
) -> Dict[str, object]:
    if not isinstance(normalize_gripper, bool):
        raise TypeError("normalize_gripper must be a bool.")
    action_dim = int(action_dim)
    action_horizon = int(action_horizon)
    n_action_steps = int(n_action_steps)
    if action_dim < 1:
        raise ValueError("action_dim must be >= 1.")
    if action_horizon < 1:
        raise ValueError("action_horizon must be >= 1.")
    if n_action_steps < 1:
        raise ValueError("n_action_steps must be >= 1.")
    if n_action_steps > action_horizon:
        raise ValueError(
            f"n_action_steps ({n_action_steps}) cannot exceed action_horizon ({action_horizon})."
        )
    if not isinstance(state_keys, list) or not state_keys:
        raise TypeError("state_keys must be a non-empty list of strings.")
    normalized_state_keys = [str(v) for v in state_keys]
    if any(not key for key in normalized_state_keys):
        raise ValueError("state_keys must contain only non-empty strings.")
    if camera_keys is None:
        normalized_camera_keys: List[str] = []
    elif isinstance(camera_keys, list):
        normalized_camera_keys = [str(v) for v in camera_keys]
    else:
        raise TypeError("camera_keys must be a list of strings or None.")
    if camera_keys_alternative is None:
        normalized_camera_keys_alternative: Optional[List[str]] = None
    elif isinstance(camera_keys_alternative, list):
        normalized_camera_keys_alternative = [str(v) for v in camera_keys_alternative]
    else:
        raise TypeError("camera_keys_alternative must be a list of strings or None.")

    metadata = {
        "action_key": str(action_key),
        "state_keys": normalized_state_keys,
        "camera_keys": normalized_camera_keys,
        "normalize_gripper": normalize_gripper,
        "action_dim": action_dim,
        "action_horizon": action_horizon,
        "n_action_steps": n_action_steps,
        "setup_type": str(setup_type),
        "control_mode": str(control_mode),
    }
    if normalized_camera_keys_alternative is not None:
        metadata["camera_keys_alternative"] = normalized_camera_keys_alternative
    return metadata


def _with_lerobot_prefix(value: str) -> str:
    normalized = str(value).strip()
    if normalized.startswith(LEROBOT_TAG_PREFIX):
        return normalized
    return f"{LEROBOT_TAG_PREFIX}{normalized}"


def build_single_lerobot_mixture(
    *,
    name: str,
    tag: str,
    repo_ids: Sequence[str],
    action_key: str,
    state_keys: List[str],
    camera_keys: Optional[List[str]] = None,
    camera_keys_alternative: Optional[List[str]] = None,
    normalize_gripper: bool,
    action_horizon: int,
    n_action_steps: int,
    setup_type: str,
    control_mode: str,
    action_dim: int = 32,
    rate: float = 1.0,
) -> Tuple[List[RawMixtureEntry], Dict[str, Dict[str, object]]]:
    """Build a one-tag LeRobot mixture.

    Add most new LeRobot datasets by adding a small builder that calls this helper
    and then registering it in MOLMOACT2_LEROBOT_MIXTURES below.
    """
    mixture_tag = _with_lerobot_prefix(tag)
    repos = [_with_lerobot_prefix(repo_id) for repo_id in repo_ids]
    if not repos:
        raise ValueError(f"Mixture '{name}' must include at least one repo id.")
    data_mixture = [
        (
            mixture_tag,
            repos,
            float(rate),
        ),
    ]
    metadata_per_tag = {
        mixture_tag: _make_tag_metadata(
            action_key=action_key,
            state_keys=state_keys,
            camera_keys=camera_keys,
            camera_keys_alternative=camera_keys_alternative,
            normalize_gripper=normalize_gripper,
            action_dim=action_dim,
            action_horizon=action_horizon,
            n_action_steps=n_action_steps,
            setup_type=setup_type,
            control_mode=control_mode,
        ),
    }
    return data_mixture, metadata_per_tag


def build_molmoact2_pre_post_train() -> Tuple[List[RawMixtureEntry], Dict[str, Dict[str, object]]]:
    data_mixture = [
        (
            "lerobot:yam_dual_molmoact2",
            list(YAM_BIMANUAL_MOLMOACT2),
            0.3
        ),
        (
            "lerobot:so100_so101_molmoact2",
            list(SO100_SO101_MOLMOACT2),
            0.3
        ),
        (
            "lerobot:franka_molmoact",
            [
                "lerobot:allenai/molmoact_tabletop_lerobot",
                "lerobot:allenai/molmoact_household_lerobot",
            ],
            0.025
        ),
        (
            "lerobot:franka_droid",
            [
                "lerobot:allenai/droid_lerobot"
            ],
            0.3
        ),
        (
            "lerobot:google_robot_bc_z",
            [
                "lerobot:allenai/bc_z_lerobot",
            ],
            0.025
        ),
        (
            "lerobot:google_robot_fractal",
            [
                "lerobot:allenai/fractal_lerobot",
            ],
            0.025
        ),
        (
            "lerobot:widowx_bridge",
            [
                "lerobot:allenai/bridge_lerobot",
            ],
            0.025
        ),
    ]

    metadata_per_tag = {
        "lerobot:yam_dual_molmoact2": _make_tag_metadata(
            action_key="action",
            state_keys=["observation.state"],
            camera_keys=[
                "observation.images.top",
                "observation.images.left",
                "observation.images.right",
            ],
            normalize_gripper=False,
            setup_type="bimanual yam robotic arms in molmoact2",
            control_mode="absolute joint pose",
            action_horizon=30,
            n_action_steps=30,
        ),
        "lerobot:so100_so101_molmoact2": _make_tag_metadata(
            action_key="action",
            state_keys=["observation.state"],
            normalize_gripper=True,
            setup_type="single so100/so101 robotic arm in molmoact2",
            control_mode="absolute joint pose",
            action_horizon=30,
            n_action_steps=30,
        ),
        "lerobot:franka_molmoact": _make_tag_metadata(
            action_key="action.del_ee_action",
            state_keys=["observation.state"],
            camera_keys=[
                "observation.images.primary",
                "observation.images.secondary",
            ],
            normalize_gripper=False,
            setup_type="single franka robotic arm in molmoact2",
            control_mode="delta end-effector pose",
            action_horizon=10,
            n_action_steps=10,
        ),
        "lerobot:franka_droid": _make_tag_metadata(
            action_key="action",
            state_keys=["observation.state"],
            camera_keys=[
                "observation.images.exterior_1_left",
                "observation.images.exterior_2_left",
                "observation.images.wrist_left",
            ],
            normalize_gripper=False,
            setup_type="single franka robotic arm in droid",
            control_mode="absolute joint pose",
            action_horizon=15,
            n_action_steps=15,
        ),
        "lerobot:google_robot_bc_z": _make_tag_metadata(
            action_key="action",
            state_keys=["observation.state"],
            camera_keys=[
                "observation.images.image",
            ],
            normalize_gripper=False,
            setup_type="google robot in bc_z",
            control_mode="delta end-effector pose",
            action_horizon=10,
            n_action_steps=10,
        ),
        "lerobot:google_robot_fractal": _make_tag_metadata(
            action_key="action",
            state_keys=["observation.state"],
            camera_keys=[
                "observation.images.image",
            ],
            normalize_gripper=False,
            setup_type="google robot in rt_1",
            control_mode="delta end-effector pose",
            action_horizon=3,
            n_action_steps=3,
        ),
        "lerobot:widowx_bridge": _make_tag_metadata(
            action_key="action",
            state_keys=["observation.state"],
            camera_keys=[
                "observation.images.image_0",
                "observation.images.image_1",
                "observation.images.image_2",
                "observation.images.image_3",
            ],
            normalize_gripper=False,
            setup_type="single widowx robotic arm in bridge",
            control_mode="delta end-effector pose",
            action_horizon=5,
            n_action_steps=5,
        ),
    }

    return data_mixture, metadata_per_tag


def build_molmoact2_droid() -> Tuple[List[RawMixtureEntry], Dict[str, Dict[str, object]]]:
    return build_single_lerobot_mixture(
        name="droid",
        tag="franka_droid",
        repo_ids=["allenai/droid_lerobot"],
        action_key="action",
        state_keys=["observation.state"],
        camera_keys=[
            "observation.images.exterior_1_left",
            "observation.images.exterior_2_left",
            "observation.images.wrist_left",
        ],
        normalize_gripper=False,
        setup_type="single franka robotic arm in droid",
        control_mode="absolute joint pose",
        action_horizon=15,
        n_action_steps=15,
    )


def build_molmoact2_libero() -> Tuple[List[RawMixtureEntry], Dict[str, Dict[str, object]]]:
    return build_single_lerobot_mixture(
        name="libero",
        tag="libero",
        repo_ids=["allenai/MolmoAct2-LIBERO-Dataset"],
        action_key="action",
        state_keys=["observation.state"],
        camera_keys=[
            "observation.images.image",
            "observation.images.wrist_image",
        ],
        normalize_gripper=False,
        setup_type="single franka robotic arm in libero",
        control_mode="delta end-effector pose",
        action_horizon=10,
        n_action_steps=10,
    )


def build_molmoact2_libero_goal() -> Tuple[List[RawMixtureEntry], Dict[str, Dict[str, object]]]:
    return build_single_lerobot_mixture(
        name="libero_goal",
        tag="libero",
        repo_ids=["allenai/MolmoAct2-LIBERO-Dataset"],
        action_key="action",
        state_keys=["observation.state"],
        camera_keys=[
            "observation.images.image",
            "observation.images.wrist_image",
        ],
        normalize_gripper=False,
        setup_type="single franka robotic arm in libero",
        control_mode="delta end-effector pose",
        action_horizon=10,
        n_action_steps=10,
    )


def build_molmoact2_yam() -> Tuple[List[RawMixtureEntry], Dict[str, Dict[str, object]]]:
    return build_single_lerobot_mixture(
        name="yam",
        tag="yam_dual_molmoact2",
        repo_ids=YAM_BIMANUAL_MOLMOACT2,
        action_key="action",
        state_keys=["observation.state"],
        camera_keys=[
            "observation.images.top",
            "observation.images.left",
            "observation.images.right",
        ],
        normalize_gripper=False,
        setup_type="bimanual yam robotic arms in molmoact2",
        control_mode="absolute joint pose",
        action_horizon=30,
        n_action_steps=30,
    )


def build_molmoact2_so100_so101() -> Tuple[List[RawMixtureEntry], Dict[str, Dict[str, object]]]:
    return build_single_lerobot_mixture(
        name="so100_so101",
        tag="so100_so101_molmoact2",
        repo_ids=SO100_SO101_MOLMOACT2,
        action_key="action",
        state_keys=["observation.state"],
        normalize_gripper=True,
        setup_type="single so100/so101 robotic arm in molmoact2",
        control_mode="absolute joint pose",
        action_horizon=30,
        n_action_steps=30,
    )


MOLMOACT2_LEROBOT_MIXTURES: Dict[str, MixtureBuilder] = {
    "pre_post_train": build_molmoact2_pre_post_train,
    "droid": build_molmoact2_droid,
    "libero": build_molmoact2_libero,
    "libero_goal": build_molmoact2_libero_goal,
    "yam": build_molmoact2_yam,
    "so100_so101": build_molmoact2_so100_so101,
}


# YAM-ABC-Reproduce bimanual YAM: a single LeRobot dataset whose repo_id + action
# horizon come from the environment, so the GUI drives the dataset without editing
# this file. Reads YAM_ABC_* with a fallback to the legacy FREEMANI_* names so old
# station installs keep working. Camera keys carry the "_rgb" suffix to match the
# converter output.
def build_yam_abc_yam() -> "Tuple[List[RawMixtureEntry], Dict[str, Dict[str, object]]]":
    import os

    def _env(new, legacy, default):
        return os.environ.get(new) or os.environ.get(legacy) or default

    horizon = int(_env("YAM_ABC_ACTION_HORIZON", "FREEMANI_ACTION_HORIZON", "30"))
    return build_single_lerobot_mixture(
        name="yam_abc_yam",
        tag="yam_abc_yam",
        repo_ids=[_env("YAM_ABC_LEROBOT_REPO", "FREEMANI_LEROBOT_REPO", "put_the_bottle_into_the_bin")],
        action_key="action",
        state_keys=["observation.state"],
        camera_keys=[
            "observation.images.top_rgb",
            "observation.images.left_rgb",
            "observation.images.right_rgb",
        ],
        normalize_gripper=False,
        setup_type="bimanual yam robotic arms",
        control_mode="absolute joint pose",
        action_horizon=horizon,
        n_action_steps=horizon,
    )


MOLMOACT2_LEROBOT_MIXTURES["yam_abc_yam"] = build_yam_abc_yam
# Legacy alias so pre-rebrand station installs keep training without edits.
MOLMOACT2_LEROBOT_MIXTURES["freemani_yam"] = build_yam_abc_yam
