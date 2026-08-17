#!/usr/bin/env python3
"""Plot full-episode open-loop predictions for a LeRobot or local MolmoAct2 policy checkpoint.

Pipeline:
1. Load a LeRobot dataset episode.
2. Load either:
   - a local LeRobot policy checkpoint (for example `diffusion`) plus its saved
     pre/post processors, or
   - a MolmoAct2 checkpoint through the local LeRobot MolmoAct2 wrapper.
3. Roll through the episode using the policy's own inference path.
4. Save both the policy-scale action trajectory and the raw environment-scale action
   trajectory, and plot both for each action dimension.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import draccus

# LeRobot rejects the deprecated LEROBOT_HOME env var during import. Preserve
# compatibility with older shells by translating it before importing lerobot modules.
if "HF_LEROBOT_HOME" not in os.environ and "LEROBOT_DATA_ROOT" in os.environ:
    os.environ["HF_LEROBOT_HOME"] = os.environ["LEROBOT_DATA_ROOT"]
if "LEROBOT_HOME" in os.environ:
    os.environ.setdefault("HF_LEROBOT_HOME", os.environ["LEROBOT_HOME"])
    del os.environ["LEROBOT_HOME"]

REPO_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
for candidate in (REPO_ROOT, LEROBOT_SRC):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

# Import the policy factory module to register all policy config subclasses.
from lerobot.policies import factory as _policy_factory_registry  # noqa: F401,E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.configs.types import FeatureType  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config  # noqa: E402
from lerobot.processor.normalize_processor import (  # noqa: E402
    NormalizerProcessorStep,
)
from lerobot.utils.constants import ACTION  # noqa: E402
from olmo.extra_tokens import style_uses_depth_output  # noqa: E402

log = logging.getLogger(__name__)


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="LeRobot dataset repo id, with or without 'lerobot:' prefix.")
    parser.add_argument("--episode_idx", "--episode_id", dest="episode_idx", type=int, required=True)
    parser.add_argument(
        "--episode_frames",
        default=None,
        help="Optional episode-relative frame slice as 'start,end' with end exclusive, e.g. '0,100' -> frames 0..99.",
    )
    parser.add_argument("--checkpoint", required=True, help="Local LeRobot checkpoint directory containing config.json and model.safetensors.")
    parser.add_argument("--output_dir", required=True, help="Directory to save plots and arrays.")
    parser.add_argument("--policy_type", default=None, help="Optional expected policy type, e.g. diffusion.")
    parser.add_argument(
        "--hf_ckpt",
        dest="hf_ckpt",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help=(
            "Whether --checkpoint points to a converted HuggingFace MolmoAct2 checkpoint. "
            "Accepts true/false; default false."
        ),
    )
    parser.add_argument("--dataset_root", default=None, help="Optional local LeRobot cache root or dataset directory.")
    parser.add_argument("--device", default="cuda", help="Torch device for the policy.")
    parser.add_argument("--video_backend", default="pyav", help="LeRobot video backend. Use pyav to avoid torchcodec issues.")
    parser.add_argument(
        "--inference_action_mode",
        default=None,
        choices=("continuous", "discrete"),
        help="MolmoAct2 inference action mode. Defaults to continuous.",
    )
    parser.add_argument(
        "--discrete_action_tokenizer",
        default=None,
        help="Discrete action tokenizer override. Required for discrete MolmoAct2 inference.",
    )
    parser.add_argument(
        "--enable_depth_reasoning",
        action="store_true",
        help="Enable depth-token reasoning; inference uses robot_depth_action instead of robot_action.",
    )
    parser.add_argument("--norm_tag", default="", help="MolmoAct2 normalization tag.")
    parser.add_argument("--n_action_steps", type=int, default=None, help="Optional MolmoAct2 executed-step override.")
    parser.add_argument("--num_steps", type=int, default=None, help="MolmoAct2 flow-matching step override.")
    parser.add_argument("--seq_len", type=int, default=None, help="MolmoAct2 tokenizer sequence length override.")
    parser.add_argument(
        "--enable_inference_cuda_graph",
        dest="enable_inference_cuda_graph",
        action="store_true",
        default=True,
        help="Enable HF action-expert CUDA graph capture/replay when supported.",
    )
    parser.add_argument(
        "--disable_inference_cuda_graph",
        dest="enable_inference_cuda_graph",
        action="store_false",
        help="Disable HF action-expert CUDA graph capture/replay.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for deterministic open-loop evaluation.")
    parser.add_argument("--verbose", action="store_true", help="Print checkpoint, feature, and rollout information.")
    return parser.parse_args()


def _resolve_inference_action_mode(args: argparse.Namespace) -> str:
    return str(args.inference_action_mode or "continuous").strip().lower()


def _requires_discrete_action_tokenizer(
    inference_action_mode: str,
    discrete_action_tokenizer: Optional[str],
) -> Optional[str]:
    normalized = str(discrete_action_tokenizer or "").strip() or None
    if str(inference_action_mode).strip().lower() == "discrete" and normalized is None:
        raise ValueError(
            "--discrete_action_tokenizer is required when "
            "--inference_action_mode is set to 'discrete'."
        )
    return normalized


def _use_hf_checkpoint(args: argparse.Namespace) -> bool:
    return bool(args.hf_ckpt)


def _use_molmoact2_policy(args: argparse.Namespace) -> bool:
    policy_type = str(args.policy_type or "").strip().lower().replace("-", "_")
    return bool(args.hf_ckpt) or policy_type == "molmoact2"


def _is_molmoact2_policy_type(policy_cfg: PreTrainedConfig) -> bool:
    return str(policy_cfg.type) == "molmoact2"


def _normalize_dataset_name(dataset_name: str) -> str:
    return dataset_name[len("lerobot:") :] if dataset_name.startswith("lerobot:") else dataset_name


def _resolve_dataset_root(repo_id: str, dataset_root: Optional[str]) -> Optional[Path]:
    if dataset_root is None:
        return None
    root = Path(dataset_root).expanduser()
    repo_root = root / repo_id
    if (repo_root / "meta").exists() or (repo_root / "data").exists():
        return repo_root
    if (root / "meta").exists() or (root / "data").exists():
        return root
    return root


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_float_vector(value: Any) -> np.ndarray:
    array = _to_numpy(value).astype(np.float32)
    if array.ndim == 0:
        array = array.reshape(1)
    return array.reshape(-1)


def _resolve_molmoact2_style(policy_cfg: PreTrainedConfig, policy: torch.nn.Module) -> Optional[str]:
    policy_type = str(policy_cfg.type)
    if policy_type != "molmoact2":
        return None
    hf_backend = getattr(policy, "_hf_backend", None)
    if hf_backend is not None:
        resolve_depth_flags = getattr(hf_backend, "_resolve_depth_flags", None)
        if callable(resolve_depth_flags):
            enable_depth_reasoning, _ = resolve_depth_flags()
        else:
            enable_depth_reasoning = bool(getattr(policy_cfg, "enable_depth_reasoning", False))
        return "robot_depth_action" if enable_depth_reasoning else "robot_action"
    handles = getattr(policy, "_handles", None)
    style = str(getattr(handles, "default_style", "") or "")
    return style or None


def _resolve_policy_inference_path(policy_cfg: PreTrainedConfig, policy: torch.nn.Module) -> str:
    if not _is_molmoact2_policy_type(policy_cfg):
        return "select_action_queue"

    inference_action_mode = str(getattr(policy_cfg, "inference_action_mode", "continuous") or "continuous").lower()
    resolved_style = _resolve_molmoact2_style(policy_cfg, policy) or ""
    if inference_action_mode != "continuous" or style_uses_depth_output(resolved_style):
        return "per_frame_chunk_regen"
    return "select_action_queue"


def _resolve_molmoact2_n_obs_steps(policy_cfg: PreTrainedConfig, policy: torch.nn.Module) -> int:
    if str(policy_cfg.type) == "molmoact2":
        handles = getattr(policy, "_handles", None)
        if handles is not None:
            return int(handles.n_obs_steps)
    return int(getattr(policy_cfg, "n_obs_steps", 1) or 1)


def _extract_action_vector(action_value: Any) -> np.ndarray:
    if hasattr(action_value, "detach"):
        action_value = action_value.detach()
    action_array = _to_numpy(action_value)
    if action_array.ndim == 3:
        action_array = action_array[0, 0]
    elif action_array.ndim == 2:
        action_array = action_array[0]
    return _to_float_vector(action_array)


def _episode_range(dataset: LeRobotDataset, episode_idx: int) -> tuple[int, int]:
    if int(episode_idx) < 0 or int(episode_idx) >= len(dataset.meta.episodes):
        raise ValueError(f"episode_idx={episode_idx} is out of range for dataset with {len(dataset.meta.episodes)} episodes.")
    episode_meta = dataset.meta.episodes[int(episode_idx)]
    return int(episode_meta["dataset_from_index"]), int(episode_meta["dataset_to_index"])


def _parse_episode_frames(episode_frames: Optional[str]) -> Optional[tuple[int, int]]:
    if episode_frames is None:
        return None

    raw_value = str(episode_frames).strip()
    if not raw_value:
        return None

    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 2 or any(part == "" for part in parts):
        raise ValueError(
            f"Invalid --episode_frames value '{episode_frames}'. Expected 'start,end' with end exclusive, e.g. '0,100'."
        )

    try:
        start_frame, end_frame = (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise ValueError(
            f"Invalid --episode_frames value '{episode_frames}'. Start/end must be integers."
        ) from exc

    if start_frame < 0 or end_frame < 0:
        raise ValueError(f"Invalid --episode_frames value '{episode_frames}'. Start/end must be non-negative.")
    if end_frame <= start_frame:
        raise ValueError(
            f"Invalid --episode_frames value '{episode_frames}'. Expected end > start because the end is exclusive."
        )

    return start_frame, end_frame


def _load_policy_config(checkpoint: str, device: str) -> PreTrainedConfig:
    checkpoint_path = Path(checkpoint).expanduser()
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing LeRobot config.json at {config_path}")

    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    policy_type = raw_config.get("type")
    if not isinstance(policy_type, str) or not policy_type:
        raise ValueError(f"Checkpoint config is missing a valid 'type': {config_path}")

    try:
        cfg = PreTrainedConfig.from_pretrained(str(checkpoint_path))
    except Exception as exc:
        config_cls = PreTrainedConfig.get_choice_class(policy_type)
        allowed_keys = {field.name for field in dataclasses.fields(config_cls)}
        cleaned_config = {key: value for key, value in raw_config.items() if key in allowed_keys}
        dropped_keys = sorted(set(raw_config) - allowed_keys - {"type"})
        if dropped_keys:
            log.warning(
                "Ignoring legacy/unknown config keys for %s checkpoint %s: %s",
                policy_type,
                checkpoint_path,
                dropped_keys,
            )

        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp_file:
            json.dump(cleaned_config, tmp_file)
            tmp_path = tmp_file.name
        try:
            with draccus.config_type("json"):
                cfg = draccus.parse(config_cls, tmp_path, args=[])
        except Exception:
            raise exc
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    cfg.device = str(device)
    cfg.pretrained_path = checkpoint_path
    return cfg


def _load_policy_and_processors(
    checkpoint: str,
    dataset: LeRobotDataset,
    *,
    device: str,
    args: argparse.Namespace,
) -> tuple[PreTrainedConfig, torch.nn.Module, Any, Any]:
    inference_action_mode = _resolve_inference_action_mode(args)
    discrete_action_tokenizer = _requires_discrete_action_tokenizer(
        inference_action_mode,
        args.discrete_action_tokenizer,
    )
    use_hf_ckpt = _use_hf_checkpoint(args)
    if _use_molmoact2_policy(args):
        norm_tag = str(args.norm_tag or "").strip()
        if not norm_tag:
            raise ValueError("MolmoAct2 open-loop inference requires --norm_tag.")
        cfg = MolmoAct2Config(
            checkpoint_path=str(Path(checkpoint).expanduser()),
            device=str(device),
            seq_len=args.seq_len,
            num_steps=args.num_steps,
            inference_action_mode=inference_action_mode,
            discrete_action_tokenizer=discrete_action_tokenizer,
            enable_depth_reasoning=bool(args.enable_depth_reasoning),
            verbose=bool(args.verbose),
            norm_tag=norm_tag,
            enable_inference_cuda_graph=bool(args.enable_inference_cuda_graph),
        )
        policy = make_policy(cfg, ds_meta=dataset.meta)
        preprocessor, postprocessor = make_pre_post_processors(cfg)
        return cfg, policy, preprocessor, postprocessor

    cfg = _load_policy_config(checkpoint, device)
    policy = make_policy(cfg, ds_meta=dataset.meta)
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            cfg,
            pretrained_path=checkpoint,
        )
    except FileNotFoundError:
        preprocessor, postprocessor = make_pre_post_processors(
            cfg,
            dataset_stats=dataset.meta.stats,
        )
    return cfg, policy, preprocessor, postprocessor


def _extract_input_observation(item: Dict[str, Any], input_keys: Sequence[str]) -> Dict[str, Any]:
    observation: Dict[str, Any] = {}
    missing_keys: List[str] = []
    for key in input_keys:
        if key not in item:
            missing_keys.append(str(key))
            continue
        observation[str(key)] = item[key]
    if missing_keys:
        raise KeyError(f"Dataset item is missing required policy input keys: {missing_keys}")
    if "task" in item:
        observation["task"] = item["task"]
    return observation


def _find_normalizer_step(preprocessor: Any) -> Optional[NormalizerProcessorStep]:
    steps = getattr(preprocessor, "steps", [])
    for step in steps:
        if isinstance(step, NormalizerProcessorStep):
            return step
    return None


def _normalize_gt_action(normalizer_step: NormalizerProcessorStep, action_raw: np.ndarray) -> np.ndarray:
    normalized = normalizer_step._normalize_action(  # noqa: SLF001 - reuse LeRobot's saved normalizer directly
        torch.as_tensor(action_raw, dtype=torch.float32, device=normalizer_step.device),
        inverse=False,
    )
    return _to_float_vector(normalized)


def _normalize_action_with_molmoact2_policy(
    policy: torch.nn.Module,
    action: Any,
    *,
    repo_id: str,
) -> np.ndarray:
    handles = getattr(policy, "_handles", None)
    if handles is None or handles.robot_processor is None:
        raise RuntimeError("MolmoAct2 policy does not expose robot normalization metadata.")
    normalized = handles.robot_processor.normalize_action(action, repo_id=repo_id)
    return _to_float_vector(normalized)


def _get_hf_robot_stats(policy: torch.nn.Module) -> Any:
    hf_backend = getattr(policy, "_hf_backend", None)
    model = getattr(hf_backend, "model", None) if hf_backend is not None else getattr(policy, "model", None)
    get_robot_stats = getattr(model, "_get_robot_stats", None)
    if not callable(get_robot_stats):
        raise RuntimeError("MolmoAct2 HF backend does not expose robot normalization metadata.")
    return get_robot_stats()


def _normalize_action_with_hf_policy(
    policy: torch.nn.Module,
    action: Any,
    *,
    repo_id: str,
) -> np.ndarray:
    stats = _get_hf_robot_stats(policy)
    resolved_tag = stats.validate_tag(repo_id)
    normalizer = getattr(stats, "action_normalizers", {}).get(resolved_tag)
    normalized = action if normalizer is None else normalizer.normalize(action)
    return _to_float_vector(normalized)


def _normalize_action_with_policy(
    policy_cfg: PreTrainedConfig,
    policy: torch.nn.Module,
    action: Any,
    *,
    repo_id: str,
) -> np.ndarray:
    if getattr(policy, "_hf_backend", None) is not None:
        return _normalize_action_with_hf_policy(policy, action, repo_id=repo_id)
    return _normalize_action_with_molmoact2_policy(policy, action, repo_id=repo_id)


def _postprocess_policy_action(postprocessor: Any, action: Any) -> np.ndarray:
    if postprocessor is None:
        return _to_float_vector(action)

    action_value = action
    if not torch.is_tensor(action_value):
        postprocessor_device = None
        for step in getattr(postprocessor, "steps", []):
            device = getattr(step, "device", None)
            if device is not None:
                postprocessor_device = device
                break
        action_value = torch.as_tensor(action_value, dtype=torch.float32, device=postprocessor_device)

    processed = postprocessor(action_value)
    return _to_float_vector(processed)


def _resolve_action_policy_scale(
    policy_cfg: PreTrainedConfig,
    policy: torch.nn.Module,
    normalizer_step: Optional[NormalizerProcessorStep],
    *,
    norm_tag: Optional[str] = None,
) -> Optional[str]:
    if str(policy_cfg.type) == "molmoact2" and getattr(policy, "_hf_backend", None) is None:
        handles = getattr(policy, "_handles", None)
        robot_processor = getattr(handles, "robot_processor", None)
        if robot_processor is None:
            return None
        resolved_tag = robot_processor.resolve_tag(norm_tag or getattr(handles, "norm_tag", ""))
        if resolved_tag is None:
            return None
        normalizer = robot_processor.action_normalizers.get(resolved_tag)
        return None if normalizer is None else str(getattr(normalizer, "mode", "") or None)

    if str(policy_cfg.type) == "molmoact2" and getattr(policy, "_hf_backend", None) is not None:
        stats = _get_hf_robot_stats(policy)
        requested_tag = str(norm_tag or getattr(policy_cfg, "norm_tag", "") or "").strip()
        if not requested_tag:
            return None
        resolved_tag = stats.validate_tag(requested_tag)
        normalizer = getattr(stats, "action_normalizers", {}).get(resolved_tag)
        if normalizer is None:
            return None
        return str(getattr(normalizer, "mode", "") or getattr(stats, "norm_mode", "") or None)

    if normalizer_step is None:
        return None
    norm_mode = getattr(normalizer_step, "norm_map", {}).get(FeatureType.ACTION)
    if norm_mode is None:
        return None
    return str(getattr(norm_mode, "value", norm_mode))


def _coerce_action_mask(mask_value: Any, action_dim: int) -> Optional[List[bool]]:
    if mask_value is None:
        return [True] * action_dim

    mask_array = _to_numpy(mask_value).astype(bool).reshape(-1)
    if mask_array.shape[0] != action_dim:
        log.warning(
            "Ignoring action normalization mask with dim %d because action_dim=%d.",
            int(mask_array.shape[0]),
            int(action_dim),
        )
        return None
    return [bool(v) for v in mask_array.tolist()]


def _resolve_action_normalization_mask(
    policy_cfg: PreTrainedConfig,
    policy: torch.nn.Module,
    normalizer_step: Optional[NormalizerProcessorStep],
    action_dim: int,
    *,
    norm_tag: Optional[str] = None,
) -> Optional[List[bool]]:
    if action_dim < 1:
        return None

    if str(policy_cfg.type) == "molmoact2" and getattr(policy, "_hf_backend", None) is None:
        handles = getattr(policy, "_handles", None)
        robot_processor = getattr(handles, "robot_processor", None)
        if robot_processor is None:
            return [True] * action_dim
        resolved_tag = robot_processor.resolve_tag(norm_tag or getattr(handles, "norm_tag", ""))
        if resolved_tag is None:
            return [True] * action_dim
        normalizer = robot_processor.action_normalizers.get(resolved_tag)
        if normalizer is None:
            return [True] * action_dim
        return _coerce_action_mask(getattr(normalizer, "mask", None), action_dim)

    if str(policy_cfg.type) == "molmoact2" and getattr(policy, "_hf_backend", None) is not None:
        stats = _get_hf_robot_stats(policy)
        requested_tag = str(norm_tag or getattr(policy_cfg, "norm_tag", "") or "").strip()
        if not requested_tag:
            return [True] * action_dim
        resolved_tag = stats.validate_tag(requested_tag)
        normalizer = getattr(stats, "action_normalizers", {}).get(resolved_tag)
        if normalizer is None:
            return [True] * action_dim
        return _coerce_action_mask(getattr(normalizer, "mask", None), action_dim)

    if normalizer_step is None:
        return [True] * action_dim
    tensor_stats = getattr(normalizer_step, "_tensor_stats", {})
    action_stats = tensor_stats.get(ACTION, {}) if isinstance(tensor_stats, dict) else {}
    return _coerce_action_mask(action_stats.get("mask"), action_dim)


def _gripper_action_dims(action_names: Sequence[str]) -> List[int]:
    return [idx for idx, name in enumerate(action_names) if "gripper" in str(name).lower()]


def _masked_action_dims(normalization_mask: Optional[Sequence[bool]]) -> List[int]:
    if normalization_mask is None:
        return []
    return [idx for idx, is_normalized in enumerate(normalization_mask) if not bool(is_normalized)]


def _compute_average_mse_for_dims(
    predicted: Optional[np.ndarray],
    gt_reference: Optional[np.ndarray],
    dims: Sequence[int],
) -> Optional[float]:
    if predicted is None or gt_reference is None or not dims:
        return None
    dim_indices = [int(idx) for idx in dims]
    mean_mse, _ = _compute_average_mse(predicted[:, dim_indices], gt_reference[:, dim_indices])
    return mean_mse


def _episode_absolute_indices(
    dataset: LeRobotDataset,
    episode_idx: int,
    episode_frames: Optional[tuple[int, int]] = None,
) -> List[int]:
    start_idx, end_idx = _episode_range(dataset, episode_idx)
    indices = list(range(start_idx, end_idx))
    if not indices:
        raise ValueError(f"Episode {episode_idx} contains no frames.")

    if episode_frames is not None:
        frame_start, frame_end = episode_frames
        episode_length = len(indices)
        if frame_start >= episode_length:
            raise ValueError(
                f"--episode_frames start {frame_start} is out of range for episode {episode_idx} with {episode_length} frames."
            )
        if frame_end > episode_length:
            raise ValueError(
                f"--episode_frames end {frame_end} exceeds episode {episode_idx} length {episode_length}. "
                "The end is exclusive."
            )
        indices = indices[frame_start:frame_end]
        if not indices:
            raise ValueError(
                f"--episode_frames {frame_start},{frame_end} selected no frames for episode {episode_idx}."
            )

    return indices


def _rollout_episode(
    dataset: LeRobotDataset,
    policy_cfg: PreTrainedConfig,
    policy: torch.nn.Module,
    preprocessor: Any,
    postprocessor: Any,
    normalizer_step: Optional[NormalizerProcessorStep],
    episode_indices: Sequence[int],
    *,
    norm_tag: Optional[str] = None,
    n_action_steps: Optional[int] = None,
    num_steps: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    policy.reset()
    input_keys = list(policy_cfg.input_features.keys())
    policy_inference_path = _resolve_policy_inference_path(policy_cfg, policy)

    predicted_actions_policy_scale: List[np.ndarray] = []
    gt_actions_policy_scale: List[np.ndarray] = []
    predicted_actions_raw: List[np.ndarray] = []
    gt_actions_raw: List[np.ndarray] = []
    predicted_depths: List[np.ndarray] = []
    frame_indices: List[int] = []
    resolved_style: Optional[str] = None
    molmoact2_history: List[Dict[str, Any]] = []

    for step_idx, dataset_idx in enumerate(episode_indices):
        item = dataset[int(dataset_idx)]
        observation = _extract_input_observation(item, input_keys)
        gt_action_raw = _to_float_vector(item[ACTION])
        processed_observation = preprocessor(observation)
        with torch.inference_mode():
            if _is_molmoact2_policy_type(policy_cfg):
                if policy_inference_path == "select_action_queue":
                    predicted_action = policy.select_action(
                        processed_observation,
                        norm_tag=norm_tag,
                        num_steps=num_steps,
                        n_action_steps=n_action_steps,
                        generator=generator,
                    )
                    resolved_style = resolved_style or _resolve_molmoact2_style(policy_cfg, policy)
                    result = None
                else:
                    molmoact2_history.append(processed_observation)
                    n_obs_steps = _resolve_molmoact2_n_obs_steps(policy_cfg, policy)
                    if len(molmoact2_history) > n_obs_steps:
                        molmoact2_history = molmoact2_history[-n_obs_steps:]
                    result = policy.generate_inference_result_from_observations(
                        list(molmoact2_history),
                        norm_tag=norm_tag,
                        num_steps=num_steps,
                        n_action_steps=n_action_steps,
                        generator=generator,
                    )
                    resolved_style = str(result.style)
            else:
                predicted_action = policy.select_action(processed_observation)

        if _is_molmoact2_policy_type(policy_cfg):
            if not norm_tag:
                raise ValueError("MolmoAct2 open-loop inference requires --norm_tag.")
            action_array = None
            if policy_inference_path == "select_action_queue":
                action_array = _extract_action_vector(predicted_action)
            elif result is not None and result.actions is not None:
                action_array = _extract_action_vector(result.actions)

            if action_array is not None:
                predicted_actions_raw.append(_to_float_vector(action_array))
                gt_actions_raw.append(gt_action_raw)
                predicted_actions_policy_scale.append(
                    _normalize_action_with_policy(policy_cfg, policy, action_array, repo_id=str(norm_tag))
                )
                gt_actions_policy_scale.append(
                    _normalize_action_with_policy(policy_cfg, policy, gt_action_raw, repo_id=str(norm_tag))
                )
            if result is not None and result.depth_bins is not None:
                predicted_depths.append(_to_numpy(result.depth_bins).reshape(-1).astype(np.int64))
        else:
            if normalizer_step is None:
                raise RuntimeError("Expected a NormalizerProcessorStep for non-MolmoAct2 policies.")
            predicted_actions_policy_scale.append(_to_float_vector(predicted_action))
            gt_actions_policy_scale.append(_normalize_gt_action(normalizer_step, gt_action_raw))
            predicted_actions_raw.append(_postprocess_policy_action(postprocessor, predicted_action))
            gt_actions_raw.append(gt_action_raw)
        frame_indices.append(int(_to_numpy(item["frame_index"]).reshape(-1)[0]))
        if verbose and (step_idx == 0 or (step_idx + 1) % 25 == 0 or step_idx + 1 == len(episode_indices)):
            log.info("Rolled out %d/%d frames", step_idx + 1, len(episode_indices))

    return {
        "predicted_actions_policy_scale": None
        if not predicted_actions_policy_scale
        else np.stack(predicted_actions_policy_scale, axis=0).astype(np.float32),
        "gt_actions_policy_scale": None
        if not gt_actions_policy_scale
        else np.stack(gt_actions_policy_scale, axis=0).astype(np.float32),
        "predicted_actions_raw": None
        if not predicted_actions_raw
        else np.stack(predicted_actions_raw, axis=0).astype(np.float32),
        "gt_actions_raw": None
        if not gt_actions_raw
        else np.stack(gt_actions_raw, axis=0).astype(np.float32),
        "predicted_depths": None
        if not predicted_depths
        else np.stack(predicted_depths, axis=0).astype(np.int64),
        "frame_indices": np.asarray(frame_indices, dtype=np.int32),
        "style": resolved_style if resolved_style is not None else _resolve_molmoact2_style(policy_cfg, policy),
        "policy_inference_path": policy_inference_path,
    }


def _action_names(dataset: LeRobotDataset, action_dim: int) -> List[str]:
    action_feature = dataset.features.get(ACTION, {})
    names = action_feature.get("names")
    if isinstance(names, list) and len(names) == action_dim:
        return [str(name) for name in names]
    if isinstance(names, dict):
        flat_names: List[str] = []
        for values in names.values():
            if isinstance(values, list):
                flat_names.extend(str(x) for x in values)
        if len(flat_names) == action_dim:
            return flat_names
    return [f"dim_{idx}" for idx in range(action_dim)]


def _compute_average_mse(predicted: np.ndarray, gt_normalized: np.ndarray) -> tuple[float, List[float]]:
    if predicted.shape != gt_normalized.shape:
        raise ValueError(f"Prediction/GT shape mismatch for MSE: pred={predicted.shape}, gt={gt_normalized.shape}")
    squared_error = np.square(predicted - gt_normalized, dtype=np.float32)
    mean_mse = float(np.mean(squared_error, dtype=np.float64))
    per_dim_mse = [float(x) for x in np.mean(squared_error, axis=0, dtype=np.float64).tolist()]
    return mean_mse, per_dim_mse


def _plot_trajectory(
    predicted: np.ndarray,
    gt_reference: np.ndarray,
    names: Sequence[str],
    out_path: Path,
    *,
    predicted_label: str,
    gt_label: str,
    title: str,
    fixed_ylim: Optional[tuple[float, float]] = None,
) -> None:
    if predicted.shape != gt_reference.shape:
        raise ValueError(f"Prediction/GT shape mismatch: pred={predicted.shape}, gt={gt_reference.shape}")
    if predicted.ndim != 2:
        raise ValueError(f"Expected 2D trajectory arrays, got {predicted.shape}")

    traj_len, action_dim = predicted.shape
    x_values = np.arange(traj_len, dtype=np.int32)
    ncols = 1 if action_dim == 1 else 2
    nrows = math.ceil(action_dim / max(ncols, 1))
    fig_height = max(3.2 * nrows, 3.2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.5 * ncols, fig_height), squeeze=False)
    axes_flat = axes.flatten()

    for dim in range(action_dim):
        ax = axes_flat[dim]
        ax.plot(x_values, gt_reference[:, dim], color="#1f77b4", linewidth=1.8, label=gt_label)
        ax.plot(x_values, predicted[:, dim], color="#d62728", linewidth=1.6, label=predicted_label)
        ax.set_title(str(names[dim]))
        ax.set_xlabel("episode step")
        if fixed_ylim is not None:
            ax.set_ylim(*fixed_ylim)
        ax.grid(alpha=0.25)

    for dim in range(action_dim, len(axes_flat)):
        axes_flat[dim].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.norm_tag = str(args.norm_tag or "").strip()
    args.inference_action_mode = _resolve_inference_action_mode(args)
    args.discrete_action_tokenizer = _requires_discrete_action_tokenizer(
        args.inference_action_mode,
        args.discrete_action_tokenizer,
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    repo_id = _normalize_dataset_name(args.dataset)
    episode_frame_slice = _parse_episode_frames(args.episode_frames)
    resolved_root = _resolve_dataset_root(repo_id, args.dataset_root)
    if resolved_root is not None and args.verbose:
        log.info("Using dataset root: %s", resolved_root)

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=resolved_root,
        video_backend=args.video_backend,
    )
    cfg, policy, preprocessor, postprocessor = _load_policy_and_processors(
        args.checkpoint,
        dataset,
        device=args.device,
        args=args,
    )
    resolved_n_action_steps = args.n_action_steps
    if str(cfg.type) == "molmoact2" and resolved_n_action_steps is None:
        hf_backend = getattr(policy, "_hf_backend", None)
        if hf_backend is not None and args.norm_tag:
            stats = _get_hf_robot_stats(policy)
            resolved_n_action_steps = stats.get_n_action_steps(stats.validate_tag(args.norm_tag))
        else:
            handles = getattr(policy, "_handles", None)
            if handles is not None:
                resolved_n_action_steps = handles.n_action_steps

    if args.policy_type:
        requested_policy_type = str(args.policy_type).strip().lower().replace("-", "_")
        allowed_policy_types = {requested_policy_type}
        if str(cfg.type) not in allowed_policy_types:
            expected = " or ".join(sorted(allowed_policy_types))
            raise ValueError(
                f"Checkpoint policy type is '{cfg.type}', but --policy_type requested '{expected}'."
            )

    normalizer_step = _find_normalizer_step(preprocessor)
    if not _is_molmoact2_policy_type(cfg) and normalizer_step is None:
        raise RuntimeError("Could not find a NormalizerProcessorStep in the loaded preprocessor pipeline.")
    rollout_generator: Optional[torch.Generator] = None
    if _is_molmoact2_policy_type(cfg) and args.seed is not None:
        rollout_generator = torch.Generator(device=str(args.device))
        rollout_generator.manual_seed(args.seed)

    episode_indices = _episode_absolute_indices(
        dataset,
        args.episode_idx,
        episode_frames=episode_frame_slice,
    )
    rollout = _rollout_episode(
        dataset,
        cfg,
        policy,
        preprocessor,
        postprocessor,
        normalizer_step,
        episode_indices,
        norm_tag=args.norm_tag,
        n_action_steps=args.n_action_steps,
        num_steps=args.num_steps,
        generator=rollout_generator,
        verbose=args.verbose,
    )
    predicted_policy_scale = rollout["predicted_actions_policy_scale"]
    gt_policy_scale = rollout["gt_actions_policy_scale"]
    predicted_raw = rollout["predicted_actions_raw"]
    gt_raw = rollout["gt_actions_raw"]
    predicted_depths = rollout["predicted_depths"]
    frame_indices = rollout["frame_indices"]
    resolved_style = rollout["style"]
    action_policy_scale = _resolve_action_policy_scale(
        cfg,
        policy,
        normalizer_step,
        norm_tag=args.norm_tag,
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "frame_indices.npy", frame_indices)
    artifacts: Dict[str, str] = {
        "frame_indices": str(output_dir / "frame_indices.npy"),
    }

    names: Optional[List[str]] = None
    normalization_mask: Optional[List[bool]] = None
    masked_action_dims: List[int] = []
    masked_action_names: List[str] = []
    gripper_action_dims: List[int] = []
    gripper_normalized: Optional[bool] = None
    policy_scale_comparable: Optional[bool] = None
    average_mse: Optional[float] = None
    average_mse_per_dim: Optional[List[float]] = None
    average_mse_raw: Optional[float] = None
    average_mse_raw_per_dim: Optional[List[float]] = None
    average_mse_raw_nongripper: Optional[float] = None
    average_mse_raw_gripper: Optional[float] = None
    if predicted_policy_scale is not None or gt_policy_scale is not None:
        if predicted_policy_scale is None or gt_policy_scale is None or predicted_raw is None or gt_raw is None:
            raise ValueError("Expected both predicted and GT actions when action outputs are present.")
        if predicted_policy_scale.shape != gt_policy_scale.shape:
            raise ValueError(
                f"Prediction/GT shape mismatch in policy scale: pred={predicted_policy_scale.shape}, "
                f"gt={gt_policy_scale.shape}"
            )
        if predicted_raw.shape != gt_raw.shape:
            raise ValueError(
                f"Prediction/GT shape mismatch in raw scale: pred={predicted_raw.shape}, gt={gt_raw.shape}"
            )
        names = _action_names(dataset, int(predicted_policy_scale.shape[1]))
        normalization_mask = _resolve_action_normalization_mask(
            cfg,
            policy,
            normalizer_step,
            int(predicted_policy_scale.shape[1]),
            norm_tag=args.norm_tag,
        )
        masked_action_dims = _masked_action_dims(normalization_mask)
        masked_action_names = [str(names[idx]) for idx in masked_action_dims]
        gripper_action_dims = _gripper_action_dims(names)
        if normalization_mask is not None and gripper_action_dims:
            gripper_normalized = all(bool(normalization_mask[idx]) for idx in gripper_action_dims)
        policy_scale_comparable = None if normalization_mask is None else not masked_action_dims
        average_mse, average_mse_per_dim = _compute_average_mse(predicted_policy_scale, gt_policy_scale)
        average_mse_raw, average_mse_raw_per_dim = _compute_average_mse(predicted_raw, gt_raw)
        nongripper_action_dims = [
            idx for idx in range(int(predicted_raw.shape[1])) if idx not in set(gripper_action_dims)
        ]
        average_mse_raw_nongripper = _compute_average_mse_for_dims(
            predicted_raw,
            gt_raw,
            nongripper_action_dims,
        )
        average_mse_raw_gripper = _compute_average_mse_for_dims(
            predicted_raw,
            gt_raw,
            gripper_action_dims,
        )

        policy_plot_path = output_dir / "action_trajectory_plot_policy_scale.png"
        raw_plot_path = output_dir / "action_trajectory_plot_raw.png"
        legacy_plot_path = output_dir / "action_trajectory_plot.png"
        policy_scale_label = str(action_policy_scale or "policy_scale")
        _plot_trajectory(
            predicted_policy_scale,
            gt_policy_scale,
            names,
            policy_plot_path,
            predicted_label="pred_policy",
            gt_label="gt_policy",
            title=f"LeRobot Policy Predicted Action Trajectory vs GT ({policy_scale_label})",
            fixed_ylim=(-1.2, 1.2),
        )
        _plot_trajectory(
            predicted_raw,
            gt_raw,
            names,
            raw_plot_path,
            predicted_label="pred_raw",
            gt_label="gt_raw",
            title="LeRobot Policy Predicted Action Trajectory vs GT (raw)",
        )

        np.save(output_dir / "predicted_action_trajectory_policy_scale.npy", predicted_policy_scale)
        np.save(output_dir / "gt_action_trajectory_policy_scale.npy", gt_policy_scale)
        np.save(output_dir / "predicted_action_trajectory_raw.npy", predicted_raw)
        np.save(output_dir / "gt_action_trajectory_raw.npy", gt_raw)

        # Backward-compatible aliases. These point to the policy's native action scale,
        # which may coincide with raw actions when the policy uses norm_mode='none'.
        np.save(output_dir / "predicted_action_trajectory_normalized.npy", predicted_policy_scale)
        np.save(output_dir / "gt_action_trajectory_normalized.npy", gt_policy_scale)
        legacy_plot_path.write_bytes(policy_plot_path.read_bytes())
        artifacts.update(
            {
                "plot": str(legacy_plot_path),
                "plot_policy_scale": str(policy_plot_path),
                "plot_raw": str(raw_plot_path),
                "predicted_action_trajectory_policy_scale": str(output_dir / "predicted_action_trajectory_policy_scale.npy"),
                "gt_action_trajectory_policy_scale": str(output_dir / "gt_action_trajectory_policy_scale.npy"),
                "predicted_action_trajectory_raw": str(output_dir / "predicted_action_trajectory_raw.npy"),
                "gt_action_trajectory_raw": str(output_dir / "gt_action_trajectory_raw.npy"),
                "predicted_action_trajectory_normalized": str(output_dir / "predicted_action_trajectory_normalized.npy"),
                "gt_action_trajectory_normalized": str(output_dir / "gt_action_trajectory_normalized.npy"),
            }
        )
    if predicted_depths is not None:
        np.save(output_dir / "predicted_depth_bins.npy", predicted_depths)
        artifacts["predicted_depth_bins"] = str(output_dir / "predicted_depth_bins.npy")

    summary = {
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "policy_type": str(cfg.type),
        "policy_inference_path": rollout["policy_inference_path"],
        "inference_action_mode": str(getattr(cfg, "inference_action_mode", "")) or None,
        "style": str(resolved_style or "") or None,
        "norm_tag": str(args.norm_tag or "") or None,
        "enable_depth_reasoning": bool(getattr(cfg, "enable_depth_reasoning", False)),
        "enable_inference_cuda_graph": bool(getattr(cfg, "enable_inference_cuda_graph", False)),
        "seed": int(args.seed) if args.seed is not None else None,
        "dataset": str(repo_id),
        "dataset_root": None if resolved_root is None else str(resolved_root),
        "episode_idx": int(args.episode_idx),
        "episode_frames": None if episode_frame_slice is None else [int(episode_frame_slice[0]), int(episode_frame_slice[1])],
        "trajectory_len": int(frame_indices.shape[0]),
        "outputs_produced": {
            "actions": predicted_policy_scale is not None,
            "depth": predicted_depths is not None,
        },
        "action_dim": None if predicted_policy_scale is None else int(predicted_policy_scale.shape[1]),
        "depth_dim": None if predicted_depths is None else int(predicted_depths.shape[1]),
        "n_obs_steps": int(getattr(cfg, "n_obs_steps", 1) or 1),
        "n_action_steps": int(resolved_n_action_steps) if resolved_n_action_steps is not None else None,
        "frame_indices": frame_indices.tolist(),
        "action_names": names,
        "average_mse": average_mse,
        "average_mse_per_dim": average_mse_per_dim,
        "average_mse_policy_scale": average_mse,
        "average_mse_policy_scale_per_dim": average_mse_per_dim,
        "average_mse_raw": average_mse_raw,
        "average_mse_raw_per_dim": average_mse_raw_per_dim,
        "average_mse_raw_nongripper": average_mse_raw_nongripper,
        "average_mse_raw_gripper": average_mse_raw_gripper,
        "normalization_mode": action_policy_scale,
        "normalization_mask": normalization_mask,
        "masked_action_dims": masked_action_dims,
        "masked_action_names": masked_action_names,
        "gripper_normalized": gripper_normalized,
        "policy_scale_comparable": policy_scale_comparable,
        "recommended_comparison_scale": "raw" if predicted_raw is not None else None,
        "policy_action_scale": action_policy_scale,
        "prediction_scale": "policy_scale" if predicted_policy_scale is not None else None,
        "gt_scale": "policy_scale" if gt_policy_scale is not None else None,
        "prediction_raw_scale": "raw" if predicted_raw is not None else None,
        "gt_raw_scale": "raw" if gt_raw is not None else None,
        "artifacts": artifacts,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if args.verbose:
        log.info(
            "Loaded policy type=%s n_obs_steps=%s n_action_steps=%s",
            cfg.type,
            getattr(cfg, "n_obs_steps", 1),
            resolved_n_action_steps,
        )
        log.info("Input feature keys=%s", list(cfg.input_features.keys()))
        log.info("Output feature keys=%s", list(cfg.output_features.keys()))
        if average_mse is not None:
            log.info("Average MSE=%.6f", average_mse)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
