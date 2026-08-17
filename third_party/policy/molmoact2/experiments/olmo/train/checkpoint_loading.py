from __future__ import annotations

import faulthandler
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from safetensors.torch import load_file as load_safetensors_file

from olmo.extra_tokens import (
    build_action_added_tokens,
    build_control_added_tokens,
    build_depth_added_tokens,
    build_setup_added_tokens,
    build_state_added_tokens,
)
from olmo.io import resource_path
from olmo.token_layout import (
    TokenLayout,
    checkpoint_llm_token_layout,
    get_checkpoint_wte_rows,
    model_llm_token_layout,
    resolve_source_layout_from_state_dict,
)
from olmo.torch_util import barrier, get_global_rank, get_world_size
from olmo.train.checkpointer import MODEL_FILENAME
from olmo.util import get_hf_access_token, is_hf_checkpoint_ref, resolve_hf_checkpoint_ref

log = logging.getLogger(__name__)

CHECKPOINT_LOAD_DEBUG_DIR_ENV = "OLMO_CHECKPOINT_LOAD_DEBUG_DIR"
CHECKPOINT_LOAD_TRACEBACK_TIMEOUT_ENV = "OLMO_CHECKPOINT_LOAD_TRACEBACK_TIMEOUT_SEC"
HF_INDEX_FILENAMES = ("model.safetensors.index.json", "pytorch_model.safetensors.index.json")


def _checkpoint_debug_file() -> Path | None:
    debug_dir = os.environ.get(CHECKPOINT_LOAD_DEBUG_DIR_ENV)
    if not debug_dir:
        return None
    path = Path(debug_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning("Failed to create checkpoint debug dir %s: %s", path, exc)
        return None
    return path / f"checkpoint_load_rank{get_global_rank():04d}_pid{os.getpid()}.log"


def _checkpoint_debug_log(stage: str, **fields) -> None:
    payload = {
        "stage": stage,
        "rank": get_global_rank(),
        "world_size": get_world_size(),
        **fields,
    }
    message = "checkpoint_load " + " ".join(f"{k}={v}" for k, v in payload.items())
    log.info(message)

    debug_file = _checkpoint_debug_file()
    if debug_file is not None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        try:
            with debug_file.open("a", encoding="utf-8") as handle:
                handle.write(f"{timestamp} {message}\n")
        except OSError as exc:
            log.warning("Failed to write checkpoint debug log %s: %s", debug_file, exc)


class _CheckpointLoadHangGuard:
    def __init__(self) -> None:
        raw_timeout = os.environ.get(CHECKPOINT_LOAD_TRACEBACK_TIMEOUT_ENV)
        try:
            self.timeout_seconds = int(raw_timeout) if raw_timeout else 0
        except ValueError:
            log.warning(
                "Ignoring invalid %s=%r",
                CHECKPOINT_LOAD_TRACEBACK_TIMEOUT_ENV,
                raw_timeout,
            )
            self.timeout_seconds = 0
        self.file_handle = None

    def __enter__(self) -> None:
        if self.timeout_seconds <= 0:
            return
        debug_file = _checkpoint_debug_file()
        if debug_file is not None:
            self.file_handle = debug_file.open("a", encoding="utf-8")
        target = self.file_handle or None
        faulthandler.dump_traceback_later(self.timeout_seconds, repeat=False, file=target)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.timeout_seconds > 0:
            faulthandler.cancel_dump_traceback_later()
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None


def _logged_barrier(label: str) -> None:
    if not dist.is_initialized():
        return
    _checkpoint_debug_log(f"{label}.barrier.begin")
    barrier()
    _checkpoint_debug_log(f"{label}.barrier.end")


def _resolve_hf_checkpoint_dir(checkpoint: str) -> Path:
    checkpoint = resolve_hf_checkpoint_ref(checkpoint)
    path = Path(checkpoint).expanduser()
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            checkpoint,
            repo_type="model",
            token=get_hf_access_token(),
            allow_patterns=("*.json", "*.safetensors"),
        )
    )


def _load_hf_config_dict(checkpoint_dir: Path) -> Dict[str, Any]:
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Hugging Face checkpoint is missing config.json: {checkpoint_dir}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _hf_added_tokens_from_config_dict(config: Dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    if bool(config.get("add_setup_tokens", False)):
        tokens.extend(build_setup_added_tokens())
    if bool(config.get("add_control_tokens", False)):
        tokens.extend(build_control_added_tokens())
    num_state_tokens = int(config.get("num_state_tokens") or 0)
    if num_state_tokens > 0:
        tokens.extend(build_state_added_tokens(num_state_tokens))
    num_action_tokens = int(config.get("num_action_tokens") or 0)
    if num_action_tokens > 0:
        tokens.extend(build_action_added_tokens(num_action_tokens))
    num_depth_tokens = int(config.get("num_depth_tokens") or 0)
    if num_depth_tokens > 0:
        tokens.extend(build_depth_added_tokens(num_depth_tokens))
    return tokens


def _infer_hf_base_tokens(config: Dict[str, Any], total_tokens: int, added_tokens: list[str]) -> int:
    setup_count = len(build_setup_added_tokens()) if bool(config.get("add_setup_tokens", False)) else 0
    control_count = len(build_control_added_tokens()) if bool(config.get("add_control_tokens", False)) else 0
    state_count = (
        len(build_state_added_tokens(int(config.get("num_state_tokens") or 0)))
        if int(config.get("num_state_tokens") or 0) > 0
        else 0
    )
    action_count = (
        len(build_action_added_tokens(int(config.get("num_action_tokens") or 0)))
        if int(config.get("num_action_tokens") or 0) > 0
        else 0
    )
    candidates: list[int] = []
    if config.get("state_start_token_id") is not None:
        candidates.append(int(config["state_start_token_id"]) - setup_count - control_count)
    if config.get("action_output_token_id") is not None:
        candidates.append(int(config["action_output_token_id"]) - setup_count - control_count - state_count)
    if config.get("depth_output_token_id") is not None:
        candidates.append(
            int(config["depth_output_token_id"]) - setup_count - control_count - state_count - action_count
        )
    if candidates:
        return min(candidate for candidate in candidates if candidate > 0)
    if added_tokens:
        return int(config.get("text_config", {}).get("vocab_size") or total_tokens)
    return total_tokens


def _hf_source_layout_from_config_dict(
    config: Dict[str, Any],
    state_dict: Dict[str, torch.Tensor],
) -> TokenLayout | None:
    total_tokens = get_checkpoint_wte_rows(state_dict)
    if total_tokens is None:
        return None
    added_tokens = _hf_added_tokens_from_config_dict(config)
    if not added_tokens:
        return TokenLayout(
            base_tokens=total_tokens,
            added_tokens=0,
            padding_tokens=0,
            total_tokens=total_tokens,
        )
    base_tokens = _infer_hf_base_tokens(config, total_tokens, added_tokens)
    padding_tokens = max(total_tokens - base_tokens - len(added_tokens), 0)
    return TokenLayout(
        base_tokens=base_tokens,
        added_tokens=len(added_tokens),
        padding_tokens=padding_tokens,
        total_tokens=total_tokens,
    )


def _iter_hf_safetensor_files(checkpoint_dir: Path) -> list[Path]:
    for index_name in HF_INDEX_FILENAMES:
        index_path = checkpoint_dir / index_name
        if not index_path.is_file():
            continue
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        filenames = sorted(set(index.get("weight_map", {}).values()))
        return [checkpoint_dir / filename for filename in filenames]
    return sorted(checkpoint_dir.glob("*.safetensors"))


def _convert_hf_state_dict_to_olmo(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    converted: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        if key == "lm_head.weight":
            converted["transformer.ff_out.weight"] = value
            continue
        if key.startswith("model."):
            key = key[len("model."):]
        if key.startswith("transformer.blocks."):
            key = key.replace(".self_attn.", ".").replace(".mlp.", ".")
        converted[key] = value
    return converted


def load_hf_checkpoint_state_dict(
    checkpoint: str,
) -> Tuple[Dict[str, torch.Tensor], TokenLayout | None, int | None]:
    checkpoint_dir = _resolve_hf_checkpoint_dir(checkpoint)
    config = _load_hf_config_dict(checkpoint_dir)
    safetensor_files = _iter_hf_safetensor_files(checkpoint_dir)
    if not safetensor_files:
        raise FileNotFoundError(f"Hugging Face checkpoint has no safetensors files: {checkpoint}")

    raw_state: Dict[str, torch.Tensor] = {}
    total_size = 0
    for filename in safetensor_files:
        total_size += filename.stat().st_size
        raw_state.update(load_safetensors_file(str(filename), device="cpu"))
    state_dict = _convert_hf_state_dict_to_olmo(raw_state)
    source_layout = _hf_source_layout_from_config_dict(config, state_dict)
    return state_dict, source_layout, total_size


def running_mean_std_from_tensors(
    tensors: Iterable[torch.Tensor],
    chunk_numel: int = 4_000_000,
) -> Tuple[float, float]:
    total_count = 0
    total_sum = 0.0
    total_sum_sq = 0.0
    for tensor in tensors:
        if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            continue
        flat = tensor.detach().reshape(-1)
        for start in range(0, flat.numel(), chunk_numel):
            chunk = flat[start:start + chunk_numel].to(dtype=torch.float64)
            total_count += chunk.numel()
            total_sum += chunk.sum().item()
            total_sum_sq += chunk.square().sum().item()

    if total_count == 0:
        return 0.0, 0.02

    mean = total_sum / total_count
    var = max((total_sum_sq / total_count) - (mean * mean), 0.0)
    std = var ** 0.5
    if std == 0.0:
        std = 1e-6
    return mean, std


def running_feature_mean_std_from_tensors(
    tensors: Iterable[torch.Tensor],
    feature_dim: int,
    chunk_rows: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    total_rows = 0
    total_sum = None
    total_sum_sq = None

    for tensor in tensors:
        if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            continue
        if tensor.ndim < 2:
            continue
        if tensor.shape[-1] != feature_dim:
            continue
        rows = tensor.detach().reshape(-1, feature_dim)
        for start in range(0, rows.shape[0], chunk_rows):
            chunk = rows[start:start + chunk_rows].to(dtype=torch.float64)
            if total_sum is None:
                total_sum = torch.zeros(feature_dim, dtype=torch.float64)
                total_sum_sq = torch.zeros(feature_dim, dtype=torch.float64)
            total_rows += chunk.shape[0]
            total_sum += chunk.sum(dim=0)
            total_sum_sq += chunk.square().sum(dim=0)

    if total_rows == 0 or total_sum is None or total_sum_sq is None:
        mean = torch.zeros(feature_dim, dtype=torch.float64)
        std = torch.full((feature_dim,), 0.02, dtype=torch.float64)
        return mean, std

    mean = total_sum / float(total_rows)
    var = torch.clamp((total_sum_sq / float(total_rows)) - mean.square(), min=0.0)
    std = torch.sqrt(var)
    std = torch.where(std == 0.0, torch.full_like(std, 1e-6), std)
    return mean, std


def _resize_token_matrix_rows_for_layout(
    tensor: torch.Tensor,
    *,
    key: str,
    target_layout: TokenLayout,
    source_layout: TokenLayout | None,
    default_std: float,
) -> torch.Tensor:
    target_rows = target_layout.total_tokens
    current_rows = int(tensor.shape[0])
    feature_dim = int(tensor.shape[1])
    if current_rows == target_rows:
        return tensor

    dtype = tensor.dtype
    device = tensor.device
    if source_layout is not None and source_layout.base_tokens > 0:
        stats_tensor = tensor[: min(source_layout.base_tokens, current_rows)]
    else:
        stats_tensor = tensor
    mean_vec, std_vec = running_feature_mean_std_from_tensors([stats_tensor], feature_dim=feature_dim)
    if default_std > 0:
        std_vec = torch.where(std_vec < 1e-10, torch.full_like(std_vec, default_std), std_vec)

    # Start with vector-stat sampling init for all rows, then copy the segments
    # whose token ranges can be aligned between the source and target layouts.
    mean_vec_f32 = mean_vec.to(dtype=torch.float32, device=device)
    std_vec_f32 = std_vec.to(dtype=torch.float32, device=device)
    if mean_vec_f32.numel() != feature_dim or std_vec_f32.numel() != feature_dim:
        log.warning(
            "Feature-stat dimension mismatch for %s (mean=%d, std=%d, feature_dim=%d); "
            "falling back to scalar sampling init.",
            key,
            int(mean_vec_f32.numel()),
            int(std_vec_f32.numel()),
            feature_dim,
        )
        mean_scalar, std_scalar = running_mean_std_from_tensors([stats_tensor])
        if default_std > 0 and std_scalar < 1e-10:
            std_scalar = default_std
        resized = (
            torch.randn((target_rows, feature_dim), dtype=torch.float32, device=device)
            .mul_(std_scalar)
            .add_(mean_scalar)
            .to(dtype=dtype)
        )
    else:
        resized = (
            torch.randn((target_rows, feature_dim), dtype=torch.float32, device=device)
            .mul_(std_vec_f32.unsqueeze(0))
            .add_(mean_vec_f32.unsqueeze(0))
            .to(dtype=dtype)
        )

    copied_rows = 0
    if source_layout is None:
        copy_rows = min(current_rows, target_rows)
        if copy_rows > 0:
            resized[:copy_rows].copy_(tensor[:copy_rows])
            copied_rows += copy_rows
    else:
        source_base_rows = min(source_layout.base_tokens, current_rows)
        source_added_rows = max(min(source_layout.added_tokens, current_rows - source_base_rows), 0)
        source_padding_rows = max(
            min(source_layout.padding_tokens, current_rows - source_base_rows - source_added_rows), 0
        )

        # Legacy checkpoints sometimes store padded rows as part of vocab rows.
        if (
            source_layout.added_tokens == 0
            and source_layout.padding_tokens == 0
            and source_layout.base_tokens == source_layout.total_tokens
            and target_layout.base_tokens > 0
            and target_layout.base_tokens < source_base_rows
        ):
            inferred_source_base_rows = min(target_layout.base_tokens, current_rows)
            inferred_source_padding_rows = max(current_rows - inferred_source_base_rows, 0)
            log.info(
                "Inferring legacy source base/padding split from target layout for %s: "
                "source_base %d->%d, source_padding %d->%d",
                key,
                source_base_rows,
                inferred_source_base_rows,
                source_padding_rows,
                inferred_source_padding_rows,
            )
            source_base_rows = inferred_source_base_rows
            source_added_rows = 0
            source_padding_rows = inferred_source_padding_rows

        copy_base_rows = min(source_base_rows, target_layout.base_tokens)
        if copy_base_rows > 0:
            resized[:copy_base_rows].copy_(tensor[:copy_base_rows])
            copied_rows += copy_base_rows

        copy_added_rows = min(source_added_rows, target_layout.added_tokens)
        if copy_added_rows > 0:
            source_added_start = source_base_rows
            target_added_start = target_layout.base_tokens
            resized[target_added_start:target_added_start + copy_added_rows].copy_(
                tensor[source_added_start:source_added_start + copy_added_rows]
            )
            copied_rows += copy_added_rows

        can_copy_padding_rows = not (source_added_rows == 0 and target_layout.added_tokens > 0)
        if source_padding_rows > 0 and target_layout.padding_tokens > 0 and can_copy_padding_rows:
            copy_padding_rows = min(source_padding_rows, target_layout.padding_tokens)
            source_padding_start = source_base_rows + source_added_rows
            target_padding_start = target_layout.base_tokens + target_layout.added_tokens
            resized[target_padding_start:target_padding_start + copy_padding_rows].copy_(
                tensor[source_padding_start:source_padding_start + copy_padding_rows]
            )
            copied_rows += copy_padding_rows
        elif source_padding_rows > 0 and target_layout.padding_tokens > 0 and not can_copy_padding_rows:
            log.info(
                "Skipping source padding-row copy for %s because target introduces added tokens "
                "(source_added_rows=%d, target_added_rows=%d).",
                key,
                source_added_rows,
                target_layout.added_tokens,
            )

    log.info(
        "Adjusted %s rows from %d to %d (copied=%d, reinit=%d, "
        "mean_avg=%.6f, std_avg=%.6f, target_layout=%s, source_layout=%s)",
        key,
        current_rows,
        target_rows,
        copied_rows,
        max(target_rows - copied_rows, 0),
        mean_vec.mean().item(),
        std_vec.mean().item(),
        target_layout,
        source_layout,
    )
    return resized


def _resize_token_vector_rows_for_layout(
    tensor: torch.Tensor,
    *,
    key: str,
    target_layout: TokenLayout,
    source_layout: TokenLayout | None,
    default_std: float,
) -> torch.Tensor:
    if int(tensor.shape[0]) == target_layout.total_tokens:
        return tensor
    resized_2d = _resize_token_matrix_rows_for_layout(
        tensor.unsqueeze(1),
        key=key,
        target_layout=target_layout,
        source_layout=source_layout,
        default_std=default_std,
    )
    return resized_2d.squeeze(1)


def resize_or_init_base_embedding_for_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    source_layout: TokenLayout | None = None,
) -> None:
    cfg = getattr(model, "config", None)
    if cfg is None and hasattr(model, "module"):
        cfg = getattr(model.module, "config", None)
    llm_cfg = getattr(cfg, "llm", None)
    if llm_cfg is None:
        return

    target_layout = model_llm_token_layout(model)
    default_std = float(getattr(llm_cfg, "new_embedding_init_range", 0.02))
    if target_layout.total_tokens <= 0:
        return

    prefixes = ("transformer.", "")
    for prefix in prefixes:
        emb_key = f"{prefix}wte.embedding"
        legacy_emb_key = f"{prefix}wte.weight"

        if legacy_emb_key in state_dict and emb_key not in state_dict:
            state_dict[emb_key] = state_dict.pop(legacy_emb_key)

        base_tensor = state_dict.get(emb_key)
        if isinstance(base_tensor, torch.Tensor) and base_tensor.ndim == 2:
            state_dict[emb_key] = _resize_token_matrix_rows_for_layout(
                base_tensor,
                key=emb_key,
                target_layout=target_layout,
                source_layout=source_layout,
                default_std=default_std,
            )

        head_key = f"{prefix}ff_out.weight"
        head_tensor = state_dict.get(head_key)
        if isinstance(head_tensor, torch.Tensor) and head_tensor.ndim == 2:
            state_dict[head_key] = _resize_token_matrix_rows_for_layout(
                head_tensor,
                key=head_key,
                target_layout=target_layout,
                source_layout=source_layout,
                default_std=default_std,
            )

        head_bias_key = f"{prefix}ff_out.bias"
        head_bias = state_dict.get(head_bias_key)
        if isinstance(head_bias, torch.Tensor) and head_bias.ndim == 1:
            state_dict[head_bias_key] = _resize_token_vector_rows_for_layout(
                head_bias,
                key=head_bias_key,
                target_layout=target_layout,
                source_layout=source_layout,
                default_std=default_std,
            )


def _normal_like_from_tensor(
    tensor: torch.Tensor,
    shape: Tuple[int, ...],
    *,
    dim: int | None = None,
) -> torch.Tensor:
    if tensor.numel() == 0:
        return torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)

    if dim is not None and tensor.ndim > dim:
        mean = tensor.mean(dim=dim, keepdim=True)
        std = tensor.std(dim=dim, keepdim=True, unbiased=False)
        std = torch.where(std == 0.0, torch.full_like(std, 1e-6), std)
        while mean.ndim < len(shape):
            mean = mean.unsqueeze(-1)
            std = std.unsqueeze(-1)
        return torch.randn(shape, dtype=torch.float32, device=tensor.device).to(tensor.dtype) * std + mean

    mean, std = running_mean_std_from_tensors([tensor])
    return (
        torch.randn(shape, dtype=torch.float32, device=tensor.device).to(tensor.dtype) * float(std)
        + float(mean)
    )


def _resize_2d_tensor(
    tensor: torch.Tensor,
    target_shape: Tuple[int, int],
    *,
    copy_rows: bool,
) -> Tuple[torch.Tensor, int, int]:
    if tuple(tensor.shape) == target_shape:
        return tensor, int(tensor.shape[0]), int(tensor.shape[1])

    if copy_rows:
        resized = _normal_like_from_tensor(tensor, target_shape, dim=0)
    else:
        resized = _normal_like_from_tensor(tensor, target_shape, dim=1)
    rows = min(int(tensor.shape[0]), int(target_shape[0]))
    cols = min(int(tensor.shape[1]), int(target_shape[1]))
    resized[:rows, :cols].copy_(tensor[:rows, :cols])
    return resized, rows, cols


def _resize_1d_tensor(tensor: torch.Tensor, target_length: int) -> Tuple[torch.Tensor, int]:
    if int(tensor.shape[0]) == target_length:
        return tensor, int(tensor.shape[0])
    resized = _normal_like_from_tensor(tensor, (target_length,))
    copied = min(int(tensor.shape[0]), int(target_length))
    resized[:copied].copy_(tensor[:copied])
    return resized, copied


def resize_or_init_action_expert_dim_for_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> None:
    cfg = getattr(model, "config", None)
    if cfg is None and hasattr(model, "module"):
        cfg = getattr(model.module, "config", None)
    action_expert = getattr(model, "action_expert", None)
    if action_expert is None:
        return

    action_cfg = getattr(cfg, "action_expert", None) if cfg is not None else None
    target_dim = getattr(cfg, "max_action_dim", None) if cfg is not None else None
    if target_dim is None and action_cfg is not None:
        target_dim = getattr(action_cfg, "max_action_dim", None)
    if target_dim is None:
        return
    target_dim = int(target_dim)

    embed_key = "action_expert.action_embed.weight"
    final_weight_key = "action_expert.final_layer.linear.weight"
    final_bias_key = "action_expert.final_layer.linear.bias"

    embed_weight = state_dict.get(embed_key)
    final_weight = state_dict.get(final_weight_key)
    if not isinstance(embed_weight, torch.Tensor) and not isinstance(final_weight, torch.Tensor):
        return

    if isinstance(embed_weight, torch.Tensor):
        target_shape = (int(embed_weight.shape[0]), target_dim)
        resized, copied_rows, copied_cols = _resize_2d_tensor(
            embed_weight,
            target_shape,
            copy_rows=False,
        )
        if resized is not embed_weight:
            state_dict[embed_key] = resized
            log.info(
                "Adjusted %s from %s to %s (copied_rows=%d, copied_cols=%d).",
                embed_key,
                tuple(embed_weight.shape),
                tuple(resized.shape),
                copied_rows,
                copied_cols,
            )

    if isinstance(final_weight, torch.Tensor):
        target_shape = (target_dim, int(final_weight.shape[1]))
        resized, copied_rows, copied_cols = _resize_2d_tensor(
            final_weight,
            target_shape,
            copy_rows=True,
        )
        if resized is not final_weight:
            state_dict[final_weight_key] = resized
            log.info(
                "Adjusted %s from %s to %s (copied_rows=%d, copied_cols=%d).",
                final_weight_key,
                tuple(final_weight.shape),
                tuple(resized.shape),
                copied_rows,
                copied_cols,
            )

    final_bias = state_dict.get(final_bias_key)
    if isinstance(final_bias, torch.Tensor):
        resized, copied = _resize_1d_tensor(final_bias, target_dim)
        if resized is not final_bias:
            state_dict[final_bias_key] = resized
            log.info(
                "Adjusted %s from %s to %s (copied=%d).",
                final_bias_key,
                tuple(final_bias.shape),
                tuple(resized.shape),
                copied,
            )


def load_unsharded_checkpoint_allowing_missing_action_expert(path: str, model: torch.nn.Module) -> bool:
    """
    Load an unsharded checkpoint into ``model`` while allowing the checkpoint to omit
    the action_expert weights (e.g., when initializing from a VLM-only checkpoint).

    Returns:
        bool: True if the checkpoint included action_expert weights, False otherwise.
    """
    model_has_action_expert = getattr(model, "action_expert", None) is not None
    model_has_action_expert_depth_gate = getattr(model, "action_expert_depth_gate", None) is not None
    is_hf_checkpoint = is_hf_checkpoint_ref(path)
    checkpoint_file = None
    checkpoint_size_bytes = None
    if get_global_rank() == 0:
        if is_hf_checkpoint:
            checkpoint_file = path
        else:
            checkpoint_file = resource_path(path, MODEL_FILENAME)
            try:
                checkpoint_size_bytes = checkpoint_file.stat().st_size
            except OSError:
                checkpoint_size_bytes = None

    with _CheckpointLoadHangGuard():
        _checkpoint_debug_log(
            "begin",
            checkpoint=path,
            filename=checkpoint_file if checkpoint_file is not None else MODEL_FILENAME,
            checkpoint_size_bytes=checkpoint_size_bytes,
        )

        if get_global_rank() == 0:
            load_start = time.perf_counter()
            _checkpoint_debug_log(
                "rank0.torch_load.begin",
                filename=checkpoint_file,
                checkpoint_size_bytes=checkpoint_size_bytes,
            )
            if is_hf_checkpoint:
                state_dict, source_layout_cfg, checkpoint_size_bytes = load_hf_checkpoint_state_dict(path)
            else:
                state_dict = torch.load(
                    checkpoint_file,
                    map_location="cpu",
                    weights_only=True,
                )
                source_layout_cfg = checkpoint_llm_token_layout(path)
            _checkpoint_debug_log(
                "rank0.torch_load.end",
                filename=checkpoint_file,
                checkpoint_size_bytes=checkpoint_size_bytes,
                elapsed_s=f"{time.perf_counter() - load_start:0.3f}",
                tensor_keys=sum(isinstance(v, torch.Tensor) for v in state_dict.values()),
            )
            source_layout = resolve_source_layout_from_state_dict(source_layout_cfg, state_dict)
            log.info(
                "Source checkpoint token layout: config=%s resolved=%s wte_rows=%s",
                source_layout_cfg,
                source_layout,
                get_checkpoint_wte_rows(state_dict),
            )
            has_action_expert = any(key.startswith("action_expert.") for key in state_dict.keys())
            has_action_expert_depth_gate = any(
                key.startswith("action_expert_depth_gate.") for key in state_dict.keys()
            )
            drop_prefixes = []
            if not has_action_expert or not model_has_action_expert:
                drop_prefixes.append("action_expert.")
            if not has_action_expert_depth_gate or not model_has_action_expert_depth_gate:
                drop_prefixes.append("action_expert_depth_gate.")
            if drop_prefixes:
                filtered_state = {
                    key: value
                    for key, value in state_dict.items()
                    if not any(key.startswith(prefix) for prefix in drop_prefixes)
                }
                if "_metadata" in state_dict:
                    filtered_state["_metadata"] = {
                        k: v
                        for k, v in state_dict["_metadata"].items()
                        if not any(k.startswith(prefix) for prefix in drop_prefixes)
                    }
            else:
                filtered_state = state_dict

            if model_has_action_expert and has_action_expert:
                action_expert = getattr(model, "action_expert", None)
                prepare_state_dict = getattr(action_expert, "prepare_state_dict_for_loading", None)
                if callable(prepare_state_dict):
                    added_action_keys = prepare_state_dict(filtered_state, prefix="action_expert.")
                    if added_action_keys:
                        log.info(
                            "Prepared action expert checkpoint state with %d synthesized keys.",
                            added_action_keys,
                        )

            allow_missing_prefixes = tuple(
                prefix
                for prefix, allow in (
                    ("action_expert.", model_has_action_expert and not has_action_expert),
                    (
                        "action_expert_depth_gate.",
                        model_has_action_expert_depth_gate and not has_action_expert_depth_gate,
                    ),
                )
                if allow
            )
            strict_restore = len(allow_missing_prefixes) == 0

            preprocess_start = time.perf_counter()
            resize_or_init_base_embedding_for_checkpoint(
                filtered_state,
                model,
                source_layout=source_layout,
            )
            resize_or_init_action_expert_dim_for_checkpoint(
                filtered_state,
                model,
            )
            _checkpoint_debug_log(
                "rank0.preprocess.end",
                elapsed_s=f"{time.perf_counter() - preprocess_start:0.3f}",
                has_action_expert=has_action_expert,
                has_action_expert_depth_gate=has_action_expert_depth_gate,
                strict_restore=strict_restore,
                state_keys=len(filtered_state),
            )
        else:
            state_dict = None
            filtered_state = {}
            has_action_expert = False
            has_action_expert_depth_gate = False
            allow_missing_prefixes = ()
            strict_restore = True

        if torch.cuda.is_available():
            broadcast_device = torch.device("cuda", torch.cuda.current_device())
        else:
            broadcast_device = torch.device("cpu")

        if dist.is_initialized():
            _checkpoint_debug_log("broadcast.has_action_expert.begin", checkpoint=path)
            flag = torch.tensor(int(has_action_expert), device=broadcast_device)
            dist.broadcast(flag, src=0)
            has_action_expert = bool(flag.item())
            _checkpoint_debug_log(
                "broadcast.has_action_expert.end",
                checkpoint=path,
                has_action_expert=has_action_expert,
            )
            _checkpoint_debug_log("broadcast.has_action_expert_depth_gate.begin", checkpoint=path)
            depth_gate_flag = torch.tensor(int(has_action_expert_depth_gate), device=broadcast_device)
            dist.broadcast(depth_gate_flag, src=0)
            has_action_expert_depth_gate = bool(depth_gate_flag.item())
            _checkpoint_debug_log(
                "broadcast.has_action_expert_depth_gate.end",
                checkpoint=path,
                has_action_expert_depth_gate=has_action_expert_depth_gate,
            )
            allow_missing_prefixes = tuple(
                prefix
                for prefix, allow in (
                    ("action_expert.", model_has_action_expert and not has_action_expert),
                    (
                        "action_expert_depth_gate.",
                        model_has_action_expert_depth_gate and not has_action_expert_depth_gate,
                    ),
                )
                if allow
            )
            strict_restore = len(allow_missing_prefixes) == 0

            _logged_barrier("pre_model_state_restore")

            _checkpoint_debug_log(
                "set_model_state_dict.begin",
                checkpoint=path,
                has_action_expert=has_action_expert,
                has_action_expert_depth_gate=has_action_expert_depth_gate,
                strict_restore=strict_restore,
                state_keys=len(filtered_state),
            )
            set_state_start = time.perf_counter()
            kv_errors = dist_cp_sd.set_model_state_dict(
                model=model,
                model_state_dict=filtered_state,
                options=dist_cp_sd.StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                    strict=strict_restore,
                ),
            )
            _checkpoint_debug_log(
                "set_model_state_dict.end",
                checkpoint=path,
                elapsed_s=f"{time.perf_counter() - set_state_start:0.3f}",
                missing_keys=len(getattr(kv_errors, "missing_keys", [])) if kv_errors is not None else 0,
                unexpected_keys=len(getattr(kv_errors, "unexpected_keys", [])) if kv_errors is not None else 0,
            )

            if kv_errors is not None and len(kv_errors.unexpected_keys) > 0:
                raise RuntimeError(
                    "Unexpected keys during checkpoint load: "
                    f"{sorted(kv_errors.unexpected_keys)[:8]}"
                )

            if kv_errors is not None and len(kv_errors.missing_keys) > 0:
                disallowed_missing = [
                    key
                    for key in kv_errors.missing_keys
                    if not any(key.startswith(prefix) for prefix in allow_missing_prefixes)
                ]
                if disallowed_missing:
                    raise RuntimeError(
                        "Missing required keys during checkpoint load: "
                        f"{sorted(disallowed_missing)[:8]}"
                    )

            _logged_barrier("post_model_state_restore")
        else:
            incompatible = model.load_state_dict(
                filtered_state,
                strict=strict_restore,
            )
            if incompatible.unexpected_keys:
                raise RuntimeError(
                    "Unexpected keys during checkpoint load: "
                    f"{sorted(incompatible.unexpected_keys)[:8]}"
                )
            if incompatible.missing_keys:
                disallowed_missing = [
                    key
                    for key in incompatible.missing_keys
                    if not any(key.startswith(prefix) for prefix in allow_missing_prefixes)
                ]
                if disallowed_missing:
                    raise RuntimeError(
                        "Missing required keys during checkpoint load: "
                        f"{sorted(disallowed_missing)[:8]}"
                    )

        if get_global_rank() == 0:
            del state_dict
        del filtered_state
        _checkpoint_debug_log(
            "end",
            checkpoint=path,
            has_action_expert=has_action_expert,
            has_action_expert_depth_gate=has_action_expert_depth_gate,
            strict_restore=strict_restore,
        )
        return has_action_expert
