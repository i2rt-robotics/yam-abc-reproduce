from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from olmo.io import resource_path
from olmo.tokenizer import (
    resolve_added_tokens,
)
from olmo.train.checkpointer import Checkpointer
from olmo.util import get_hf_access_token

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenLayout:
    base_tokens: int
    added_tokens: int
    padding_tokens: int
    total_tokens: int


def _resolve_tokenizer_added_tokens_from_cfg_obj(tokenizer_cfg: Any) -> List[str]:
    if tokenizer_cfg is None:
        return []
    if hasattr(tokenizer_cfg, "resolve_new_tokens_for_both_input_and_output"):
        return list(tokenizer_cfg.resolve_new_tokens_for_both_input_and_output())
    return resolve_added_tokens(getattr(tokenizer_cfg, "new_tokens_for_both_input_and_output", None))


def _resolve_tokenizer_added_tokens_from_raw_cfg(raw_cfg: Any) -> List[str]:
    tokenizer_new_tokens = OmegaConf.select(
        raw_cfg,
        "model.llm.tokenizer.new_tokens_for_both_input_and_output",
    )
    return resolve_added_tokens(tokenizer_new_tokens)


def round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _try_get_base_tokenizer_rows(
    tokenizer_identifier: str | None,
    tokenizer_dir: str | None,
) -> int | None:
    if not tokenizer_identifier:
        return None

    cache_dir = None
    if tokenizer_dir and "://" not in str(tokenizer_dir):
        cache_dir = str(tokenizer_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_identifier,
            token=get_hf_access_token(),
            cache_dir=cache_dir,
            local_files_only=True,
        )
        return int(len(tokenizer))
    except Exception as exc:
        log.warning(
            "Unable to resolve base tokenizer rows from local cache for '%s' (dir=%s): %s. "
            "Falling back to llm.vocab_size.",
            tokenizer_identifier,
            tokenizer_dir,
            exc,
        )
        return None


def model_llm_token_layout(model: torch.nn.Module) -> TokenLayout:
    cfg = getattr(model, "config", None)
    if cfg is None and hasattr(model, "module"):
        cfg = getattr(model.module, "config", None)
    llm_cfg = getattr(cfg, "llm", None)
    if llm_cfg is None:
        return TokenLayout(base_tokens=0, added_tokens=0, padding_tokens=0, total_tokens=0)

    tokenizer_cfg = getattr(llm_cfg, "tokenizer", None)
    tokenizer_identifier = getattr(tokenizer_cfg, "identifier", None)
    tokenizer_dir = getattr(tokenizer_cfg, "tokenizer_dir", None)
    default_base_tokens = int(getattr(llm_cfg, "vocab_size", 0) or 0)
    base_tokens = _try_get_base_tokenizer_rows(tokenizer_identifier, tokenizer_dir) or default_base_tokens

    added_tokens = _resolve_tokenizer_added_tokens_from_cfg_obj(tokenizer_cfg)
    num_added_tokens = len(added_tokens)

    fix_pad = bool(getattr(llm_cfg, "fix_pad_tokenizer", False))
    if num_added_tokens > 0 or fix_pad:
        total_tokens = int(getattr(llm_cfg, "embedding_size", 0) or default_base_tokens)
    else:
        total_tokens = default_base_tokens

    # If tokens are added we can infer exact base split from first added token id.
    # This captures cases where llm.vocab_size already includes legacy padded rows.
    if num_added_tokens > 0:
        try:
            built_tokenizer = llm_cfg.build_tokenizer()
            added_start_ids = built_tokenizer.encode(added_tokens[0])
            if len(added_start_ids) == 1:
                base_tokens = int(added_start_ids[0])
            total_tokens = int(getattr(built_tokenizer, "image_start_token_id"))
        except Exception as exc:
            log.warning(
                "Unable to infer base token split from built tokenizer; "
                "keeping config-based layout inference. Error: %s",
                exc,
            )

    min_total = base_tokens + num_added_tokens
    if total_tokens < min_total:
        log.warning(
            "Derived tokenizer core size (%d) is smaller than base+added tokens (%d), clamping.",
            total_tokens,
            min_total,
        )
        total_tokens = min_total
    padding_tokens = max(total_tokens - base_tokens - num_added_tokens, 0)
    return TokenLayout(
        base_tokens=base_tokens,
        added_tokens=num_added_tokens,
        padding_tokens=padding_tokens,
        total_tokens=total_tokens,
    )


def checkpoint_llm_token_layout(checkpoint_dir: str | None) -> TokenLayout | None:
    if not checkpoint_dir:
        return None
    try:
        config_path = resource_path(checkpoint_dir, Checkpointer.CONFIG_FILENAME)
        raw_cfg = OmegaConf.load(config_path)
    except FileNotFoundError:
        return None
    except Exception as exc:
        log.warning("Failed to read checkpoint config from %s: %s", checkpoint_dir, exc)
        return None

    default_base_tokens = int(OmegaConf.select(raw_cfg, "model.llm.vocab_size") or 0)
    tokenizer_identifier = OmegaConf.select(raw_cfg, "model.llm.tokenizer.identifier")
    tokenizer_dir = OmegaConf.select(raw_cfg, "model.llm.tokenizer.tokenizer_dir")
    base_tokens = _try_get_base_tokenizer_rows(tokenizer_identifier, tokenizer_dir) or default_base_tokens

    added_tokens = _resolve_tokenizer_added_tokens_from_raw_cfg(raw_cfg)
    num_added_tokens = len(added_tokens)

    embedding_size = int(OmegaConf.select(raw_cfg, "model.llm.embedding_size") or 0)
    fix_pad = bool(OmegaConf.select(raw_cfg, "model.llm.fix_pad_tokenizer") or False)
    if num_added_tokens > 0 or fix_pad:
        total_tokens = embedding_size or default_base_tokens
    else:
        total_tokens = default_base_tokens

    min_total = base_tokens + num_added_tokens
    if total_tokens < min_total:
        total_tokens = min_total
    padding_tokens = max(total_tokens - base_tokens - num_added_tokens, 0)
    return TokenLayout(
        base_tokens=base_tokens,
        added_tokens=num_added_tokens,
        padding_tokens=padding_tokens,
        total_tokens=total_tokens,
    )


def get_checkpoint_wte_rows(state_dict: Dict[str, torch.Tensor]) -> int | None:
    for key in ("transformer.wte.embedding", "wte.embedding", "transformer.wte.weight", "wte.weight"):
        tensor = state_dict.get(key)
        if isinstance(tensor, torch.Tensor):
            return int(tensor.shape[0])
    return None


def resolve_source_layout_from_state_dict(
    config_layout: TokenLayout | None,
    state_dict: Dict[str, torch.Tensor],
) -> TokenLayout | None:
    rows = get_checkpoint_wte_rows(state_dict)
    if rows is None:
        return config_layout

    if config_layout is None:
        return TokenLayout(base_tokens=rows, added_tokens=0, padding_tokens=0, total_tokens=rows)

    base_tokens = min(max(config_layout.base_tokens, 0), rows)
    added_tokens = max(config_layout.added_tokens, 0)
    padding_tokens = max(config_layout.padding_tokens, 0)
    expected_total = max(config_layout.total_tokens, 0)

    if rows != expected_total:
        delta = rows - expected_total
        if delta > 0:
            # Extra rows in tensor not accounted for by config; treat as added rows first.
            added_tokens += delta
        else:
            remaining = -delta
            # Prefer shrinking padding first, then added rows, then base.
            trim = min(padding_tokens, remaining)
            padding_tokens -= trim
            remaining -= trim
            if remaining > 0:
                trim = min(added_tokens, remaining)
                added_tokens -= trim
                remaining -= trim
            if remaining > 0:
                base_tokens = max(base_tokens - remaining, 0)

    # Final clamp to the actual tensor rows.
    total = base_tokens + added_tokens + padding_tokens
    if total > rows:
        overflow = total - rows
        trim = min(padding_tokens, overflow)
        padding_tokens -= trim
        overflow -= trim
        if overflow > 0:
            trim = min(added_tokens, overflow)
            added_tokens -= trim
            overflow -= trim
        if overflow > 0:
            base_tokens = max(base_tokens - overflow, 0)
    total = base_tokens + added_tokens + padding_tokens
    if total < rows:
        # Any remaining unassigned rows are most likely added rows in legacy configs.
        added_tokens += (rows - total)
        total = rows

    return TokenLayout(
        base_tokens=base_tokens,
        added_tokens=added_tokens,
        padding_tokens=padding_tokens,
        total_tokens=total,
    )
