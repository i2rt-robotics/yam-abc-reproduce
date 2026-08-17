from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from os import environ
from typing import Iterable, List, Optional

import requests
from transformers import AutoTokenizer

from .config import BaseConfig
from .torch_util import get_fs_local_rank, barrier
from .util import get_hf_access_token, is_url

try:
    from functools import cache
except ImportError:
    from functools import lru_cache as cache


def _iter_exception_chain(exc: Exception):
    seen = set()
    stack = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        stack.extend(
            next_exc
            for next_exc in (getattr(current, "__cause__", None), getattr(current, "__context__", None))
            if next_exc is not None
        )


def _get_http_status_code(exc: Exception) -> Optional[int]:
    for current in _iter_exception_chain(exc):
        response = getattr(current, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code is None:
            continue
        try:
            return int(status_code)
        except (TypeError, ValueError):
            return None
    return None


def _is_retryable_load_error(exc: Exception) -> bool:
    timeout_error_names = {"ReadTimeout", "ReadTimeoutError", "TimeoutException", "ConnectTimeout"}
    if any(
        isinstance(current, requests.exceptions.ReadTimeout) or current.__class__.__name__ in timeout_error_names
        for current in _iter_exception_chain(exc)
    ):
        return True

    status_code = _get_http_status_code(exc)
    return status_code == 429 or (status_code is not None and 500 <= status_code < 600)


# Special tokens, these should be present in any tokenizer we use since the preprocessor uses them
IMAGE_PATCH_TOKEN = f"<im_patch>"  # Where to insert high-res tokens
IMAGE_LOW_RES_TOKEN = f"<im_low>"  # Where to insert low-res tokens
IM_START_TOKEN = f"<im_start>"
LOW_RES_IMAGE_START_TOKEN = f"<low_res_im_start>"
FRAME_START_TOKEN = f"<frame_start>"
IM_END_TOKEN = f"<im_end>"
FRAME_END_TOKEN= f"<frame_end>"
IM_COL_TOKEN = f"<im_col>"
IMAGE_PROMPT = "<|image|>"
VIDEO_PROMPT = "<|video|>"
POINT_PROMPT = "<|points|>"
TOKEN_INDEX_TOKEN = "<|token_index|>"
SUBPATCH_INDEX_TOKEN = "<|vit_index|>"
SUBPATCH_LOC_TOKEN = "<|vit_loc|>"

EXTRA_TOKENS = (IM_START_TOKEN, IM_END_TOKEN, IMAGE_PATCH_TOKEN, IM_COL_TOKEN, LOW_RES_IMAGE_START_TOKEN,
                IMAGE_PROMPT, IMAGE_LOW_RES_TOKEN, FRAME_START_TOKEN, FRAME_END_TOKEN, VIDEO_PROMPT,
                POINT_PROMPT, TOKEN_INDEX_TOKEN, SUBPATCH_INDEX_TOKEN, SUBPATCH_LOC_TOKEN)

DEFAULT_PAD_MULTIPLE = 1024


def _dedupe_tokens_preserve_order(tokens: Iterable[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for token in tokens:
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def resolve_added_tokens(new_tokens_for_both_input_and_output: Optional[List[str]]) -> List[str]:
    return _dedupe_tokens_preserve_order(new_tokens_for_both_input_and_output or [])


class HfTokenizerWrapper:
    """Tokenizer wrapper

    This exists mostly for legacy reasons since we used to support other kinds of tokenizers
    with different APIs
    """
    def __init__(self, tokenizer, bos_token_id=None, adds_space=False):
        self.adds_space = adds_space
        self.tokenizer = tokenizer
        if bos_token_id is None:
            self.bos_token_id = tokenizer.bos_token_id
        else:
            self.bos_token_id = bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_id = -1
        special_tokens = get_special_token_ids(self)
        self.image_end_token_id = special_tokens[IM_END_TOKEN]
        self.image_start_token_id = special_tokens[IM_START_TOKEN]
        self.low_res_image_start_token_id = special_tokens[LOW_RES_IMAGE_START_TOKEN]
        self.frame_start_token_id = special_tokens[FRAME_START_TOKEN]
        self.frame_end_token_id = special_tokens[FRAME_END_TOKEN]
        self.image_col_token_id = special_tokens[IM_COL_TOKEN]
        self.image_patch_token_id = special_tokens[IMAGE_PATCH_TOKEN]
        self.image_low_res_token_id = special_tokens[IMAGE_LOW_RES_TOKEN]
        self.image_prompt_token_id = special_tokens[IMAGE_PROMPT]
        self.frame_start_token_id = special_tokens[FRAME_START_TOKEN]
        self.frame_end_token_id = special_tokens[FRAME_END_TOKEN]
        self.video_prompt_token_id = special_tokens[VIDEO_PROMPT]
        self.token_index_token_id = special_tokens[TOKEN_INDEX_TOKEN]
        self.subpatch_index_token_id = special_tokens[SUBPATCH_INDEX_TOKEN]
        self.subpatch_loc_token_id = special_tokens[SUBPATCH_LOC_TOKEN]

    def encode(self, x: str):
        return self.tokenizer.encode(x, add_special_tokens=False)

    def decode(self, x: List[int], truncate_at_eos=True):
        x = [int(t) for t in x]

        if self.eos_token_id == self.bos_token_id and (len(x) > 0 and x[0] == self.eos_token_id):
            # Assume an EOS at the start is functioning as BOS
            x = x[1:]

        if truncate_at_eos:
            # Follow seqio and automatically cut off at EOS
            try:
                eos_ix = x.index(self.eos_token_id)
                x = x[:eos_ix]
            except ValueError:
                pass
        else:
            # Keep our special tokens, but skip BOS/EOS
            x = [t for t in x if t != self.eos_token_id and t != self.bos_token_id]
        return self.tokenizer.decode(x)

    def vocab_size(self):
        return len(self.tokenizer)


def build_tokenizer(
    tokenizer_type, has_extra_token=True,
    tokenizer_dir="gs://mm-olmo/tokenizer",
    pad_tokenizer_to=None,
    memory_cache={},
    new_tokens_for_both_input_and_output: Optional[List[str]] = None,
) -> HfTokenizerWrapper:
    added_tokens = _dedupe_tokens_preserve_order(new_tokens_for_both_input_and_output or [])
    cache_key = (
        tokenizer_type,
        has_extra_token,
        pad_tokenizer_to,
        tokenizer_dir,
        tuple(added_tokens),
    )
    if cache_key in memory_cache:
        return memory_cache[cache_key]

    cache_dir = None if tokenizer_dir is None or is_url(tokenizer_dir) else tokenizer_dir

    def _load_hf_tokenizer(*, local_files_only: bool) -> AutoTokenizer:
        attempts = 1 if local_files_only else 3
        for attempt in range(1, attempts + 1):
            try:
                return AutoTokenizer.from_pretrained(
                    tokenizer_type,
                    token=get_hf_access_token(),
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
            except Exception as exc:
                if local_files_only or attempt >= attempts or not _is_retryable_load_error(exc):
                    raise
                sleep_s = attempt
                logging.warning(
                    "Failed to load tokenizer '%s' (attempt %d/%d). Retrying in %ss. Exception: %s",
                    tokenizer_type,
                    attempt,
                    attempts,
                    sleep_s,
                    exc,
                )
                time.sleep(sleep_s)

    # Only one rank per shared filesystem should talk to the Hub. Other ranks wait for the
    # cache to be populated and then read from disk to avoid duplicate API requests.
    tokenizer = None
    if get_fs_local_rank() == 0:
        try:
            tokenizer = _load_hf_tokenizer(local_files_only=True)
        except Exception as exc:
            logging.info(
                "Local tokenizer cache load failed for '%s'. Falling back to Hub access. Exception: %s",
                tokenizer_type,
                exc,
            )
            tokenizer = _load_hf_tokenizer(local_files_only=False)
    barrier()
    if tokenizer is None:
        try:
            tokenizer = _load_hf_tokenizer(local_files_only=True)
        except Exception as exc:
            logging.warning(
                "Local tokenizer cache load failed for '%s'. Retrying with Hub access. Exception: %s",
                tokenizer_type,
                exc,
            )
            tokenizer = _load_hf_tokenizer(local_files_only=False)

    extra_tokens = list(EXTRA_TOKENS)

    PADDING_TOKENS: List[str] = []

    if pad_tokenizer_to is not None:
        assert len(tokenizer) <= pad_tokenizer_to
        base_vocab = tokenizer.get_vocab()
        num_effective_added_tokens = sum(1 for tok in added_tokens if tok not in base_vocab)
        n_extra_tokens = pad_tokenizer_to - len(tokenizer) - num_effective_added_tokens
        if n_extra_tokens < 0:
            raise ValueError(
                f"pad_tokenizer_to={pad_tokenizer_to} is too small for "
                f"{num_effective_added_tokens} added tokens "
                f"(base tokenizer len={len(tokenizer)}). Increase llm.embedding_size."
            )
        # This handles a case where the LLM embedding matrix is larger than the vocab size
        # We need the extra tokens in `EXTRA_TOKENS` to be assigned id's higher than the embedding
        # matrix size, not the vocab size, since we will concat the embedding and matrix with
        # the special token embedding matrix, so we pad the vocab with additional special tokens
        if n_extra_tokens > 0:
            logging.info(f"Padding tokenizer with {n_extra_tokens} tokens")
            PADDING_TOKENS = [f"<extra_{i}>" for i in range(n_extra_tokens)]

    bos_token_id = None

    logging.info(f"original vocab size: {len(tokenizer)}")

    if added_tokens:
        num_added_tokens = tokenizer.add_tokens(added_tokens)
        logging.info(f"number of added tokens: {num_added_tokens}")
        logging.info(f"new vocab size: {len(tokenizer)}")

    if len(PADDING_TOKENS) > 0:
        tokenizer.add_special_tokens({"additional_special_tokens": PADDING_TOKENS})

    logging.info(f"padded vocab size: {len(tokenizer)}")

    additional_special_tokens = {"additional_special_tokens": extra_tokens}
    tokenizer.add_special_tokens(additional_special_tokens)

    logging.info(f"final vocab size: {len(tokenizer)}")

    if tokenizer.bos_token_id is None:
        # These tokenizers do not have a BOS, and instead use EOS as a generic seperator token.
        # In this case we will use EOS as BOS
        bos_token_id = tokenizer.eos_token_id

    if pad_tokenizer_to is not None:
        for ix, tok in enumerate(EXTRA_TOKENS):
            ids = tokenizer.encode(tok, add_special_tokens=False)
            assert ids == [pad_tokenizer_to + ix]

    tok = HfTokenizerWrapper(tokenizer, bos_token_id=bos_token_id, adds_space=False)
    memory_cache[cache_key] = tok
    return tok


def get_special_token_ids(tokenizer):
    if isinstance(tokenizer, HfTokenizerWrapper):
        ids = tokenizer.encode("".join(EXTRA_TOKENS))
        if len(ids) == len(EXTRA_TOKENS) + 1:
            ids = ids[1:]
    else:
        ids = tokenizer.encode(" ".join(EXTRA_TOKENS))

    assert len(ids) == len(EXTRA_TOKENS)
    return {k: i for k, i in zip(EXTRA_TOKENS, ids)}


@dataclass
class TokenizerConfig(BaseConfig):
    identifier: str = "gpt2"
    tokenizer_dir: Optional[str] = None
    new_tokens_for_both_input_and_output: List[str] = field(default_factory=list)

    def resolve_new_tokens_for_both_input_and_output(self) -> List[str]:
        return resolve_added_tokens(self.new_tokens_for_both_input_and_output)

    @classmethod
    def update_legacy_settings(cls, config):
        # Drop legacy fields that were replaced by `new_tokens_for_both_input_and_output`.
        for key in ("add_action_tokens", "num_action_tokens", "num_additional_action_tokens"):
            if key in config:
                del config[key]
        return config

    def build(self, pad_tokenizer_to):
        return build_tokenizer(
            self.identifier,
            tokenizer_dir=self.tokenizer_dir,
            pad_tokenizer_to=pad_tokenizer_to,
            new_tokens_for_both_input_and_output=self.resolve_new_tokens_for_both_input_and_output(),
        )
