from __future__ import annotations

"""Shared helpers for structured added tokens such as action, state, and depth."""

from dataclasses import dataclass
from typing import Any, Callable, Collection, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class IndexedExtraTokenFamily:
    name: str
    start_token: str
    end_token: str
    indexed_token_prefix: str
    default_num_tokens: int

    def build_added_tokens(self, num_tokens: int) -> List[str]:
        if num_tokens <= 0:
            raise ValueError(f"num_{self.name}_tokens must be > 0, got {num_tokens}")
        return [self.start_token, self.end_token] + [
            self.token_for_index(i) for i in range(int(num_tokens))
        ]

    def token_for_index(self, token_id: int) -> str:
        return f"{self.indexed_token_prefix}{int(token_id)}>"

    def is_indexed_token(self, token: str) -> bool:
        return token.startswith(self.indexed_token_prefix) and token.endswith(">")

    def parse_index(self, token: str) -> Optional[int]:
        if not self.is_indexed_token(token):
            return None
        token_id_str = token[len(self.indexed_token_prefix):-1]
        if not token_id_str.isdigit():
            return None
        return int(token_id_str)

    def count_bins(self, tokens: Iterable[str]) -> int:
        return sum(1 for token in tokens if self.parse_index(token) is not None)

    def has_boundaries(self, tokens: Collection[str]) -> bool:
        return self.start_token in tokens and self.end_token in tokens

    def wrap_token_ids(self, token_ids: Sequence[int]) -> str:
        token_pieces = [self.token_for_index(int(token_id)) for token_id in token_ids]
        return f"{self.start_token}{''.join(token_pieces)}{self.end_token}"


ACTION_TOKENS = IndexedExtraTokenFamily(
    name="action",
    start_token="<action_start>",
    end_token="<action_end>",
    indexed_token_prefix="<action_",
    default_num_tokens=2048,
)

STATE_TOKENS = IndexedExtraTokenFamily(
    name="state",
    start_token="<state_start>",
    end_token="<state_end>",
    indexed_token_prefix="<state_",
    default_num_tokens=256,
)

DEPTH_TOKENS = IndexedExtraTokenFamily(
    name="depth",
    start_token="<depth_start>",
    end_token="<depth_end>",
    indexed_token_prefix="<depth_",
    default_num_tokens=128,
)


ACTION_START_TOKEN = ACTION_TOKENS.start_token
ACTION_END_TOKEN = ACTION_TOKENS.end_token
STATE_START_TOKEN = STATE_TOKENS.start_token
STATE_END_TOKEN = STATE_TOKENS.end_token
DEPTH_START_TOKEN = DEPTH_TOKENS.start_token
DEPTH_END_TOKEN = DEPTH_TOKENS.end_token
ACTION_OUTPUT_TOKEN = "<action_output>"
DEPTH_OUTPUT_TOKEN = "<depth_output>"
SETUP_START_TOKEN = "<setup_start>"
SETUP_END_TOKEN = "<setup_end>"
CONTROL_START_TOKEN = "<control_start>"
CONTROL_END_TOKEN = "<control_end>"

DEFAULT_NUM_ACTION_TOKENS = ACTION_TOKENS.default_num_tokens
DEFAULT_NUM_STATE_TOKENS = STATE_TOKENS.default_num_tokens
DEFAULT_NUM_DEPTH_TOKENS = DEPTH_TOKENS.default_num_tokens

SUPPORTED_STATE_FORMATS = {"continuous", "discrete", "both"}
ROBOT_OUTPUT_STYLES = {"robot_action", "robot_depth", "robot_depth_action"}


def build_action_added_tokens(num_action_tokens: int) -> List[str]:
    return [ACTION_OUTPUT_TOKEN] + ACTION_TOKENS.build_added_tokens(num_action_tokens)


def build_state_added_tokens(num_state_tokens: int) -> List[str]:
    return STATE_TOKENS.build_added_tokens(num_state_tokens)


def build_depth_added_tokens(num_depth_tokens: int) -> List[str]:
    return [DEPTH_OUTPUT_TOKEN] + DEPTH_TOKENS.build_added_tokens(num_depth_tokens)


def build_setup_added_tokens() -> List[str]:
    return [SETUP_START_TOKEN, SETUP_END_TOKEN]


def build_control_added_tokens() -> List[str]:
    return [CONTROL_START_TOKEN, CONTROL_END_TOKEN]


def build_discrete_action_string(token_ids: Sequence[int]) -> str:
    return ACTION_TOKENS.wrap_token_ids(token_ids)


def build_discrete_depth_string(
    token_ids: Optional[Sequence[int]],
    *,
    num_depth_tokens: int = DEFAULT_NUM_DEPTH_TOKENS,
) -> str:
    if token_ids is None:
        return ""
    depth_ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    if depth_ids.size == 0:
        return ""
    invalid_mask = np.logical_or(depth_ids < 0, depth_ids >= int(num_depth_tokens))
    if np.any(invalid_mask):
        invalid_ids = depth_ids[invalid_mask][:8].tolist()
        raise ValueError(
            f"Depth token ids must be in [0, {int(num_depth_tokens)}), got invalid ids {invalid_ids}."
        )
    return DEPTH_TOKENS.wrap_token_ids(depth_ids.tolist())


def prepare_action_for_discrete_tokenizer(action: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if action is None:
        return None
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, None, :]
    elif arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim > 3:
        arr = arr.reshape(1, arr.shape[-2], arr.shape[-1])
    return arr


def normalize_discrete_tokenizer_output(tokens_out: Any) -> List[int]:
    if isinstance(tokens_out, dict):
        if "input_ids" in tokens_out:
            tokens_out = tokens_out["input_ids"]
        else:
            first_key = next(iter(tokens_out.keys()))
            tokens_out = tokens_out[first_key]
    if isinstance(tokens_out, np.ndarray):
        tokens_out = tokens_out.tolist()
    if not isinstance(tokens_out, list):
        raise TypeError(
            f"Unexpected output type from discrete tokenizer: {type(tokens_out)}"
        )
    if len(tokens_out) == 0:
        return []
    if isinstance(tokens_out[0], (list, tuple, np.ndarray)):
        token_source = tokens_out[0]
    else:
        token_source = tokens_out
    return [int(x) for x in token_source]


def tokenize_discrete_action(
    action: Optional[np.ndarray],
    processor: Any,
) -> List[int]:
    prepared_action = prepare_action_for_discrete_tokenizer(action)
    if prepared_action is None:
        return []
    return normalize_discrete_tokenizer_output(processor(prepared_action))


def find_out_of_range_token_ids(
    token_ids: Sequence[int],
    max_token_id: Optional[int],
) -> List[int]:
    if max_token_id is None:
        return []
    return [int(token_id) for token_id in token_ids if token_id < 0 or token_id >= max_token_id]


def build_discrete_action_string_from_action(
    action: Optional[np.ndarray],
    processor: Any,
    *,
    max_token_id: Optional[int] = None,
    on_out_of_range: Optional[Callable[[List[int]], None]] = None,
) -> str:
    token_ids = tokenize_discrete_action(action, processor)
    oob_token_ids = find_out_of_range_token_ids(token_ids, max_token_id)
    if oob_token_ids and on_out_of_range is not None:
        on_out_of_range(oob_token_ids)
    return build_discrete_action_string(token_ids)


def build_indexed_token_id_to_bin_map(
    tokenizer: Any,
    added_tokens: Iterable[str],
    family: IndexedExtraTokenFamily,
    *,
    require_single_token: bool = False,
) -> Dict[int, int]:
    token_id_to_bin: Dict[int, int] = {}
    for token in added_tokens:
        token_bin = family.parse_index(token)
        if token_bin is None:
            continue
        token_ids = tokenizer.encode(token)
        if len(token_ids) != 1:
            if require_single_token:
                raise ValueError(
                    f"{family.name} token '{token}' must map to a single tokenizer id, got ids={token_ids}."
                )
            continue
        token_id_to_bin[int(token_ids[0])] = int(token_bin)
    return token_id_to_bin


def resolve_single_token_id(tokenizer: Any, token: str, error_message: str) -> int:
    token_ids = tokenizer.encode(token)
    if len(token_ids) != 1:
        raise ValueError(error_message)
    return int(token_ids[0])


def resolve_family_boundary_token_ids(
    tokenizer: Any,
    family: IndexedExtraTokenFamily,
    error_message: str,
) -> Tuple[int, int]:
    return (
        resolve_single_token_id(tokenizer, family.start_token, error_message),
        resolve_single_token_id(tokenizer, family.end_token, error_message),
    )


def _discretize_uniform_values(
    values: np.ndarray,
    *,
    min_value: float,
    max_value: float,
    num_tokens: int,
    nan_value: float,
    posinf_value: float,
    neginf_value: float,
) -> np.ndarray:
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be > 0, got {num_tokens}")
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=nan_value, posinf=posinf_value, neginf=neginf_value)
    arr = np.clip(arr, min_value, max_value)
    scaled = (arr - min_value) / max(max_value - min_value, 1e-12)
    scaled = scaled * float(num_tokens - 1)
    token_ids = np.rint(scaled).astype(np.int64)
    return np.clip(token_ids, 0, int(num_tokens) - 1)


def discretize_normalized_state(
    state: np.ndarray,
    num_state_tokens: int = DEFAULT_NUM_STATE_TOKENS,
) -> np.ndarray:
    return _discretize_uniform_values(
        state,
        min_value=-1.0,
        max_value=1.0,
        num_tokens=num_state_tokens,
        nan_value=0.0,
        posinf_value=1.0,
        neginf_value=-1.0,
    )


def build_discrete_state_string(
    state: Optional[np.ndarray],
    num_state_tokens: int = DEFAULT_NUM_STATE_TOKENS,
) -> str:
    if state is None:
        return ""
    token_ids = discretize_normalized_state(
        state,
        num_state_tokens=num_state_tokens,
    ).reshape(-1)
    return STATE_TOKENS.wrap_token_ids(token_ids.tolist())


def append_discrete_state_to_prompt(prompt: str, discrete_state_string: str) -> str:
    prompt = str(prompt or "")
    if not discrete_state_string:
        return prompt
    if prompt:
        return f"{prompt}\n{discrete_state_string}"
    return discrete_state_string


def wrap_setup_text(setup_type: str, add_setup_tokens: bool = False) -> str:
    setup_type = str(setup_type or "")
    if setup_type.startswith(SETUP_START_TOKEN) and setup_type.endswith(SETUP_END_TOKEN):
        return setup_type
    if not setup_type or not add_setup_tokens:
        return setup_type
    return f"{SETUP_START_TOKEN}{setup_type}{SETUP_END_TOKEN}"


def wrap_control_text(control_mode: str, add_control_tokens: bool = False) -> str:
    control_mode = str(control_mode or "")
    if control_mode.startswith(CONTROL_START_TOKEN) and control_mode.endswith(CONTROL_END_TOKEN):
        return control_mode
    if not control_mode or not add_control_tokens:
        return control_mode
    return f"{CONTROL_START_TOKEN}{control_mode}{CONTROL_END_TOKEN}"


def build_robot_prompt_fields(
    task: str,
    *,
    style: str,
    discrete_state_string: str = "",
    setup_type: str = "",
    control_mode: str = "",
    add_setup_tokens: bool = False,
    add_control_tokens: bool = False,
) -> Dict[str, str]:
    state_text = str(discrete_state_string or "")
    state_clause = f" The current state of the robot is {state_text}." if state_text else ""
    return {
        "task": str(task or ""),
        "style": str(style or ""),
        "state_text": state_text,
        "state_clause": state_clause,
        "setup_type": wrap_setup_text(str(setup_type or ""), add_setup_tokens=add_setup_tokens),
        "control_mode": wrap_control_text(str(control_mode or ""), add_control_tokens=add_control_tokens),
    }


def get_robot_output_trigger_suffix(style: Optional[str]) -> str:
    if style == "robot_action":
        return ACTION_OUTPUT_TOKEN
    if style == "robot_depth":
        return DEPTH_OUTPUT_TOKEN
    if style == "robot_depth_action":
        return f"{DEPTH_OUTPUT_TOKEN}{ACTION_OUTPUT_TOKEN}"
    return ""

def style_uses_depth_output(style: Optional[str]) -> bool:
    return str(style or "") in {"robot_depth", "robot_depth_action"}


def style_uses_action_output(style: Optional[str]) -> bool:
    return str(style or "") in {"robot_action", "robot_depth_action"}


__all__ = [
    "ACTION_END_TOKEN",
    "ACTION_OUTPUT_TOKEN",
    "ACTION_START_TOKEN",
    "ACTION_TOKENS",
    "CONTROL_END_TOKEN",
    "CONTROL_START_TOKEN",
    "DEFAULT_NUM_ACTION_TOKENS",
    "DEFAULT_NUM_DEPTH_TOKENS",
    "DEFAULT_NUM_STATE_TOKENS",
    "DEPTH_END_TOKEN",
    "DEPTH_OUTPUT_TOKEN",
    "DEPTH_START_TOKEN",
    "DEPTH_TOKENS",
    "IndexedExtraTokenFamily",
    "ROBOT_OUTPUT_STYLES",
    "SETUP_END_TOKEN",
    "SETUP_START_TOKEN",
    "STATE_END_TOKEN",
    "STATE_START_TOKEN",
    "STATE_TOKENS",
    "SUPPORTED_STATE_FORMATS",
    "append_discrete_state_to_prompt",
    "build_indexed_token_id_to_bin_map",
    "build_action_added_tokens",
    "build_control_added_tokens",
    "build_discrete_action_string",
    "build_discrete_action_string_from_action",
    "build_discrete_depth_string",
    "build_discrete_state_string",
    "build_depth_added_tokens",
    "build_robot_prompt_fields",
    "build_setup_added_tokens",
    "build_state_added_tokens",
    "discretize_normalized_state",
    "find_out_of_range_token_ids",
    "get_robot_output_trigger_suffix",
    "normalize_discrete_tokenizer_output",
    "prepare_action_for_discrete_tokenizer",
    "resolve_family_boundary_token_ids",
    "resolve_single_token_id",
    "style_uses_action_output",
    "style_uses_depth_output",
    "tokenize_discrete_action",
    "wrap_control_text",
    "wrap_setup_text",
]
