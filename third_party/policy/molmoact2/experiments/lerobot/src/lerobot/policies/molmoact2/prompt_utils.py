from __future__ import annotations

"""Helpers for turning robot inputs into final MolmoAct2 text prompts."""

import dataclasses
from typing import Any, Dict, List, Optional

import numpy as np

from olmo.extra_tokens import (
    DEFAULT_NUM_STATE_TOKENS,
    ROBOT_OUTPUT_STYLES,
    append_discrete_state_to_prompt,
    build_discrete_state_string,
    build_robot_prompt_fields,
)
from olmo.models.molmo.data_formatter import DataFormatter

DEFAULT_PROMPT_TEMPLATES = "uber_model_v2"
DEFAULT_MESSAGE_FORMAT = "qwen3"
DEFAULT_SYSTEM_PROMPT = "demo_or_style_v2"


def _to_numpy_state(normalized_states: Any) -> Optional[np.ndarray]:
    if normalized_states is None:
        return None
    if hasattr(normalized_states, "detach") and hasattr(normalized_states, "cpu"):
        normalized_states = normalized_states.detach().cpu().numpy()
    return np.asarray(normalized_states, dtype=np.float32)


def _resolve_rng(rng: Optional[np.random.RandomState]) -> np.random.RandomState:
    return rng if rng is not None else np.random.RandomState(0)


def _resolve_formatter(
    formatter: Optional[DataFormatter],
    *,
    prompt_templates: Optional[str],
    message_format: Optional[str],
    system_prompt: Optional[str],
    always_start_with_space: Optional[bool],
    add_setup_tokens: Optional[bool],
    add_control_tokens: Optional[bool],
) -> DataFormatter:
    if formatter is None:
        return DataFormatter(
            prompt_templates=(
                DEFAULT_PROMPT_TEMPLATES if prompt_templates is None else str(prompt_templates)
            ),
            message_format=(
                DEFAULT_MESSAGE_FORMAT if message_format is None else str(message_format)
            ),
            system_prompt=DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt,
            always_start_with_space=bool(always_start_with_space)
            if always_start_with_space is not None
            else False,
            add_setup_tokens=bool(add_setup_tokens) if add_setup_tokens is not None else False,
            add_control_tokens=bool(add_control_tokens)
            if add_control_tokens is not None
            else False,
        )

    updates: Dict[str, Any] = {}
    if prompt_templates is not None:
        updates["prompt_templates"] = str(prompt_templates)
    if message_format is not None:
        updates["message_format"] = str(message_format)
    if system_prompt is not None:
        updates["system_prompt"] = system_prompt
    if always_start_with_space is not None:
        updates["always_start_with_space"] = bool(always_start_with_space)
    if add_setup_tokens is not None:
        updates["add_setup_tokens"] = bool(add_setup_tokens)
    if add_control_tokens is not None:
        updates["add_control_tokens"] = bool(add_control_tokens)
    if updates:
        return dataclasses.replace(formatter, **updates)
    return formatter


def build_prompt_fields(
    *,
    task: str,
    style: str = "robot_action",
    normalized_states: Any = None,
    discrete_state_string: Optional[str] = None,
    setup_type: str = "",
    control_mode: str = "",
    num_state_tokens: int = DEFAULT_NUM_STATE_TOKENS,
    add_setup_tokens: bool = False,
    add_control_tokens: bool = False,
    extra_message_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    task_text = str(task or "")
    style_text = str(style or "")
    if discrete_state_string is None:
        state_array = _to_numpy_state(normalized_states)
        discrete_state_string = build_discrete_state_string(
            state_array,
            num_state_tokens=num_state_tokens,
        )

    if style_text in ROBOT_OUTPUT_STYLES:
        prompt_fields: Dict[str, Any] = build_robot_prompt_fields(
            task_text,
            style=style_text,
            discrete_state_string=discrete_state_string,
            setup_type=str(setup_type or ""),
            control_mode=str(control_mode or ""),
            add_setup_tokens=add_setup_tokens,
            add_control_tokens=add_control_tokens,
        )
    else:
        prompt_text = append_discrete_state_to_prompt(task_text, discrete_state_string)
        prompt_fields = {
            "question": prompt_text,
            "style": style_text,
        }

    if extra_message_fields:
        prompt_fields.update(extra_message_fields)
    return prompt_fields


def build_formatted_messages(
    *,
    task: str,
    normalized_states: Any = None,
    style: str = "robot_action",
    setup_type: str = "",
    control_mode: str = "",
    discrete_state_string: Optional[str] = None,
    num_state_tokens: int = DEFAULT_NUM_STATE_TOKENS,
    formatter: Optional[DataFormatter] = None,
    prompt_templates: Optional[str] = None,
    message_format: Optional[str] = None,
    system_prompt: Optional[str] = None,
    always_start_with_space: Optional[bool] = None,
    add_setup_tokens: Optional[bool] = None,
    add_control_tokens: Optional[bool] = None,
    extra_message_fields: Optional[Dict[str, Any]] = None,
    rng: Optional[np.random.RandomState] = None,
) -> List[str]:
    formatter = _resolve_formatter(
        formatter,
        prompt_templates=prompt_templates,
        message_format=message_format,
        system_prompt=system_prompt,
        always_start_with_space=always_start_with_space,
        add_setup_tokens=add_setup_tokens,
        add_control_tokens=add_control_tokens,
    )
    if str(style or "") in ROBOT_OUTPUT_STYLES:
        if formatter.prompt_templates not in {"uber_model", "uber_model_v2"}:
            formatter = dataclasses.replace(
                formatter,
                prompt_templates=DEFAULT_PROMPT_TEMPLATES,
            )
        if formatter.message_format in {None, "none"}:
            formatter = dataclasses.replace(
                formatter,
                message_format=DEFAULT_MESSAGE_FORMAT,
            )
        if formatter.system_prompt in {None, "none"}:
            formatter = dataclasses.replace(
                formatter,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
            )

    prompt_fields = build_prompt_fields(
        task=task,
        style=style,
        normalized_states=normalized_states,
        discrete_state_string=discrete_state_string,
        setup_type=setup_type,
        control_mode=control_mode,
        num_state_tokens=num_state_tokens,
        add_setup_tokens=bool(formatter.add_setup_tokens),
        add_control_tokens=bool(formatter.add_control_tokens),
        extra_message_fields=extra_message_fields,
    )
    messages, _ = formatter(
        {"messages": prompt_fields},
        is_training=False,
        for_inference=True,
        rng=_resolve_rng(rng),
    )
    return messages


def build_final_text_input(
    *,
    task: str,
    normalized_states: Any = None,
    style: str = "robot_action",
    setup_type: str = "",
    control_mode: str = "",
    discrete_state_string: Optional[str] = None,
    num_state_tokens: int = DEFAULT_NUM_STATE_TOKENS,
    formatter: Optional[DataFormatter] = None,
    prompt_templates: Optional[str] = None,
    message_format: Optional[str] = None,
    system_prompt: Optional[str] = None,
    always_start_with_space: Optional[bool] = None,
    add_setup_tokens: Optional[bool] = None,
    add_control_tokens: Optional[bool] = None,
    extra_message_fields: Optional[Dict[str, Any]] = None,
    rng: Optional[np.random.RandomState] = None,
) -> str:
    return "".join(
        build_formatted_messages(
            task=task,
            normalized_states=normalized_states,
            style=style,
            setup_type=setup_type,
            control_mode=control_mode,
            discrete_state_string=discrete_state_string,
            num_state_tokens=num_state_tokens,
            formatter=formatter,
            prompt_templates=prompt_templates,
            message_format=message_format,
            system_prompt=system_prompt,
            always_start_with_space=always_start_with_space,
            add_setup_tokens=add_setup_tokens,
            add_control_tokens=add_control_tokens,
            extra_message_fields=extra_message_fields,
            rng=rng,
        )
    )


__all__ = [
    "DEFAULT_MESSAGE_FORMAT",
    "DEFAULT_PROMPT_TEMPLATES",
    "DEFAULT_SYSTEM_PROMPT",
    "build_final_text_input",
    "build_formatted_messages",
    "build_prompt_fields",
]
