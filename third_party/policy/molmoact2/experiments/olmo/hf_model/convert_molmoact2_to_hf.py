import argparse
import dataclasses
import os
import re
import shutil
import logging
import json
import gc
from typing import Dict, Any, Optional

import torch
from transformers import GenerationConfig
from transformers.image_utils import (
    PILImageResampling,
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
)

import numpy as np

from olmo.models.model_config import BaseModelConfig
from olmo.train.checkpointer import load_model_state
from olmo.util import (
    prepare_cli_environment,
    resource_path,
    select_checkpoint
)

from .configuration_molmoact2 import (
    MolmoAct2ActionExpertConfig,
    MolmoAct2Config,
    MolmoAct2VitConfig,
    MolmoAct2AdapterConfig,
    MolmoAct2TextConfig,
)
from .modeling_molmoact2 import MolmoAct2ForConditionalGeneration
from .processing_molmoact2 import MolmoAct2Processor
from .image_processing_molmoact2 import MolmoAct2ImageProcessor
from .video_processing_molmoact2 import MolmoAct2VideoProcessor


logger = logging.getLogger(__name__)


CHAT_TEMPLATE = (
    "{% set DEMO_STYLES = ["
    "'point_count','pointing','cosyn_point','user_qa','long_caption','short_caption',"
    "'video_long_caption','video_short_caption','video_point_track_per_frame',"
    "'video_point_track_start_end','video_point_track_all_frames','video_single_point_track_start_end',"
    "'video_transcript','video_clip_caption_start_end','video_clip_caption_start_end_in_seconds',"
    "'video_clip_transcript_start_end','video_clip_transcript_start_end_in_seconds',"
    "'video_frame_caption_timestamp','video_frame_caption_timestamp_in_seconds',"
    "'correction_qa','text_sft','video_point','video_point_count','video_count','video_count_point',"
    "'multi_image_pointing','multi_image_counting','multi_image_point_then_count','multi_image_count_then_point','demo',"
    "'a_okvqa_mc','ai2_diagram_no_letter','ai2_diagram','science_qa',"
    "'multi_image_mc','multi_image_mc_exp','mantis_instruct_mc',"
    "'video_multiple_choice','video_multiple_choice_count_without_pointing',"
    "'video_multiple_choice_multiple_correct','video_multiple_choice_w_subtitle'"
    "] %}"

    "{% set image_count = namespace(value=0) %}"
    "{% set video_count = namespace(value=0) %}"

    "{% set has_subtitle = messages and messages[0]['role'].lower() == 'subtitle' %}"

    "{% for message in messages %}"
        "{% if message['content'] is not string %}"
            "{% for content in message['content'] %}"
                "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
                    "{% set image_count.value = image_count.value + 1 %}"
                "{% elif content['type'] == 'video' or 'video' in content or 'video_url' in content %}"
                    "{% set video_count.value = video_count.value + 1 %}"
                "{% endif %}"
            "{% endfor %}"
        "{% endif %}"
    "{% endfor %}"

    "{% if image_count.value == 1 %}"
        "{{ '<|image|>' }}"
    "{% elif image_count.value > 1 %}"
        "{% for i in range(image_count.value) %}"
            "{{ 'Image ' ~ (i + 1) ~ '<|image|>' }}"
        "{% endfor %}"
    "{% endif %}"

    "{% for _ in range(video_count.value) %}"
        "{{ '<|video|>' }}"
    "{% endfor %}"

    "{% if has_subtitle %}"
        "{{ messages[0]['content'] }}"
    "{% endif %}"

    "{% for message in messages %}"
        "{% set role = message['role'].lower() %}"

        "{% if role == 'subtitle' %}"
            "{% continue %}"
        "{% endif %}"

        "{% set conv_index = loop.index - (1 if has_subtitle else 0) %}"

        "{%- if (conv_index % 2 == 1 and role != 'user') "
        "or (conv_index % 2 == 0 and role != 'assistant') -%}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
        "{%- endif -%}"

        "{% if message['content'] is string %}"
            "{% set text_content = message['content'] %}"
        "{% else %}"
            "{% set m = namespace(text='') %}"
            "{% for content in message['content'] %}"
                "{% if content['type'] == 'text' %}"
                    "{% if content['style'] is defined and content['style'] not in DEMO_STYLES %}"
                        "{% set seg = content['style'] ~ ': ' ~ content['text'] %}"
                    "{% else %}"
                        "{% set seg = content['text'] %}"
                    "{% endif %}"
                    "{% set m.text = m.text ~ ('' if not m.text else ' ') ~ seg %}"
                "{% endif %}"
            "{% endfor %}"
            "{% set text_content = m.text %}"
        "{% endif %}"

        "{% if role == 'user' %}"
            "{% if not (has_subtitle and loop.index == 2) and not (not has_subtitle and loop.first) %}{{ '<|im_end|>\\n' }}{% endif %}"
            "{{ '<|im_start|>user\\n' }}"
            "{{ text_content }}"
            "{{ '<|im_end|>\\n' }}"
        "{% else %} {# assistant #}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{{ text_content }}"
        "{% endif %}"
    "{% endfor %}"

    "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(subvalue) for subvalue in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _validate_release_config(model_config: BaseModelConfig, *, add_action_expert: bool = True) -> None:
    model_name = str(getattr(model_config, "model_name", getattr(model_config, "_model_name", "")))
    if model_name != "molmoact2":
        raise ValueError(f"Expected source model_name='molmoact2' for MolmoAct2 export, got {model_name!r}.")
    if getattr(model_config, "state_format", None) != "discrete":
        raise ValueError("MolmoAct2 HF export supports only state_format='discrete'.")
    if not add_action_expert:
        return
    if not bool(getattr(model_config, "add_action_expert", False)):
        raise ValueError("MolmoAct2 HF export requires add_action_expert=True.")
    action_expert = getattr(model_config, "action_expert", None)
    if action_expert is None:
        raise ValueError("MolmoAct2 HF export requires action_expert config.")


def _resolve_single_token_id(tokenizer: Any, token: str) -> Optional[int]:
    token_ids = tokenizer.encode(token)
    if len(token_ids) != 1:
        return None
    return int(token_ids[0])


def _resolve_indexed_token_family(
    tokenizer: Any,
    added_tokens: list[str],
    *,
    prefix: str,
) -> tuple[Optional[int], int]:
    items = []
    for token in added_tokens:
        if not (token.startswith(prefix) and token.endswith(">")):
            continue
        token_bin = token[len(prefix):-1]
        if not token_bin.isdigit():
            continue
        token_ids = tokenizer.encode(token)
        if len(token_ids) != 1:
            raise ValueError(f"Indexed token {token!r} must map to a single tokenizer ID, got {token_ids}.")
        items.append((int(token_bin), int(token_ids[0])))
    if not items:
        return None, 0
    items.sort()
    bins = [item[0] for item in items]
    token_ids = [item[1] for item in items]
    expected_bins = list(range(len(items)))
    if bins != expected_bins:
        raise ValueError(f"Indexed token bins for prefix {prefix!r} must be contiguous from 0, got {bins[:8]}...")
    start_id = token_ids[0]
    if token_ids != list(range(start_id, start_id + len(token_ids))):
        raise ValueError(f"Tokenizer IDs for prefix {prefix!r} must be contiguous.")
    return int(start_id), len(items)


def _save_norm_stats(model_config: BaseModelConfig, output_dir: str) -> None:
    proc_cfg = getattr(model_config, "robot_processor", None)
    if proc_cfg is None:
        proc_cfg = getattr(model_config, "robot_preprocessor", None) or getattr(
            model_config,
            "robot_postprocessor",
            None,
        )
    if proc_cfg is None:
        raise ValueError("MolmoAct2 HF export requires robot_processor normalization metadata.")
    metadata_by_tag = {}
    for tag, metadata in dict(getattr(proc_cfg, "metadata_by_tag", {}) or {}).items():
        cleaned_metadata = dict(metadata or {})
        cleaned_metadata.pop("repo_ids", None)
        metadata_by_tag[str(tag)] = cleaned_metadata
    payload = {
        "format": "molmoact2_norm_stats.v1",
        "norm_mode": str(getattr(proc_cfg, "norm_mode", "min_max")),
        "metadata_by_tag": _json_safe(metadata_by_tag),
    }
    with open(os.path.join(output_dir, "norm_stats.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def convert_config(
    model_config: BaseModelConfig,
    attn_implementation: str,
    override_max_model_len: Optional[int],
    add_action_expert: bool = True,
) -> MolmoAct2Config:
    """Convert config to HF-compatible config"""
    _validate_release_config(model_config, add_action_expert=add_action_expert)
    vision_backbone_cfg = model_config.vision_backbone
    vit_config = vision_backbone_cfg.vit
    llm_config = model_config.llm

    molmoact2_vit_config = MolmoAct2VitConfig(
        hidden_size=vit_config.image_emb_dim,
        intermediate_size=vit_config.image_mlp_dim,
        num_hidden_layers=vit_config.image_num_layers,
        num_attention_heads=vit_config.image_num_heads,
        num_key_value_heads=vit_config.image_num_key_value_heads,
        head_dim=vit_config.image_head_dim,
        hidden_act=vit_config.image_mlp_activations,
        layer_norm_eps=vit_config.image_norm_eps,
        image_default_input_size=vit_config.image_default_input_size,
        image_patch_size=vit_config.image_patch_size,
        image_num_pos=vit_config.image_num_pos,
        attention_dropout=0.0,
        residual_dropout=0.0,
        initializer_range=vit_config.initializer_range,
        float32_attention=vit_config.float32_attention,
        attn_implementation=attn_implementation,
    )
    adapter_hidden_act = "silu" if llm_config.activation_type == "swiglu" else llm_config.activation_type
    adapter_intermediate_size = (
        llm_config.mlp_hidden_size if llm_config.mlp_hidden_size is not None
        else llm_config.mlp_ratio * llm_config.d_model
    ) // 2
    molmoact2_adapter_config = MolmoAct2AdapterConfig(
        vit_layers=vision_backbone_cfg.vit_layers,
        pooling_attention_mask=vision_backbone_cfg.pooling_attention_mask,
        hidden_size=vit_config.image_emb_dim,
        num_attention_heads=vit_config.image_num_heads,
        num_key_value_heads=vit_config.image_num_key_value_heads,
        head_dim=vit_config.image_head_dim,
        float32_attention=vit_config.float32_attention,
        attention_dropout=0.0,
        residual_dropout=0.0,
        hidden_act=adapter_hidden_act,
        intermediate_size=adapter_intermediate_size,
        text_hidden_size=llm_config.d_model,
        image_feature_dropout=vision_backbone_cfg.image_feature_dropout,
        initializer_range=llm_config.initializer_range,
        attn_implementation=attn_implementation,
    )
    llm_head_dim = llm_config.d_model // llm_config.n_heads if llm_config.head_dim is None else llm_config.head_dim
    llm_intermediate_size = (
        llm_config.mlp_hidden_size if llm_config.mlp_hidden_size is not None
        else llm_config.mlp_ratio * llm_config.d_model
    ) // 2
    llm_hidden_act = "silu" if llm_config.activation_type == "swiglu" else llm_config.activation_type
    rope_scaling: Optional[Dict[str, Any]] = None
    if llm_config.rope_type != "default":
        rope_scaling = dict(rope_type=llm_config.rope_type)
        for key in [
            "rope_factor",
            "rope_high_freq_factor",
            "rope_low_freq_factor",
            "rope_attention_factor",
            "rope_original_max_position_embeddings",
            "rope_beta_fast",
            "rope_beta_slow",
            "rope_mscale",
            "rope_mscale_all_dim",
            "rope_truncate",
        ]:
            if getattr(llm_config, key) is not None:
                rope_scaling[key[len("rope_"):]] = getattr(llm_config, key)

    max_position_embeddings = llm_config.max_position_embeddings or llm_config.max_sequence_length
    if override_max_model_len is not None:
        max_position_embeddings = override_max_model_len
    rope_scaling_layers: list[int] | None = None
    if llm_config.full_attention_layers is not None:
        # HACK: The original Olmo3 applies scaling to full attention layers,
        # while we applies scaling to slinding attention layers.
        if llm_config.sliding_attention_rope_scaling:
            rope_scaling_layers = [idx for idx in range(llm_config.n_layers) if idx not in llm_config.full_attention_layers]
        else:
            rope_scaling_layers = list(llm_config.full_attention_layers)
    molmoact2_text_config = MolmoAct2TextConfig(
        hidden_size=llm_config.d_model,
        num_attention_heads=llm_config.n_heads,
        num_key_value_heads=llm_config.effective_n_kv_heads,
        head_dim=llm_head_dim,
        vocab_size=llm_config.embedding_size or llm_config.vocab_size,
        additional_vocab_size=llm_config.additional_vocab_size,
        qkv_bias=llm_config.qkv_bias,
        num_hidden_layers=llm_config.n_layers,
        intermediate_size=llm_intermediate_size,
        hidden_act=llm_hidden_act,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        max_position_embeddings=max_position_embeddings,
        rope_theta=llm_config.rope_theta,
        rope_scaling=rope_scaling,
        rope_scaling_layers=rope_scaling_layers,
        use_qk_norm=llm_config.attention_layer_norm,
        qk_norm_type=llm_config.attention_layer_norm_type,
        layer_norm_eps=llm_config.layer_norm_eps,
        norm_after=llm_config.norm_after,
        initializer_range=llm_config.initializer_range,
        attn_implementation=attn_implementation,
    )

    tokenizer = model_config.build_tokenizer()
    image_start_token_id = tokenizer.image_start_token_id
    image_end_token_id = tokenizer.image_end_token_id
    low_res_image_start_token_id = tokenizer.low_res_image_start_token_id
    image_low_res_id = tokenizer.image_low_res_token_id
    image_patch_id = tokenizer.image_patch_token_id
    image_col_id = tokenizer.image_col_token_id
    frame_start_token_id = tokenizer.frame_start_token_id
    frame_end_token_id = tokenizer.frame_end_token_id
    added_tokens = list(model_config.llm.tokenizer.resolve_new_tokens_for_both_input_and_output())
    action_output_token_id = _resolve_single_token_id(tokenizer, "<action_output>")
    action_start_token_id = _resolve_single_token_id(tokenizer, "<action_start>")
    action_end_token_id = _resolve_single_token_id(tokenizer, "<action_end>")
    action_token_start_id, num_action_tokens = _resolve_indexed_token_family(
        tokenizer,
        added_tokens,
        prefix="<action_",
    )
    state_start_token_id = _resolve_single_token_id(tokenizer, "<state_start>")
    state_end_token_id = _resolve_single_token_id(tokenizer, "<state_end>")
    state_token_start_id, num_state_tokens = _resolve_indexed_token_family(
        tokenizer,
        added_tokens,
        prefix="<state_",
    )
    depth_output_token_id = _resolve_single_token_id(tokenizer, "<depth_output>")
    depth_start_token_id = _resolve_single_token_id(tokenizer, "<depth_start>")
    depth_end_token_id = _resolve_single_token_id(tokenizer, "<depth_end>")
    depth_token_start_id, num_depth_tokens = _resolve_indexed_token_family(
        tokenizer,
        added_tokens,
        prefix="<depth_",
    )

    use_frame_special_tokens = getattr(model_config.mm_preprocessor, "use_frame_special_tokens", False)
    action_expert_cfg = getattr(model_config, "action_expert", None)
    max_action_horizon = int(getattr(model_config, "action_horizon"))
    molmoact2_action_expert_config = None
    if add_action_expert:
        molmoact2_action_expert_config = MolmoAct2ActionExpertConfig(
            max_action_horizon=max_action_horizon,
            max_action_dim=int(getattr(action_expert_cfg, "max_action_dim")),
            hidden_size=int(getattr(action_expert_cfg, "hidden_size")),
            num_layers=int(getattr(action_expert_cfg, "num_layers")),
            num_heads=int(getattr(action_expert_cfg, "num_heads")),
            mlp_ratio=float(getattr(action_expert_cfg, "mlp_ratio")),
            ffn_multiple_of=int(getattr(action_expert_cfg, "ffn_multiple_of", 256)),
            timestep_embed_dim=int(getattr(action_expert_cfg, "timestep_embed_dim")),
            dropout=float(getattr(action_expert_cfg, "dropout", 0.0)),
            attn_dropout=float(getattr(action_expert_cfg, "attn_dropout", 0.0)),
            context_layer_norm=bool(getattr(action_expert_cfg, "context_layer_norm", True)),
            qk_norm=bool(getattr(action_expert_cfg, "qk_norm", True)),
            qk_norm_eps=float(getattr(action_expert_cfg, "qk_norm_eps", 1e-6)),
            rope=bool(getattr(action_expert_cfg, "rope", True)),
            causal_attn=bool(getattr(action_expert_cfg, "causal_attn", False)),
        )
    formatter_cfg = getattr(model_config, "data_formatter", None)
    max_action_dim = int(getattr(model_config, "max_action_dim", 32))
    if not add_action_expert:
        max_action_dim = 32

    molmoact2_config = MolmoAct2Config(
        vit_config=molmoact2_vit_config,
        adapter_config=molmoact2_adapter_config,
        text_config=molmoact2_text_config,
        action_expert_config=molmoact2_action_expert_config,
        add_action_expert=bool(add_action_expert),
        image_start_token_id=image_start_token_id,
        low_res_image_start_token_id=low_res_image_start_token_id,
        image_end_token_id=image_end_token_id,
        image_low_res_id=image_low_res_id,
        image_patch_id=image_patch_id,
        image_col_id=image_col_id,
        frame_start_token_id=frame_start_token_id,
        frame_end_token_id=frame_end_token_id,
        use_frame_special_tokens=use_frame_special_tokens,
        initializer_range=llm_config.initializer_range,
        max_action_dim=max_action_dim,
        max_action_horizon=max_action_horizon,
        n_obs_steps=int(getattr(model_config, "n_obs_steps", 1)),
        action_mode=str(
            (
                getattr(model_config, "action_mode", getattr(model_config, "action_format", "continuous"))
                if add_action_expert
                else "discrete"
            )
        ),
        state_format=str(getattr(model_config, "state_format", "discrete")),
        flow_matching_num_steps=int(getattr(model_config, "flow_matching_num_steps", 10)),
        flow_matching_cutoff=float(getattr(model_config, "flow_matching_cutoff", 1.0)),
        flow_matching_time_offset=float(getattr(model_config, "flow_matching_time_offset", 0.001)),
        flow_matching_time_scale=float(getattr(model_config, "flow_matching_time_scale", 0.999)),
        flow_matching_beta_alpha=float(getattr(model_config, "flow_matching_beta_alpha", 1.0)),
        flow_matching_beta_beta=float(getattr(model_config, "flow_matching_beta_beta", 1.5)),
        mask_action_dim_padding=bool(getattr(model_config, "mask_action_dim_padding", True)),
        enable_depth_reasoning=bool(getattr(model_config, "enable_depth_reasoning", False)),
        num_depth_codes=int(getattr(model_config, "num_depth_codes", 100)),
        action_expert_depth_gate=bool(add_action_expert and getattr(model_config, "action_expert_depth_gate", False)),
        action_expert_depth_gate_per_layer=bool(add_action_expert and getattr(model_config, "action_expert_depth_gate_per_layer", False)),
        action_expert_depth_gate_init_bias=float(getattr(model_config, "action_expert_depth_gate_init_bias", -4.0)),
        action_output_token_id=action_output_token_id,
        action_start_token_id=action_start_token_id,
        action_end_token_id=action_end_token_id,
        action_token_start_id=action_token_start_id,
        num_action_tokens=int(num_action_tokens),
        depth_output_token_id=depth_output_token_id,
        depth_start_token_id=depth_start_token_id,
        depth_end_token_id=depth_end_token_id,
        depth_token_start_id=depth_token_start_id,
        num_depth_tokens=int(num_depth_tokens),
        state_start_token_id=state_start_token_id,
        state_end_token_id=state_end_token_id,
        state_token_start_id=state_token_start_id,
        num_state_tokens=int(num_state_tokens),
        add_setup_tokens=bool(getattr(formatter_cfg, "add_setup_tokens", True)),
        add_control_tokens=bool(getattr(formatter_cfg, "add_control_tokens", True)),
        norm_stats_filename="norm_stats.json",
        use_cache=True,
        tie_word_embeddings=False,  # Always false for MolmoAct2
    )
    return molmoact2_config


def convert_lm_head_and_prefix(
    state_dict: dict[str, Any],
    base_model_prefix: str,
    weight_tying: bool
) -> dict[str, Any]:
    new_state_dict = {}
    for key, val in state_dict.items():
        if key == "transformer.ff_out.weight":
            new_key = "lm_head.weight"
        else:
            new_key = f"{base_model_prefix}.{key}"
        new_state_dict[new_key] = val

    if weight_tying:
        new_state_dict["lm_head.weight"] = state_dict["transformer.wte.embedding"]

    return new_state_dict


def convert_molmoact2(
    state_dict: dict[str, Any],
    config: MolmoAct2Config,
    weight_tying: bool,
) -> dict[str, Any]:
    base_model_prefix = MolmoAct2ForConditionalGeneration.base_model_prefix
    new_state_dict = convert_lm_head_and_prefix(state_dict, base_model_prefix, weight_tying)
    model_prefix = f"{base_model_prefix}.transformer"
    qkv_bias = config.qkv_bias if isinstance(config, MolmoAct2TextConfig) else config.text_config.qkv_bias
    use_qk_norm = config.use_qk_norm if isinstance(config, MolmoAct2TextConfig) else config.text_config.use_qk_norm
    for layer_i in range(config.num_hidden_layers):
        prefix = f"{model_prefix}.blocks.{layer_i}"

        move_to_attn = ["att_proj.weight", "attn_out.weight"]
        if qkv_bias:
            move_to_attn.append("att_proj.bias")
        if use_qk_norm:
            move_to_attn += ["q_norm.weight", "k_norm.weight"]

        for k in move_to_attn:
            assert f"{prefix}.self_attn.{k}" not in new_state_dict
            new_state_dict[f"{prefix}.self_attn.{k}"] = new_state_dict.pop(f"{prefix}.{k}")

        move_to_mlp = ["ff_proj.weight", "ff_out.weight"]
        for k in move_to_mlp:
            assert f"{prefix}.mlp.{k}" not in new_state_dict
            new_state_dict[f"{prefix}.mlp.{k}"] = new_state_dict.pop(f"{prefix}.{k}")
    state_encoder_dropped = []
    state_encoder_prefix = f"{base_model_prefix}.action_expert.state_encoder."
    for key in list(new_state_dict):
        if key.startswith(state_encoder_prefix):
            state_encoder_dropped.append(key)
            new_state_dict.pop(key)
    if state_encoder_dropped:
        logger.info(
            "Dropping %d unused continuous-state action expert weights; "
            "HF MolmoAct2 supports only discrete state tokens.",
            len(state_encoder_dropped),
        )
    cross_kv_dropped = []
    cross_kv_pattern = re.compile(
        rf"^{re.escape(base_model_prefix)}\.action_expert\.blocks\.\d+\.cross_attn\.kv_proj\."
    )
    for key in list(new_state_dict):
        if cross_kv_pattern.match(key):
            cross_kv_dropped.append(key)
            new_state_dict.pop(key)
    if cross_kv_dropped:
        logger.info(
            "Dropping %d unused action expert cross-attention KV projection weights; "
            "MolmoAct2 uses shared per-layer context_k_proj/context_v_proj instead.",
            len(cross_kv_dropped),
        )
    if not bool(config.add_action_expert):
        action_dropped = []
        action_prefixes = (
            f"{base_model_prefix}.action_expert.",
            f"{base_model_prefix}.action_expert_depth_gate.",
        )
        for key in list(new_state_dict):
            if key.startswith(action_prefixes):
                action_dropped.append(key)
                new_state_dict.pop(key)
        if action_dropped:
            logger.info(
                "Dropping %d action expert weights because add_action_expert=False.",
                len(action_dropped),
            )
    return new_state_dict


def _refresh_deterministic_hf_buffers(hf_model: MolmoAct2ForConditionalGeneration) -> None:
    """Materialize deterministic buffers that are not present in the source OLMo checkpoint."""
    device = torch.device("cpu")
    for module in hf_model.modules():
        if hasattr(module, "rope_init_fn") and hasattr(module, "inv_freq") and hasattr(module, "config"):
            inv_freq, attention_scaling = module.rope_init_fn(module.config, device)
            module.attention_scaling = attention_scaling
            module.register_buffer("inv_freq", inv_freq.to(dtype=torch.float32), persistent=True)
            module.original_inv_freq = module.inv_freq


def convert_model(
    checkpoint_dir: str,
    model_config: BaseModelConfig,
    hf_config: MolmoAct2Config,
) -> MolmoAct2ForConditionalGeneration:
    """Convert model to HF-compatible model"""
    with torch.device("meta"):
        model = model_config.build_model()
        hf_model = MolmoAct2ForConditionalGeneration(hf_config)
    model.to_empty(device=torch.device("cpu"))
    hf_model.to_empty(device=torch.device("cpu"))

    load_model_state(checkpoint_dir, model)
    model.eval()
    model = model.to(torch.float32)
    state_dict = model.state_dict()

    new_state_dict = convert_molmoact2(state_dict, hf_config, model_config.llm.weight_tying)
    hf_model.eval()
    hf_model = hf_model.to(torch.float32)
    _refresh_deterministic_hf_buffers(hf_model)
    for key, value in hf_model.state_dict().items():
        if key not in new_state_dict and key.endswith(".inv_freq"):
            new_state_dict[key] = value.detach().clone()
    hf_model.load_state_dict(new_state_dict, strict=True)
    return hf_model


def save(
    checkpoint_dir: str,
    output_dir: str,
    attn_implementation: str,
    override_max_model_len: Optional[int],
    max_shard_size: str,
    add_action_expert: bool = True,
) -> None:
    logger.info(f"Loading model config from {checkpoint_dir}")
    config_path = resource_path(select_checkpoint(checkpoint_dir), "config.yaml")
    model_config: BaseModelConfig = BaseModelConfig.load(config_path, key="model", validate_paths=False)

    hf_config = convert_config(
        model_config,
        attn_implementation,
        override_max_model_len,
        add_action_expert=add_action_expert,
    )

    logger.info(f"Save HF-compatible model config and checkpoint to {output_dir}")
    hf_model = convert_model(checkpoint_dir, model_config, hf_config)

    hf_model.save_pretrained(output_dir, max_shard_size=max_shard_size)
    _save_norm_stats(model_config, output_dir)

    gc.collect()

    module_dir = os.path.dirname(os.path.abspath(__file__))
    for filename in (
        "configuration_molmoact2.py",
        "modeling_molmoact2.py",
        "inference.py",
        "processing_molmoact2.py",
        "image_processing_molmoact2.py",
        "video_processing_molmoact2.py",
    ):
        dst = os.path.join(output_dir, filename)
        logger.info("Copy remote-code file %s", filename)
        shutil.copyfile(os.path.join(module_dir, filename), dst)

    tokenizer = model_config.build_tokenizer().tokenizer
    if not tokenizer.bos_token:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    tokenizer.chat_template = CHAT_TEMPLATE

    with open(os.path.join(output_dir, "config.json")) as f:
        config = json.load(f)

    auto_map = config.get("auto_map", None)
    if auto_map is None:
        auto_map = {}
        config["auto_map"] = auto_map
    auto_map.setdefault("AutoConfig", "configuration_molmoact2.MolmoAct2Config")
    if "AutoModelForImageTextToText" not in auto_map:
        logger.warning("Add AutoModelForImageTextToText to auto_map")
        auto_map["AutoModelForImageTextToText"] = "modeling_molmoact2.MolmoAct2ForConditionalGeneration"
    config["bos_token_id"] = tokenizer.bos_token_id
    config["eos_token_id"] = tokenizer.eos_token_id
    config["pad_token_id"] = tokenizer.pad_token_id
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Save tokenizer and processor to {output_dir}")

    mm_cfg = model_config.mm_preprocessor
    vit_cfg = model_config.vision_backbone.vit

    img_cfg = mm_cfg.image

    assert vit_cfg.resize_mode == "siglip", "Only siglip resize is supported for now"
    assert vit_cfg.normalize == "siglip", "Only siglip normalization is supported for now"
    assert img_cfg.crop_mode in {"resize", "overlap-and-resize-c2"}, "Only resize and overlap-and-resize-c2 crop modes are supported for MolmoAct2"
    # assert img_cfg.max_crops == img_cfg.max_multi_image_crops, "max_crops and max_multi_image_crops must be the same"
    # NOTE: Relaxed for spatial finetuned checkpoint where max_crops=8, max_multi_image_crops=4
    assert img_cfg.pooling_w == img_cfg.multi_image_pooling_w, "pooling_w and multi_image_pooling_w must be the same"
    assert img_cfg.pooling_h == img_cfg.multi_image_pooling_h, "pooling_h and multi_image_pooling_h must be the same"

    image_processor = MolmoAct2ImageProcessor(
        size={"height": vit_cfg.image_default_input_size[0], "width": vit_cfg.image_default_input_size[1]},
        resample=PILImageResampling.BILINEAR,
        image_mean=IMAGENET_STANDARD_MEAN,
        image_std=IMAGENET_STANDARD_STD,
        do_convert_rgb=True,
        max_crops=img_cfg.max_crops,
        overlap_margins=img_cfg.overlap_margins,
        crop_mode=img_cfg.crop_mode,
        patch_size=vit_cfg.image_patch_size,
        pooling_size=[img_cfg.pooling_h, img_cfg.pooling_w],
    )

    image_use_col_tokens = img_cfg.use_col_tokens
    use_single_crop_col_tokens = img_cfg.use_single_crop_col_tokens
    use_single_crop_start_token = img_cfg.use_single_crop_start_token

    assert vit_cfg.resize_mode == "siglip", "Only siglip resize is supported for now"
    assert vit_cfg.normalize == "siglip", "Only siglip normalization is supported for now"
    assert mm_cfg.time_mode == "per-frame-compact", "Only per-frame-compact time mode is supported for now"

    max_fps = mm_cfg.max_fps
    if isinstance(max_fps, (tuple, list)):
        assert len(max_fps) == 1, "Only one max_fps is supported for now"
        max_fps = max_fps[0]
    video_processor = MolmoAct2VideoProcessor(
        size={"height": vit_cfg.image_default_input_size[0], "width": vit_cfg.image_default_input_size[1]},
        resample=PILImageResampling.BILINEAR,
        image_mean=IMAGENET_STANDARD_MEAN,
        image_std=IMAGENET_STANDARD_STD,
        do_convert_rgb=True,
        patch_size=vit_cfg.image_patch_size,
        pooling_size=[mm_cfg.pooling_h, mm_cfg.pooling_w],
        frame_sample_mode=mm_cfg.frame_sample_mode,
        num_frames=mm_cfg.max_frames,
        max_fps=max_fps,
        sampling_fps=2,
    )

    video_use_col_tokens = mm_cfg.use_col_tokens
    use_frame_special_tokens = mm_cfg.use_frame_special_tokens

    processor = MolmoAct2Processor(
        image_processor,
        video_processor,
        tokenizer,
        chat_template=CHAT_TEMPLATE,
        image_use_col_tokens=image_use_col_tokens,
        use_single_crop_col_tokens=use_single_crop_col_tokens,
        use_single_crop_start_token=use_single_crop_start_token,
        video_use_col_tokens=video_use_col_tokens,
        use_frame_special_tokens=use_frame_special_tokens,
    )
    processor.save_pretrained(output_dir)
    for filename in (
        "processing_molmoact2.py",
        "image_processing_molmoact2.py",
        "video_processing_molmoact2.py",
    ):
        shutil.copyfile(os.path.join(module_dir, filename), os.path.join(output_dir, filename))
    processor_config_path = os.path.join(output_dir, "processor_config.json")
    if os.path.isfile(processor_config_path):
        with open(processor_config_path, "r", encoding="utf-8") as f:
            processor_config = json.load(f)
        auto_map = processor_config.get("auto_map") or {}
        auto_map.setdefault("AutoProcessor", "processing_molmoact2.MolmoAct2Processor")
        processor_config["auto_map"] = auto_map
        with open(processor_config_path, "w", encoding="utf-8") as f:
            json.dump(processor_config, f, indent=2)

    logger.info(f"Save generation config to {output_dir}")
    generation_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generation_config.save_pretrained(output_dir)

    del hf_model, processor, tokenizer, generation_config
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Convert Molmo checkpoint to HuggingFace format."
    )
    parser.add_argument("checkpoint_dir", help="Location of MolmoAct2 checkpoint.")
    parser.add_argument("output_dir", help="Location to save the converted checkpoint.", default="./hf-ckpt")
    parser.add_argument(
        "--attn_implementation", type=str, default="sdpa", help="Attention type",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument(
        "--override_max_model_len",
        type=int,
        default=36864,
        help="Override the max model length (default: 36864 for MolmoAct2)",
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="5GB",
        help="Maximum size for each checkpoint shard written by save_pretrained (default: 5GB).",
    )
    parser.add_argument(
        "--add_action_expert",
        type=_parse_bool,
        default=True,
        help=(
            "Whether to include the MolmoAct2 action expert in the HF checkpoint. "
            "Use '--add_action_expert false' to export only the VLM behavior."
        ),
    )
    args = parser.parse_args()
    prepare_cli_environment()

    save(
        args.checkpoint_dir,
        args.output_dir,
        args.attn_implementation,
        args.override_max_model_len,
        args.max_shard_size,
        args.add_action_expert,
    )


if __name__ == "__main__":
    main()
