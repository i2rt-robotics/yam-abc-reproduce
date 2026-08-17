from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence

from launch_scripts.lerobot_utils.train_plan import _dedupe_tokens_preserve_order
from olmo.extra_tokens import (
    build_action_added_tokens,
    build_control_added_tokens,
    build_depth_added_tokens,
    build_setup_added_tokens,
    build_state_added_tokens,
)
from olmo.model_configs import VISION_BACKBONES, LLMS
from olmo.models.molmo.data_formatter import DataFormatter
from olmo.models.molmoact2.molmoact2 import MolmoAct2Config
from olmo.models.molmo2.molmo2_preprocessor import Molmo2PreprocessorConfig
from olmo.nn.action_expert import ActionExpertConfig
from olmo.nn.vision_backbone import MolmoVisionBackboneConfig
from olmo.preprocessing.multicrop_preprocessor import MultiCropConfig
from olmo.tokenizer import TokenizerConfig
from olmo.util import get_hf_access_token, resolve_hf_checkpoint_ref

log = logging.getLogger(__name__)
HF_BASE_TOKENIZER = "Qwen/Qwen3-4B"


def _load_hf_json(checkpoint: str, filename: str) -> Dict[str, Any]:
    checkpoint = resolve_hf_checkpoint_ref(checkpoint)
    path = Path(checkpoint).expanduser()
    if path.exists():
        candidate = path / filename
        if not candidate.is_file():
            return {}
    else:
        try:
            from huggingface_hub import hf_hub_download

            candidate = Path(
                hf_hub_download(
                    checkpoint,
                    filename,
                    repo_type="model",
                    token=get_hf_access_token(),
                )
            )
        except Exception as exc:
            log.info("HF checkpoint %s has no %s: %s", checkpoint, filename, exc)
            return {}
    try:
        with candidate.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        log.warning("Failed to read %s from %s: %s", filename, checkpoint, exc)
        return {}


def _hf_added_tokens_from_config(hf_config: Any) -> List[str]:
    tokens: List[str] = []
    if bool(getattr(hf_config, "add_setup_tokens", False)):
        tokens.extend(build_setup_added_tokens())
    if bool(getattr(hf_config, "add_control_tokens", False)):
        tokens.extend(build_control_added_tokens())
    num_state_tokens = int(getattr(hf_config, "num_state_tokens", 0) or 0)
    if num_state_tokens > 0:
        tokens.extend(build_state_added_tokens(num_state_tokens))
    num_action_tokens = int(getattr(hf_config, "num_action_tokens", 0) or 0)
    if num_action_tokens > 0:
        tokens.extend(build_action_added_tokens(num_action_tokens))
    num_depth_tokens = int(getattr(hf_config, "num_depth_tokens", 0) or 0)
    if num_depth_tokens > 0:
        tokens.extend(build_depth_added_tokens(num_depth_tokens))
    return _dedupe_tokens_preserve_order(tokens)


def _infer_hf_base_vocab_size(hf_config: Any, added_tokens: Sequence[str]) -> int:
    setup_count = len(build_setup_added_tokens()) if bool(getattr(hf_config, "add_setup_tokens", False)) else 0
    control_count = len(build_control_added_tokens()) if bool(getattr(hf_config, "add_control_tokens", False)) else 0
    state_count = (
        len(build_state_added_tokens(int(getattr(hf_config, "num_state_tokens", 0) or 0)))
        if int(getattr(hf_config, "num_state_tokens", 0) or 0) > 0
        else 0
    )
    action_count = (
        len(build_action_added_tokens(int(getattr(hf_config, "num_action_tokens", 0) or 0)))
        if int(getattr(hf_config, "num_action_tokens", 0) or 0) > 0
        else 0
    )

    candidates: List[int] = []
    state_start = getattr(hf_config, "state_start_token_id", None)
    if state_start is not None:
        candidates.append(int(state_start) - setup_count - control_count)
    action_output = getattr(hf_config, "action_output_token_id", None)
    if action_output is not None:
        candidates.append(int(action_output) - setup_count - control_count - state_count)
    depth_output = getattr(hf_config, "depth_output_token_id", None)
    if depth_output is not None:
        candidates.append(int(depth_output) - setup_count - control_count - state_count - action_count)
    if candidates:
        return min(candidate for candidate in candidates if candidate > 0)

    text_config = getattr(hf_config, "text_config", hf_config)
    if added_tokens:
        return int(getattr(text_config, "vocab_size"))
    return int(getattr(hf_config, "image_start_token_id", None) or getattr(text_config, "vocab_size"))


def _hf_hidden_act_to_internal(hidden_act: str) -> str:
    return "swiglu" if hidden_act == "silu" else hidden_act


def _hf_llm_config(checkpoint: str, hf_config: Any, added_tokens: Sequence[str]):
    text_config = getattr(hf_config, "text_config", hf_config)
    base_vocab_size = _infer_hf_base_vocab_size(hf_config, added_tokens)
    embedding_size = int(getattr(hf_config, "image_start_token_id", None) or getattr(text_config, "vocab_size"))
    return replace(
        LLMS["qwen3_4b"],
        init_path=None,
        d_model=int(getattr(text_config, "hidden_size")),
        n_heads=int(getattr(text_config, "num_attention_heads")),
        n_kv_heads=int(getattr(text_config, "num_key_value_heads")),
        head_dim=int(getattr(text_config, "head_dim")),
        n_layers=int(getattr(text_config, "num_hidden_layers")),
        mlp_hidden_size=int(getattr(text_config, "intermediate_size")) * 2,
        activation_type=_hf_hidden_act_to_internal(str(getattr(text_config, "hidden_act", "silu"))),
        vocab_size=base_vocab_size,
        embedding_size=embedding_size,
        additional_vocab_size=int(getattr(text_config, "additional_vocab_size", 128)),
        qkv_bias=bool(getattr(text_config, "qkv_bias", False)),
        weight_tying=bool(getattr(text_config, "tie_word_embeddings", False)),
        include_bias=False,
        max_sequence_length=int(getattr(text_config, "max_position_embeddings", 4096)),
        rope_theta=float(getattr(text_config, "rope_theta", 1000000.0)),
        layer_norm_eps=float(getattr(text_config, "layer_norm_eps", 1e-6)),
        residual_dropout=0.0,
        response_residual_dropout=0.0,
        attention_dropout=0.0,
        embedding_dropout=0.0,
        fix_pad_tokenizer=bool(embedding_size > base_vocab_size or added_tokens),
        tokenizer=TokenizerConfig(
            identifier=HF_BASE_TOKENIZER,
            new_tokens_for_both_input_and_output=list(added_tokens),
        ),
    )


def _hf_vision_backbone_config(hf_config: Any) -> MolmoVisionBackboneConfig:
    vit_config = getattr(hf_config, "vit_config")
    adapter_config = getattr(hf_config, "adapter_config")
    vit = replace(
        VISION_BACKBONES["siglip2"],
        init_path=None,
        image_emb_dim=int(getattr(vit_config, "hidden_size")),
        image_mlp_dim=int(getattr(vit_config, "intermediate_size")),
        image_num_layers=int(getattr(vit_config, "num_hidden_layers")),
        image_num_heads=int(getattr(vit_config, "num_attention_heads")),
        image_num_key_value_heads=int(getattr(vit_config, "num_key_value_heads")),
        image_head_dim=int(getattr(vit_config, "head_dim")),
        image_mlp_activations=str(getattr(vit_config, "hidden_act")),
        image_norm_eps=float(getattr(vit_config, "layer_norm_eps")),
        image_default_input_size=tuple(getattr(vit_config, "image_default_input_size")),
        image_patch_size=int(getattr(vit_config, "image_patch_size")),
        image_pos_patch_size=int(getattr(vit_config, "image_patch_size")),
        image_num_pos=int(getattr(vit_config, "image_num_pos")),
        initializer_range=float(getattr(vit_config, "initializer_range", 0.02)),
        float32_attention=bool(getattr(vit_config, "float32_attention", True)),
    )
    return MolmoVisionBackboneConfig(
        vit=vit,
        vit_layers=tuple(getattr(adapter_config, "vit_layers", (-3, -9))),
        pooling_attention_mask=bool(getattr(adapter_config, "pooling_attention_mask", True)),
        image_feature_dropout=float(getattr(adapter_config, "image_feature_dropout", 0.0)),
        image_padding_embed=None,
        normalize_on_gpu=True,
    )


def _hf_video_preprocessor_config(checkpoint: str, hf_config: Any, frame_loading_backend: str) -> Molmo2PreprocessorConfig:
    processor_config = _load_hf_json(checkpoint, "processor_config.json")
    image_config = processor_config.get("image_processor") or _load_hf_json(checkpoint, "preprocessor_config.json")
    video_config = processor_config.get("video_processor") or _load_hf_json(checkpoint, "video_preprocessor_config.json")

    image_pooling = tuple(image_config.get("pooling_size", (2, 2)))
    video_pooling = tuple(video_config.get("pooling_size", (3, 3)))
    max_fps = video_config.get("max_fps", [2])
    if isinstance(max_fps, (int, float)):
        max_fps = [max_fps]

    image = MultiCropConfig(
        crop_mode=str(image_config.get("crop_mode", "resize")),
        use_col_tokens=bool(processor_config.get("image_use_col_tokens", True)),
        max_crops=int(image_config.get("max_crops", 8)),
        high_res_max_crops=24,
        p_high_res=0,
        pooling_h=int(image_pooling[0]),
        pooling_w=int(image_pooling[1]),
        overlap_margins=tuple(image_config.get("overlap_margins", (4, 4))),
        max_multi_image_crops=4,
        multi_image_pooling_h=int(image_pooling[0]),
        multi_image_pooling_w=int(image_pooling[1]),
        use_single_crop_col_tokens=processor_config.get("use_single_crop_col_tokens", False),
        use_single_crop_start_token=bool(processor_config.get("use_single_crop_start_token", True)),
    )
    return Molmo2PreprocessorConfig(
        use_col_tokens=bool(processor_config.get("video_use_col_tokens", False)),
        max_crops=1,
        pooling_h=int(video_pooling[0]),
        pooling_w=int(video_pooling[1]),
        high_res_pooling_h=None,
        high_res_pooling_w=None,
        periodic_high_res_frame=None,
        time_mode="per-frame-compact",
        max_frames=int(video_config.get("num_frames", 64)),
        time_sampling=True,
        loading_method=frame_loading_backend,
        frame_sample_mode=str(video_config.get("frame_sample_mode", "uniform_last_frame")),
        max_fps=max_fps,
        use_frame_special_tokens=bool(getattr(hf_config, "use_frame_special_tokens", True)),
        image=image,
    )


def _hf_action_expert_config(hf_config: Any, llm_layers: int) -> ActionExpertConfig:
    action_config = getattr(hf_config, "action_expert_config", None)
    if action_config is None:
        return ActionExpertConfig(num_layers=llm_layers, hidden_size=768, num_heads=8, max_action_dim=32, max_horizon=30)
    return ActionExpertConfig(
        max_horizon=int(getattr(action_config, "max_action_horizon", getattr(hf_config, "max_action_horizon", 30))),
        max_action_dim=int(getattr(action_config, "max_action_dim", getattr(hf_config, "max_action_dim", 32))),
        hidden_size=int(getattr(action_config, "hidden_size")),
        num_layers=int(getattr(action_config, "num_layers")),
        num_heads=int(getattr(action_config, "num_heads")),
        mlp_ratio=float(getattr(action_config, "mlp_ratio", 8.0 / 3.0)),
        ffn_multiple_of=int(getattr(action_config, "ffn_multiple_of", 256)),
        timestep_embed_dim=int(getattr(action_config, "timestep_embed_dim", 256)),
        dropout=float(getattr(action_config, "dropout", 0.0)),
        attn_dropout=float(getattr(action_config, "attn_dropout", 0.0)),
        context_layer_norm=bool(getattr(action_config, "context_layer_norm", True)),
        qk_norm=bool(getattr(action_config, "qk_norm", True)),
        qk_norm_eps=float(getattr(action_config, "qk_norm_eps", 1e-6)),
        rope=bool(getattr(action_config, "rope", True)),
        causal_attn=bool(getattr(action_config, "causal_attn", False)),
    )


def get_hf_model_config(checkpoint: str, frame_loading_backend: str = "torchcodec_exact") -> MolmoAct2Config:
    from transformers import AutoConfig

    checkpoint = resolve_hf_checkpoint_ref(checkpoint)
    hf_config = AutoConfig.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        token=get_hf_access_token(),
    )
    added_tokens = _hf_added_tokens_from_config(hf_config)
    llm = _hf_llm_config(checkpoint, hf_config, added_tokens)
    vision_backbone = _hf_vision_backbone_config(hf_config)
    mm_preprocessor = _hf_video_preprocessor_config(checkpoint, hf_config, frame_loading_backend)
    action_expert = _hf_action_expert_config(hf_config, llm.n_layers)
    has_action_expert = bool(getattr(hf_config, "add_action_expert", False))

    return MolmoAct2Config(
        llm=llm,
        vision_backbone=vision_backbone,
        data_formatter=DataFormatter(
            system_prompt="demo_or_style_v2",
            message_format="qwen3",
            pointing_format="html-v2",
            prompt_templates="uber_model_v2",
            p_multi_point_all_image=0.5,
            p_choice_content_in_mc=1.0,
            add_setup_tokens=bool(getattr(hf_config, "add_setup_tokens", True)),
            add_control_tokens=bool(getattr(hf_config, "add_control_tokens", True)),
        ),
        mm_preprocessor=mm_preprocessor,
        bi_directional_attn=None,
        action_expert=action_expert,
        add_action_expert=has_action_expert,
        action_expert_depth_gate=bool(getattr(hf_config, "action_expert_depth_gate", False)),
        action_expert_depth_gate_per_layer=bool(getattr(hf_config, "action_expert_depth_gate_per_layer", False)),
        action_expert_depth_gate_init_bias=float(getattr(hf_config, "action_expert_depth_gate_init_bias", -4.0)),
        max_action_dim=int(getattr(hf_config, "max_action_dim", action_expert.max_action_dim)),
        action_horizon=int(getattr(hf_config, "max_action_horizon", action_expert.max_horizon)),
        n_obs_steps=int(getattr(hf_config, "n_obs_steps", 1)),
        action_format=str(getattr(hf_config, "action_mode", "continuous")),
        state_format=str(getattr(hf_config, "state_format", "discrete")),
        flow_matching_num_steps=int(getattr(hf_config, "flow_matching_num_steps", 10)),
        flow_matching_cutoff=float(getattr(hf_config, "flow_matching_cutoff", 1.0)),
        flow_matching_time_offset=float(getattr(hf_config, "flow_matching_time_offset", 0.001)),
        flow_matching_time_scale=float(getattr(hf_config, "flow_matching_time_scale", 0.999)),
        flow_matching_beta_alpha=float(getattr(hf_config, "flow_matching_beta_alpha", 1.0)),
        flow_matching_beta_beta=float(getattr(hf_config, "flow_matching_beta_beta", 1.5)),
        mask_action_dim_padding=bool(getattr(hf_config, "mask_action_dim_padding", True)),
        enable_depth_reasoning=bool(getattr(hf_config, "enable_depth_reasoning", False)),
        num_depth_codes=int(getattr(hf_config, "num_depth_codes", 100)),
    )


def _apply_lerobot_molmoact2_defaults(model_cfg: MolmoAct2Config) -> MolmoAct2Config:
    # Fine-tuning settings
    model_cfg.llm.residual_dropout = 0.1
    model_cfg.llm.response_residual_dropout = 0.0
    model_cfg.data_formatter.prompt_templates = "uber_model_v2"
    model_cfg.data_formatter.message_format = "qwen3"
    model_cfg.data_formatter.system_prompt = "demo_or_style_v2"
    model_cfg.data_formatter.pointing_format = "html-v2"
    model_cfg.data_formatter.p_multi_point_all_image = 0.5
    model_cfg.mm_preprocessor.loss_token_weighting = "root_subsegments_root_tokens"
    model_cfg.data_formatter.p_choice_content_in_mc = 1.0
    model_cfg.vision_backbone.pooling_attention_mask = True

    if model_cfg.mm_preprocessor.image is None:
        raise ValueError("LeRobot training requires an image preprocessor config.")

    # Multi-image settings
    model_cfg.mm_preprocessor.image.max_multi_image_crops = 8
    model_cfg.mm_preprocessor.image.max_images = 5

    model_cfg.llm.max_sequence_length = 4096*4

    # Reduce shared memory requirements
    model_cfg.vision_backbone.normalize_on_gpu = True

    # Action expert settings
    model_cfg.action_expert.num_layers = model_cfg.llm.n_layers
    model_cfg.action_expert.hidden_size = 768
    model_cfg.action_expert.num_heads = 8
    model_cfg.action_expert.max_horizon = max(
        model_cfg.action_expert.max_horizon, model_cfg.action_horizon
    )
    return model_cfg
