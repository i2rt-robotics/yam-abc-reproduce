import logging
from typing import Dict
from dataclasses import replace

from olmo.models.molmo.data_formatter import DataFormatter
from olmo.data.data_loader import DataLoaderConfig
from olmo.models.molmo.molmo_preprocessor import MolmoPreprocessorConfig
from olmo.models.molmo2.molmo2 import Molmo2Config
from olmo.models.molmo2.molmo2_preprocessor import Molmo2PreprocessorConfig
from olmo.eval.inf_evaluator import InfDatasetEvaluatorConfig, EvaluatorConfig
from olmo.eval.loss_evaluator import LossDatasetEvaluatorConfig
from olmo.nn.image_vit import VitConfig
from olmo.nn.llm import LlmConfig, AttentionType, LayerNormType, AttentionLayerNormType, RopeType
from olmo.models.molmo.molmo import MolmoConfig
from olmo.tokenizer import TokenizerConfig
from olmo.nn.vision_backbone import MolmoVisionBackboneConfig

log = logging.getLogger(__name__)


DEBUG_LLM = LlmConfig(
    d_model=128,
    n_heads=2,
    n_layers=3,
    max_sequence_length=4096,
    additional_vocab_size=128,
    vocab_size=152064,
    rope=True,
    embedding_size=None,
    weight_tying=False,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2-7B",
    )
)

DEBUG_VIT = VitConfig(
    image_num_layers=1,
    image_model_type="siglip",
    image_default_input_size=(378, 378),
    image_emb_dim=1152,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_head_dim=72,
    image_mlp_dim=4304,
    image_mlp_activations="gelu_pytorch_tanh",
    image_num_pos=729,  # no CLS token
    resize_mode="siglip",
    normalize="siglip"
)


DEBUG_MOLMO = MolmoConfig(
    llm=DEBUG_LLM,
    vision_backbone=MolmoVisionBackboneConfig(
        vit=DEBUG_VIT
    ),
    data_formatter=DataFormatter(),
    mm_preprocessor=MolmoPreprocessorConfig(crop_mode="resize", max_crops=1)
)


DEBUG_MOLMO2 = Molmo2Config(
    llm=DEBUG_LLM,
    vision_backbone=MolmoVisionBackboneConfig(
        vit=DEBUG_VIT
    ),
    data_formatter=DEBUG_MOLMO.data_formatter,
    mm_preprocessor=Molmo2PreprocessorConfig(
        pooling_h=3,
        pooling_w=3,
        max_frames=4,
        max_crops=1
    )
)


DEBUG_VISION_BACKBONE = VitConfig(
    init_path=None,
    resize_mode="siglip",
    image_model_type="openai",
    image_default_input_size=(378, 378),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=128,
    image_num_heads=2,
    image_num_key_value_heads=2,
    image_num_layers=2,
    image_head_dim=64,
    image_mlp_dim=256,
    image_mlp_activations="quick_gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-5,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
)


DEFAULT_VISION_BACKBONE = VitConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/vit-l-14-336.pt",
    image_model_type="openai",
    image_default_input_size=(336, 336),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1024,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=23,
    image_head_dim=64,
    image_mlp_dim=4096,
    image_mlp_activations="quick_gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-5,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
)


SIGLIP_VISION_BACKBONE = VitConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/siglip-so400m-14-384.pt",
    image_model_type="siglip",
    image_default_input_size=(378, 378),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1152,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=27,
    image_head_dim=72,
    image_mlp_dim=4304,
    image_mlp_activations="gelu_pytorch_tanh",
    image_dropout_rate=0.0,
    image_num_pos=729, # no CLS token
    image_norm_eps=1e-6,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="siglip",
    normalize="siglip"
)


SIGLIP2_VISION_BACKBONE = replace(
    SIGLIP_VISION_BACKBONE,
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/siglip2-so400m-14-384.pt",
)


SIGLIP2_BASE_PATCH16_224_VISION_BACKBONE = VitConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/siglip2-base-patch16-224.pt",
    image_model_type="siglip",
    image_default_input_size=(224, 224),
    image_patch_size=16,
    image_pos_patch_size=16,
    image_emb_dim=768,
    image_num_heads=12,
    image_num_key_value_heads=12,
    image_num_layers=12,
    image_head_dim=64,
    image_mlp_dim=3072,
    image_mlp_activations="gelu_pytorch_tanh",
    image_dropout_rate=0.0,
    image_num_pos=196, # no CLS token
    image_norm_eps=1e-6,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="siglip",
    normalize="siglip"
)


DINOV2_LARGE_336_VISION_BACKBONE = VitConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/dinov2-large-336.pt",
    image_model_type="dino",
    image_default_input_size=(336, 336),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1024,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=24,
    image_head_dim=64,
    image_mlp_dim=4096,
    image_mlp_activations="gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-6,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="dino",
    normalize="dino",
)


METACLIP_L14_336_VISION_BACKBONE = VitConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/metaclip-l14-336.pt",
    image_model_type="openai",
    image_default_input_size=(336, 336),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1024,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=24,
    image_head_dim=64,
    image_mlp_dim=4096,
    image_mlp_activations="quick_gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-5,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="metaclip",
)


METACLIP_B16_224_VISION_BACKBONE = VitConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/metaclip-b16-224.pt",
    image_model_type="openai",
    image_default_input_size=(224, 224),
    image_patch_size=16,
    image_pos_patch_size=16,
    image_emb_dim=768,
    image_num_heads=12,
    image_num_key_value_heads=12,
    image_num_layers=12,
    image_head_dim=64,
    image_mlp_dim=3072,
    image_mlp_activations="quick_gelu",
    image_dropout_rate=0.0,
    image_num_pos=197,
    image_norm_eps=1e-5,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="metaclip",
)


METACLIP_400M_B16_224_VISION_BACKBONE = replace(
    METACLIP_B16_224_VISION_BACKBONE,
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/metaclip-400m-b16-224.pt",
)



OLMOE = LlmConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/olmoe.pt",
    d_model=2048,
    n_heads=16,
    n_layers=16,
    mlp_ratio=1,
    activation_type='swiglu',
    block_type='moe',
    rope=True,
    rope_full_precision=True,
    rope_theta=10000.0,
    attention_type='sdpa',
    attention_layer_norm=True,
    residual_dropout=0.1,
    response_residual_dropout=0.0,
    embedding_dropout=0.0,
    layer_norm_type='rms',
    layer_norm_with_affine=True,
    layer_norm_eps=1e-05,
    attention_layer_norm_with_affine=True,
    max_sequence_length=4096,
    max_position_embeddings=32768,
    include_bias=False,
    bias_for_layer_norm=False,
    scale_logits=False,
    vocab_size=50280,
    embedding_size=50304,
    additional_vocab_size=128,
    new_embedding_init_range=0.02,
    weight_tying=False,
    normalize_input_embeds=False,
    use_position_ids=True,

    # MOE parameters
    moe_num_experts=64,
    moe_top_k=8,
    moe_mlp_impl='sparse',
    moe_log_expert_assignment=False,
    moe_shared_expert=False,
    moe_lbl_in_fp32=False,
    moe_interleave=False,
    moe_loss_weight=0.0,
    moe_zloss_weight=0.0,
    moe_dropless=True,
    moe_capacity_factor=1.25,

    tokenizer=TokenizerConfig(
        identifier='allenai/OLMoE-1B-7B-0924',
    ),
    fix_pad_tokenizer=True,
)


OLMO_1024_PREVIEW = LlmConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/olmo-1024-preview.pt",
    d_model=4096,
    n_heads=32,
    n_kv_heads=None,
    clip_qkv=None,
    n_layers=32,
    mlp_ratio=4,
    mlp_hidden_size=22016,
    activation_type="swiglu",
    block_type="sequential",
    rope=True,
    rope_full_precision=True,
    rope_theta=500000,
    attention_dropout=0.0,
    attention_layer_norm=True,
    layer_norm_type="rms",
    layer_norm_with_affine=True,
    layer_norm_eps=1.0e-06,
    attention_layer_norm_with_affine=True,
    max_sequence_length=4096,
    include_bias=False,
    bias_for_layer_norm=False,
    scale_logits=False,
    vocab_size=100278,
    embedding_size=100352,
    additional_vocab_size=128,
    weight_tying=False,
    attention_type=AttentionType.sdpa,
    norm_after=True,
    tokenizer=TokenizerConfig(
        identifier="allenai/dolma2-tokenizer",
    ),
    embedding_dropout=0,
    fix_pad_tokenizer=True,
)


OLMO2_1124_7B = replace(
    OLMO_1024_PREVIEW,
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/olmo2-1124-7b.pt",
    tokenizer=TokenizerConfig(
        identifier="allenai/OLMo-2-1124-7B",
    ),
)


OLMO2_1124_13B = replace(
    OLMO_1024_PREVIEW,
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/olmo2-1124-13b.pt",
    d_model=5120,
    n_heads=40,
    n_layers=40,
    mlp_hidden_size=27648,
    tokenizer=TokenizerConfig(
        identifier="allenai/OLMo-2-1124-13B",
    ),
)


OLMO2_1124_13B_INSTRUCT = replace(
    OLMO2_1124_13B,
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/olmo2-1124-13b-instruct.pt",
    tokenizer=TokenizerConfig(
        identifier="allenai/OLMo-2-1124-13B-Instruct",
    ),
)


OLMO2_0325_32B = replace(
    OLMO_1024_PREVIEW,
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/olmo2-0325-32b.pt",
    d_model=5120,
    n_heads=40,
    n_kv_heads=8,
    n_layers=64,
    mlp_hidden_size=55296,
    tokenizer=TokenizerConfig(
        identifier="allenai/OLMo-2-0325-32B",
    ),
)


OLMO2_0325_32B_INSTRUCT = replace(
    OLMO2_0325_32B,
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/olmo2-0325-32b-instruct.pt",
    tokenizer=TokenizerConfig(
        identifier="allenai/OLMo-2-0325-32B-Instruct",
    ),
)


OLMO3_1025_7B = LlmConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/olmo3-1025-7b.pt",
    d_model=4096,
    n_heads=32,
    n_kv_heads=None,
    clip_qkv=None,
    n_layers=32,
    mlp_ratio=4,
    mlp_hidden_size=22016,
    activation_type="swiglu",
    block_type="sequential",
    rope=True,
    rope_full_precision=True,
    rope_theta=500000,
    rope_type=RopeType.yarn,
    rope_factor=8.0,
    rope_attention_factor=1.2079441541679836,
    rope_beta_fast=32,
    rope_beta_slow=1,
    rope_original_max_position_embeddings=8192,
    full_attention_layers=(3, 7, 11, 15, 19, 23, 27, 31),
    attention_dropout=0.0,
    attention_layer_norm=True,
    layer_norm_type="rms",
    layer_norm_with_affine=True,
    layer_norm_eps=1.0e-06,
    attention_layer_norm_with_affine=True,
    max_sequence_length=4096,
    include_bias=False,
    bias_for_layer_norm=False,
    scale_logits=False,
    vocab_size=100278,
    embedding_size=100278,
    additional_vocab_size=128,
    weight_tying=False,
    attention_type=AttentionType.sdpa,
    norm_after=True,
    tokenizer=TokenizerConfig(
        identifier="allenai/Olmo-3-1025-7B",
    ),
    embedding_dropout=0,
    fix_pad_tokenizer=True,
)


OLMO3_1025_7B_INSTRUCT_DPO = replace(
    OLMO3_1025_7B,
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/olmo3-1025-7b-instruct-dpo.pt",
    tokenizer=TokenizerConfig(
        identifier="allenai/Olmo-3-Instruct-DPO",
    ),
)


OLMO3_1025_7B_THINKING_DPO = replace(
    OLMO3_1025_7B,
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/olmo3-1025-7b-thinking-dpo.pt",
    tokenizer=TokenizerConfig(
        identifier="allenai/Olmo-3-Thinking-DPO",
    ),
)


OLMO3_7B_INSTRUCT = LlmConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/olmo3-7b-instruct.pt",
    d_model=4096,
    n_heads=32,
    n_kv_heads=None,
    clip_qkv=None,
    n_layers=32,
    mlp_ratio=4,
    mlp_hidden_size=22016,
    activation_type="swiglu",
    block_type="sequential",
    rope=True,
    rope_full_precision=True,
    rope_theta=500000,
    rope_type=RopeType.yarn,
    rope_factor=8.0,
    rope_attention_factor=1.2079441541679836,
    rope_beta_fast=32,
    rope_beta_slow=1,
    rope_original_max_position_embeddings=8192,
    full_attention_layers=(3, 7, 11, 15, 19, 23, 27, 31),
    attention_dropout=0.0,
    attention_layer_norm=True,
    layer_norm_type="rms",
    layer_norm_with_affine=True,
    layer_norm_eps=1.0e-06,
    attention_layer_norm_with_affine=True,
    max_sequence_length=4096,
    include_bias=False,
    bias_for_layer_norm=False,
    scale_logits=False,
    vocab_size=100278,
    embedding_size=100278,
    additional_vocab_size=128,
    weight_tying=False,
    attention_type=AttentionType.sdpa,
    norm_after=True,
    tokenizer=TokenizerConfig(
        identifier="allenai/Olmo-3-7B-Instruct",
    ),
    embedding_dropout=0,
    fix_pad_tokenizer=True,
)


QWEN2_7B = LlmConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen2-7b.pt",
    vocab_size=152064,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=True,
    weight_tying=False,
    include_bias=False,
    embedding_size=152064,
    d_model=3584,
    mlp_hidden_size=18944*2,
    n_layers=28,
    additional_vocab_size=128,
    n_heads=28,
    n_kv_heads=4,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2-7B",
    ),
)


QWEN25_15B = LlmConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen2.5-1.5b.pt",
    vocab_size=151936,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=True,
    weight_tying=True,
    include_bias=False,
    embedding_size=151936,
    d_model=1536,
    mlp_hidden_size=8960*2,
    n_layers=28,
    additional_vocab_size=128,
    n_heads=12,
    n_kv_heads=2,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2.5-1.5B",
    ),
)


QWEN25_3B = LlmConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen2.5-3b.pt",
    vocab_size=151936,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=True,
    weight_tying=True,
    include_bias=False,
    embedding_size=151936,
    d_model=2048,
    mlp_hidden_size=11008*2,
    n_layers=36,
    additional_vocab_size=128,
    n_heads=16,
    n_kv_heads=2,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2.5-3B",
    ),
)


QWEN25_7B = LlmConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen2.5-7b.pt",
    vocab_size=152064,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=True,
    weight_tying=False,
    include_bias=False,
    embedding_size=152064,
    d_model=3584,
    mlp_hidden_size=18944*2,
    n_layers=28,
    additional_vocab_size=128,
    n_heads=28,
    n_kv_heads=4,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2.5-7B",
    ),
)


QWEN25_14B = LlmConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen2.5-14b.pt",
    vocab_size=152064,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=True,
    weight_tying=False,
    include_bias=False,
    embedding_size=152064,
    d_model=5120,
    mlp_hidden_size=13824*2,
    n_layers=48,
    additional_vocab_size=128,
    n_heads=40,
    n_kv_heads=8,
    rope_theta=1000000.0,
    layer_norm_eps=1e-5,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2.5-14B",
    ),
)


QWEN25_14B_INSTRUCT = replace(
    QWEN25_14B,
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen2.5-14b-instruct.pt",
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2.5-14B-Instruct",
    ),
    layer_norm_eps=1e-6,
    # The only difference is the layer norm eps
    # and the tokenizer identifier
)


QWEN2_72B = LlmConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen2-70b.pt",
    additional_vocab_size=128,
    vocab_size=152064,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    rope=True,
    qkv_bias=True,
    weight_tying=False,
    include_bias=False,
    embedding_size=152064,
    d_model=8192,
    mlp_hidden_size=29568*2,
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,
    rope_theta=1000000.0,
    layer_norm_eps=1e-5,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2-72B",
    ),
)


OMLO_19_13B = LlmConfig(
    d_model=5120,
    n_heads=40,
    n_kv_heads=None,
    clip_qkv=None,
    n_layers=40,
    mlp_ratio=4,
    mlp_hidden_size=27648,
    activation_type="swiglu",
    block_type="sequential",
    rope=True,
    rope_full_precision=True,
    rope_theta=500000,
    attention_dropout=0.0,
    attention_layer_norm=True,
    layer_norm_type="rms",
    layer_norm_with_affine=True,
    layer_norm_eps=1.0e-06,
    attention_layer_norm_with_affine=True,
    max_sequence_length=4096,
    include_bias=False,
    bias_for_layer_norm=False,
    scale_logits=False,
    vocab_size=100278,
    embedding_size=100352,
    weight_tying=False,
    attention_type=AttentionType.sdpa,
    init_fn="normal",
    init_std=0.02,
    init_cutoff_factor=3.0,
    norm_after=True,
    tokenizer=TokenizerConfig(
        identifier="allenai/dolma2-tokenizer",
    ),
    embedding_dropout=0,
    fix_pad_tokenizer=True,
)


LLAMA31_TULU31_8B = LlmConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/llama3.1-tulu3.1-8b.pt",
    d_model=4096,
    n_heads=32,
    n_kv_heads=8,
    qkv_bias=False,
    n_layers=32,
    mlp_hidden_size=14336*2,
    block_type="llama",
    rope=True,
    rope_theta=500000.0,
    rope_type=RopeType.llama3,
    rope_factor=8.0,
    rope_high_freq_factor=4.0,
    rope_low_freq_factor=1.0,
    rope_original_max_position_embeddings=8192,
    attention_dropout=0,
    residual_dropout=0,
    response_residual_dropout=0,
    layer_norm_type=LayerNormType.rms,
    layer_norm_eps=1e-5,
    max_sequence_length=4096,
    include_bias=False,
    embedding_dropout=0,
    vocab_size=128384, # multiple of 128
    additional_vocab_size=128,
    weight_tying=False,
    embedding_size=128384,
    tokenizer=TokenizerConfig(
        identifier="allenai/Llama-3.1-Tulu-3.1-8B",
    ),
)


QWEN3_06B = LlmConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen3-0.6b.pt",
    vocab_size=151936,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    attention_layer_norm=True,
    attention_layer_norm_type=AttentionLayerNormType.qwen3,
    rope=True,
    qkv_bias=False,
    weight_tying=True,
    include_bias=False,
    embedding_size=151936,
    d_model=1024,
    mlp_hidden_size=3072*2,
    n_layers=28,
    additional_vocab_size=128,
    n_heads=16,
    n_kv_heads=8,
    head_dim=128,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen3-4B",
    ),
)


QWEN3_4B = LlmConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen3-4b.pt",
    vocab_size=151936,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    attention_layer_norm=True,
    attention_layer_norm_type=AttentionLayerNormType.qwen3,
    rope=True,
    qkv_bias=False,
    weight_tying=True,
    include_bias=False,
    embedding_size=151936,
    d_model=2560,
    mlp_hidden_size=9728*2,
    n_layers=36,
    additional_vocab_size=128,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen3-4B",
    ),
)


QWEN3_4B_INSTRUCT = LlmConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen3-4b-instruct.pt",
    vocab_size=151936,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    attention_layer_norm=True,
    attention_layer_norm_type=AttentionLayerNormType.qwen3,
    rope=True,
    qkv_bias=False,
    weight_tying=True,
    include_bias=False,
    embedding_size=151936,
    d_model=2560,
    mlp_hidden_size=9728*2,
    n_layers=36,
    additional_vocab_size=128,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    rope_theta=5000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen3-4B-Instruct-2507",
    ),
)


QWEN3_8B_BASE = LlmConfig(
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen3-8b-base.pt",
    vocab_size=151936,
    max_sequence_length=4096,
    residual_dropout=0,
    embedding_dropout=0,
    response_residual_dropout=0,
    attention_dropout=0,
    attention_layer_norm=True,
    attention_layer_norm_type=AttentionLayerNormType.qwen3,
    rope=True,
    qkv_bias=False,
    weight_tying=False,
    include_bias=False,
    embedding_size=151936,
    d_model=4096,
    mlp_hidden_size=12288*2,
    n_layers=36,
    additional_vocab_size=128,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    rope_theta=1000000.0,
    layer_norm_eps=1e-6,
    layer_norm_type=LayerNormType.rms,
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen3-8B-Base",
    ),
)


QWEN3_8B = replace(
    QWEN3_8B_BASE,
    init_path="${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen3-8b.pt",
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen3-8B",
    ),
)


DEFAULT_LOAD_PATHS = {
    "openai": "${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/vit-l-14-336.pt",
    "siglip": "${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/siglip-so400m-14-384.pt",
    "dinov2_large_336": "${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/dinov2-large-336.pt",
    "metaclip_l14_336": "${oc.env:MOLMO_DATA_DIR}/pretrained_image_encoders/metaclip-l14-336.pt",
    "olmoe": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/olmoe.pt",
    "olmo_1024_preview": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/olmo-1024-preview.pt",
    "qwen2.5_1.5b": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen2.5-1.5b.pt",
    "qwen2.5_3b": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen2.5-3b.pt",
    "qwen2_7b": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen2-7b.pt",
    "qwen2_72b": "${oc.env:MOLMO_DATA_DIR}/pretrained_llms/qwen2-70b.pt",
}


VISION_BACKBONES: Dict[str, VitConfig] = {
    "debug": DEBUG_VISION_BACKBONE,
    "openai": DEFAULT_VISION_BACKBONE,
    "siglip": SIGLIP_VISION_BACKBONE,
    "siglip2": SIGLIP2_VISION_BACKBONE,
    "siglip2_base_patch16_224": SIGLIP2_BASE_PATCH16_224_VISION_BACKBONE,
    "dinov2_large_336": DINOV2_LARGE_336_VISION_BACKBONE,
    "metaclip_l14_336": METACLIP_L14_336_VISION_BACKBONE,
    "metaclip_b16_224": METACLIP_B16_224_VISION_BACKBONE,
    "metaclip_400m_b16_224": METACLIP_400M_B16_224_VISION_BACKBONE,
}


LLMS: Dict[str, LlmConfig] = {
    "olmoe": OLMOE,
    "olmo_1024_preview": OLMO_1024_PREVIEW,
    "olmo2_1124_7b": OLMO2_1124_7B,
    "olmo2_1124_13b": OLMO2_1124_13B,
    "olmo2_1124_13b_instruct": OLMO2_1124_13B_INSTRUCT,
    "olmo2_0325_32b": OLMO2_0325_32B,
    "olmo2_0325_32b_instruct": OLMO2_0325_32B_INSTRUCT,
    "olmo3_1025_7b": OLMO3_1025_7B,
    "olmo3_1025_7b_instruct_dpo": OLMO3_1025_7B_INSTRUCT_DPO,
    "olmo3_1025_7b_thinking_dpo": OLMO3_1025_7B_THINKING_DPO,
    "olmo3_7b_instruct": OLMO3_7B_INSTRUCT,
    "qwen2_7b": QWEN2_7B,
    "qwen2_72b": QWEN2_72B,
    "qwen2.5_14b_instruct": QWEN25_14B_INSTRUCT,
    "qwen2.5_14b": QWEN25_14B,
    "qwen2.5_7b": QWEN25_7B,
    "qwen2.5_3b": QWEN25_3B,
    "qwen2.5_1.5b": QWEN25_15B,
    "olmo1120_13b": OMLO_19_13B,
    "llama3.1_tulu3.1_8b": LLAMA31_TULU31_8B,
    "qwen3_8b_base": QWEN3_8B_BASE,
    "qwen3_8b": QWEN3_8B,
    "qwen3_4b": QWEN3_4B,
    "qwen3_4b_instruct": QWEN3_4B_INSTRUCT,
    "qwen3_0.6b": QWEN3_06B,
}
