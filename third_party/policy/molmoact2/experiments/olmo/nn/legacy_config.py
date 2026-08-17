from typing import Dict

from omegaconf import omegaconf as om

from olmo.models.molmo.data_formatter import DataFormatter
from olmo.models.molmo.molmo_preprocessor import MolmoPreprocessorConfig
from olmo.nn.image_vit import VitConfig
from olmo.nn.llm import LlmConfig
from olmo.nn.vision_backbone import MolmoVisionBackboneConfig


IGNORE = object()
MISSING = object()


VISION_BACKBONE_REMOVED = dict(
    fsdp_wrap=IGNORE
)


REMOVED = dict(
    block_group_size=1,
    alibi=False,
    alibi_bias_max=IGNORE,
    rope_impl="llama",
    low_cpu_fsdp=IGNORE,
    response_attention_dropout=0,
    attn_logit_softcapping=None,
    final_logit_softcapping=None,
    do_random_scale=False,
    fix_image_input_idx=2,
    unconditioned=False,
    use_cls_feature=False,
    pad_to=None,
    pad_value=0,
    gin_bindings=None,
    prompt_override=None,
    pad_token_id=IGNORE,
    bos_token_id=IGNORE,
    multi_query_attention=None,
    init_device=IGNORE,
    precision=IGNORE,
    pad_tokenizer=True,
    query_pre_attn_scalar=IGNORE
)


def convert_vision_backbone(config) -> MolmoVisionBackboneConfig:
    image_vit = {k: MISSING for k in VitConfig().asdict()}
    vision_backbone = {}
    for key, value in config.items():
        if key in VISION_BACKBONE_REMOVED:
            if VISION_BACKBONE_REMOVED[key] != IGNORE:
                assert ((value is None and VISION_BACKBONE_REMOVED[key] is None)
                        or (value == VISION_BACKBONE_REMOVED[key]))
        elif key in image_vit:
            image_vit[key] = value
        else:
            vision_backbone[key] = value
    vision_backbone["vit"] = {k: v for k, v in image_vit.items() if v is not MISSING}
    return {k: v for k, v in vision_backbone.items() if v is not MISSING}


def convert_legacy_config(config):
    """Converts old monolithic model configs into the new format

    This also include some old backwards-compatibility fixes that are no longer
    needed for new configs
    """
    config = dict(config)

    llm = {k: MISSING for k in LlmConfig().asdict()}
    preprocessor = {k: MISSING for k in MolmoPreprocessorConfig().asdict()}
    data_formater = {k: MISSING for k in DataFormatter().asdict()}
    image_vit_keys = {k: MISSING for k in VitConfig().asdict()}
    vision_backbone = {k: MISSING for k in MolmoVisionBackboneConfig().asdict()}
    llm_args = {}

    # Renamed
    data_formater["system_prompt"] = config.pop("system_prompt_kind")
    data_formater["prompt_templates"] = config.pop("prompt_type")
    data_formater["message_format"] = config.pop("message_formatting")
    llm["init_path"] = config.pop("llm_load_path")
    preprocessor["loss_token_weighting"] = config.pop("multi_annotation_weighting", None)

    # Moved to vision backbone
    vision_backbone: Dict = config.pop("vision_backbone")
    for key in ["image_padding_embed", "vit_layers", "image_pooling_h", "image_feature_dropout",
                "image_pooling_w", "image_pooling_2d", "image_projector"]:
        vision_backbone[key] = config.pop(key)
    vision_backbone["init_path"] = config.pop("vit_load_path")
    vision_backbone = convert_vision_backbone(vision_backbone)

    # Old version always loaded all layers
    vision_backbone["skip_unused_layers"] = False

    # Fix up tokenizer
    tok = config["tokenizer"]
    for k in ["truncate_direction", "olmo_bos_token_id", "olmo_eos_token_id"]:
        if k in tok:
            del tok[k]
    if "tokenizer_adds_space" in tok:
        assert not tok["tokenizer_adds_space"]
        del tok["tokenizer_adds_space"]
    identifier = tok["identifier"]
    if identifier[:3] == "mm:":
        identifier = identifier[3:]
    if identifier[:3] == "hf-":
        identifier = identifier[3:]
    tok["identifier"] = identifier

    # Re-assign key to correct sub-config
    for key, val in config.items():
        if key in llm:
            llm[key] = val
        elif key in preprocessor:
            preprocessor[key] = val
        elif key in data_formater:
            data_formater[key] = val
        elif key in REMOVED:
            if REMOVED[key] != IGNORE:
                if REMOVED[key] is None:
                    assert val is None
                else:
                    if val != REMOVED[key]:
                        raise ValueError(f"Expected {REMOVED[key]} for removed key {key} but got {val}")
        else:
            raise ValueError(f"Unknown key {key} in legacy config")

    llm = {k: v for k, v in llm.items() if v is not MISSING}
    preprocessor = {k: v for k, v in preprocessor.items() if v is not MISSING}
    preprocessor["legacy_image_mask"] = True
    formatter = {k: v for k, v in data_formater.items() if v is not MISSING}
    return om.DictConfig(dict(
        llm=llm,
        vision_backbone=vision_backbone,
        data_formatter=formatter,
        mm_preprocessor=preprocessor
    ))