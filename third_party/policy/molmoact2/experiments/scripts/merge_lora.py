import argparse
import os
import logging
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

import torch
import torch.distributed as dist

from olmo.train.checkpointer import (
    Checkpointer,
    MODEL_FILENAME,
    is_unsharded_checkoint,
    save_unsharded,
)
from olmo.train.distributed_checkpointing import (
    get_checkpoint_metadata,
    _load_unsharded_keys,
)
from olmo.util import (
    prepare_cli_environment,
    resource_path
)

from olmo.train.trainer_config import TrainConfig

from olmo.io import join_path

from peft import PeftModel


logger = logging.getLogger(__name__)

def _ensure_process_group() -> None:
    if not dist.is_available() or dist.is_initialized():
        return
    init_file = tempfile.NamedTemporaryFile(prefix="olmo_merge_lora_pg", delete=False)
    init_file.close()
    dist.init_process_group(
        backend="gloo",
        rank=0,
        world_size=1,
        init_method=f"file://{init_file.name}",
    )
    logger.info("Initialized single-process group for sharded checkpoint load.")


def _load_unsharded_state_dict(checkpoint_dir: str) -> dict:
    model_path = resource_path(checkpoint_dir, MODEL_FILENAME)
    return torch.load(model_path, map_location="cpu", weights_only=True)


def _load_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_dir: str,
    *,
    strict: bool = True,
) -> None:
    if is_unsharded_checkoint(checkpoint_dir):
        state_dict = _load_unsharded_state_dict(checkpoint_dir)
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if unexpected:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected[:10]}")
        if missing:
            logger.info(f"Missing keys when loading checkpoint: {missing[:10]}")
        return
    _ensure_process_group()
    Checkpointer().load(
        checkpoint_dir,
        model,
        optim=None,
        load_optimizer_state=False,
        load_trainer_state=False,
    )


def _merge_lora_into_module(module: torch.nn.Module, lora_dir: str, name: str) -> torch.nn.Module:
    if not os.path.exists(lora_dir):
        raise FileNotFoundError(f"Missing LoRA adapter directory for {name}: {lora_dir}")
    lora_model = PeftModel.from_pretrained(module, lora_dir)
    return lora_model.merge_and_unload()


def _extract_action_state(state_dict: dict) -> dict:
    candidate = state_dict
    if "model" in candidate and isinstance(candidate["model"], dict):
        candidate = candidate["model"]
    if "action_expert" in candidate and isinstance(candidate["action_expert"], dict):
        return candidate["action_expert"]
    action_state: dict = {}
    for key, value in candidate.items():
        if key.startswith("action_expert."):
            action_state[key[len("action_expert."):]] = value
        elif key.startswith("model.action_expert."):
            action_state[key[len("model.action_expert."):]] = value
    return action_state


def _load_action_expert_weights(model: torch.nn.Module, full_dir: str, model_config) -> None:
    if is_unsharded_checkoint(full_dir):
        state_dict = _load_unsharded_state_dict(full_dir)
        action_state = _extract_action_state(state_dict)
        if not action_state:
            logger.warning("No action_expert weights found in full checkpoint.")
            return
        missing, unexpected = model.load_state_dict(action_state, strict=False)
        if unexpected:
            logger.warning(f"Unexpected keys when loading action expert weights: {unexpected[:10]}")
        if missing:
            logger.info(f"Missing keys when loading action expert weights: {missing[:10]}")
        return

    metadata = get_checkpoint_metadata(join_path(full_dir, "model_and_optim"))
    keys = [
        key for key in metadata.state_dict_metadata.keys()
        if key.startswith("model.action_expert.")
    ]
    if not keys:
        logger.warning("No action_expert keys found in sharded checkpoint metadata.")
        return
    state_dict = _load_unsharded_keys(join_path(full_dir, "model_and_optim"), keys)
    action_state = _extract_action_state(state_dict)
    if not action_state:
        logger.warning("No action_expert weights found after loading sharded keys.")
        return
    missing, unexpected = model.action_expert.load_state_dict(action_state, strict=False)
    if unexpected:
        logger.warning(f"Unexpected keys when loading action expert weights: {unexpected[:10]}")
    if missing:
        logger.info(f"Missing keys when loading action expert weights: {missing[:10]}")

def convert_checkpoint(base_dir: str, full_dir: str, output_dir: str) -> None:
    full_dir_unsharded = f"{full_dir}-unsharded"
    full_dir_load = full_dir_unsharded if os.path.exists(full_dir_unsharded) else full_dir
    logger.info(f"Loading model config from {full_dir_load}")
    config_path = resource_path(full_dir_load, "config.yaml")
    config: TrainConfig = TrainConfig.load(config_path)
    model_config = config.model

    logger.info("Building model on CPU")
    with torch.device("meta"):
        model = model_config.build_model()
    model.to_empty(device=torch.device("cpu"))

    lora_dir = f"{full_dir}-lora"
    lora_dir_llm = lora_dir + "-llm"
    lora_dir_vision = lora_dir + "-vision"

    base_is_unsharded = is_unsharded_checkoint(base_dir)
    if base_is_unsharded:
        logger.info(f"Loading base VLM checkpoint from {base_dir}")
        _load_checkpoint_into_model(model, base_dir, strict=False)
        logger.info("Merging LoRA adapters into the VLM weights")
        model.transformer = _merge_lora_into_module(model.transformer, lora_dir_llm, "transformer")
        model.vision_backbone = _merge_lora_into_module(model.vision_backbone, lora_dir_vision, "vision_backbone")
    else:
        base_config_path = resource_path(base_dir, "config.yaml")
        if os.path.exists(base_config_path):
            base_config = TrainConfig.load(base_config_path)
            base_model_config = base_config.model
        else:
            logger.warning("Base config not found; falling back to main config for base weights.")
            base_model_config = model_config
        with torch.device("meta"):
            base_model = base_model_config.build_model()
        base_model.to_empty(device=torch.device("cpu"))
        logger.info(f"Loading base VLM checkpoint from {base_dir}")
        _load_checkpoint_into_model(base_model, base_dir, strict=True)
        logger.info("Merging LoRA adapters into the VLM weights")
        base_model.transformer = _merge_lora_into_module(base_model.transformer, lora_dir_llm, "transformer")
        base_model.vision_backbone = _merge_lora_into_module(base_model.vision_backbone, lora_dir_vision, "vision_backbone")
        model.transformer.load_state_dict(base_model.transformer.state_dict(), strict=True)
        model.vision_backbone.load_state_dict(base_model.vision_backbone.state_dict(), strict=True)

    logger.info(f"Loading action expert weights from {full_dir_load}")
    _load_action_expert_weights(model, full_dir_load, model_config)

    logger.info(f"Saving model config and merged checkpoint to {output_dir}")
    save_unsharded(output_dir, model, None, config, True)

    logger.info("Completed")


def main():
    parser = argparse.ArgumentParser(
        description="Adds a config.json to the checkpoint directory, creates pytorch_model.bin, and save the toeknizer,"
        "making it easier to load weights as HF models."
    )
    parser.add_argument(
        "--base_dir",
        help="Location of VLM-only base checkpoint (sharded or unsharded).",
    )
    parser.add_argument(
        "--full_dir",
        required=True,
        help="Location of full checkpoint (stepX). LoRA adapters must be in stepX-lora.",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to save the converted checkpoint.",
    )
    args = parser.parse_args()
    prepare_cli_environment()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"{args.full_dir}-merged"
        logger.info(f"Resolved output_dir to {output_dir}")
    convert_checkpoint(args.base_dir, args.full_dir, output_dir)


if __name__ == "__main__":
    main()
