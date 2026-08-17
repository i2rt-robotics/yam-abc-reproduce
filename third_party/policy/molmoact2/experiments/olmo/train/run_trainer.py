"""Run this script with 'torchrun'."""

import logging
import os
import signal
import socket
import sys
import time
from datetime import datetime
from os.path import join
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from beaker import Beaker
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from wandb.sdk.wandb_run import Run

from olmo.exceptions import OLMoCliError, OLMoConfigurationError
from olmo.io import file_exists, write_file, resource_path
from olmo.train.checkpointer import (
    Checkpointer,
    is_unsharded_checkoint,
)
from olmo.torch_util import (
    barrier,
    get_global_rank,
    get_local_rank,
    get_world_size,
    peak_gpu_memory,
    seed_all,
    freeze_module,
)
from olmo.dist_util import (
    parallelize_model,
    build_world_mesh
)
from olmo.token_layout import model_llm_token_layout
from olmo.train.remote_filesystem import RemoteFileSystemReader
from olmo.train.checkpoint_loading import load_unsharded_checkpoint_allowing_missing_action_expert
from olmo.train.peft_compat import peft_fsdp2_linear_shape_compat, validate_post_fsdp2_lora_shapes
from olmo.train.trainer import Trainer, BeakerLogger
from olmo.train.trainer_config import TrainConfig, RuntimeData, CheckpointLoadStrategy
from olmo.util import (
    clean_opt,
    is_hf_checkpoint_ref,
    log_extra_field,
    prepare_torchrun_environment,
)

log = logging.getLogger("train")


def log_model_info(fsdp_model, olmo_model):
    log.info("Model:")
    log.info(fsdp_model)
    log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
    log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")
    log.info(f"VLM number of parameters: {olmo_model.num_params_vlm():,d}")
    log.info(f"Action expert number of parameters: {olmo_model.num_params_action_expert():,d}")
    get_trainable_params(olmo_model)
    if olmo_model.config.llm.block_type == "moe":
        log.info(f"Number of active parameters: {olmo_model.num_params(include_inactive_params=False):,d}")
    log.info(f"Peak GPU Memory (MB) after FSDP: {int(peak_gpu_memory() or 0)}")


def get_trainable_params(model):
    """
    Calculate the size of a PyTorch model in bytes.
    """
    param_size = 0
    trainable_param_size = 0
    param_num = 0
    trainable_para_num = 0
    for param in model.parameters():
        param_num += param.nelement() 
        param_size += param.nelement() * param.element_size()
        trainable_para_num += param.nelement() if param.requires_grad else 0
        trainable_param_size += param.nelement() * param.element_size() if param.requires_grad else 0

    log.info(f'Number of trainable parameters: {trainable_para_num:,d}')


def _checkpoint_has_lora(checkpoint_dir: str | None) -> bool:
    if not checkpoint_dir:
        return False
    try:
        config_path = resource_path(checkpoint_dir, Checkpointer.CONFIG_FILENAME)
        raw_cfg = OmegaConf.load(config_path)
    except FileNotFoundError:
        return False
    except Exception as exc:
        log.warning("Failed to read checkpoint config from %s: %s", checkpoint_dir, exc)
        return False
    lora_enabled = OmegaConf.select(raw_cfg, "model.lora_enable")
    return bool(lora_enabled)


def _distributed_nonzero_count(param: torch.Tensor) -> int:
    tensor = param.detach()
    local_tensor = tensor.to_local() if hasattr(tensor, "to_local") else tensor
    nonzero = torch.count_nonzero(local_tensor).to(dtype=torch.long)
    if dist.is_initialized():
        backend = dist.get_backend()
        if backend == "nccl" and nonzero.device.type != "cuda":
            nonzero = nonzero.to(device=torch.device(f"cuda:{torch.cuda.current_device()}"))
        dist.all_reduce(nonzero, op=dist.ReduceOp.SUM)
    return int(nonzero.item())


def _distributed_nonfinite_count(param: torch.Tensor) -> int:
    tensor = param.detach()
    local_tensor = tensor.to_local() if hasattr(tensor, "to_local") else tensor
    nonfinite = torch.count_nonzero(~torch.isfinite(local_tensor)).to(dtype=torch.long)
    if dist.is_initialized():
        backend = dist.get_backend()
        if backend == "nccl" and nonfinite.device.type != "cuda":
            nonfinite = nonfinite.to(device=torch.device(f"cuda:{torch.cuda.current_device()}"))
        dist.all_reduce(nonfinite, op=dist.ReduceOp.SUM)
    return int(nonfinite.item())


def _assert_action_expert_initialized(model: torch.nn.Module) -> None:
    action_expert = getattr(model, "action_expert", None)
    if action_expert is None:
        return

    params_to_check: list[tuple[str, torch.Tensor]] = []
    time_embed = getattr(action_expert, "time_embed", None)
    if isinstance(time_embed, torch.nn.Sequential) and len(time_embed) > 3:
        if isinstance(time_embed[1], torch.nn.Linear):
            params_to_check.append(("action_expert.time_embed.1.weight", time_embed[1].weight))
        if isinstance(time_embed[3], torch.nn.Linear):
            params_to_check.append(("action_expert.time_embed.3.weight", time_embed[3].weight))

    final_layer = getattr(action_expert, "final_layer", None)
    final_ada_ln = getattr(final_layer, "adaLN", None)
    if (
        isinstance(final_ada_ln, torch.nn.Sequential)
        and len(final_ada_ln) > 1
        and isinstance(final_ada_ln[1], torch.nn.Linear)
    ):
        params_to_check.append(("action_expert.final_layer.adaLN.1.weight", final_ada_ln[1].weight))

    for attr in ("action_in_proj", "action_out_proj", "prefix_proj"):
        module = getattr(action_expert, attr, None)
        if isinstance(module, torch.nn.Linear):
            params_to_check.append((f"action_expert.{attr}.weight", module.weight))

    if not params_to_check:
        return

    nonfinite_params = [name for name, param in params_to_check if _distributed_nonfinite_count(param) > 0]
    if nonfinite_params:
        raise RuntimeError(
            "Action expert parameters contain non-finite values after reset: "
            f"{', '.join(nonfinite_params)}. "
            "This indicates a broken action expert initialization path."
        )

    zero_params = [name for name, param in params_to_check if _distributed_nonzero_count(param) == 0]
    if zero_params:
        raise RuntimeError(
            "Action expert parameters are all zero after reset: "
            f"{', '.join(zero_params)}. "
            "This indicates a broken action expert initialization path."
        )


def run_trainer(cfg: TrainConfig) -> None:
    if cfg.run_name is None:
        log_extra_field("run_name", cfg.run_name)

    # Additional environment setup
    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    device = torch.device("cuda")
    seed_all(cfg.seed)
    barrier()

    # Display the configuration.
    if get_global_rank() == 0:
        log.info("Configuration:")
        log.info(cfg)

    # Figure out what checkpoint we are starting from, if any
    start_from = None
    reset_opt, reset_train = False, False
    is_resuming = False
    if cfg.allow_resume:
        # Check if there is a checkpoint for us to resume from in our save folder, in which
        # case we ignore `cfg.load_from` and use it
        try:
            lastest_checkpoint = Checkpointer.latest_checkpoint(cfg.save_folder)
        except FileNotFoundError:
            lastest_checkpoint = None
        if lastest_checkpoint:
            log.info(f"Resuming from {lastest_checkpoint}")
            if get_global_rank() == 0:
                saved_config: TrainConfig = TrainConfig.load(join(cfg.save_folder, "config.yaml"))
                if saved_config.model != cfg.model:
                    log.warning("Model config does not match the one resuming from")
                if saved_config.optimizer != cfg.optimizer:
                    log.warning("Optimizer config does not match the one resuming from")
                if saved_config.data != cfg.data:
                    log.warning("Data config does not match the one resuming from")
            start_from = str(lastest_checkpoint)
            reset_opt, reset_train = False, False
            is_resuming = True
        else:
            log.info("Not resuming since no latest checkpoint found")

    if start_from is None and cfg.load_path:
        start_from = cfg.load_path
        reset_train, reset_opt = cfg.reset_trainer_state, cfg.reset_optimizer_state
    elif start_from is None and cfg.initial_model_checkpoint is not None:
        start_from = cfg.initial_model_checkpoint
        reset_train, reset_opt = True, True
    start_from_unsharded = start_from and is_unsharded_checkoint(start_from)
    start_from_hf = bool(start_from and is_hf_checkpoint_ref(start_from))
    start_from_model_only = bool(start_from_unsharded or start_from_hf)
    if start_from_model_only:
        assert reset_opt and reset_train, "Model-only checkpoints do not support optim/train state loading"
    if start_from_model_only and cfg.checkpoint_load_strategy == CheckpointLoadStrategy.distributed_sharded:
        raise OLMoConfigurationError(
            "checkpoint_load_strategy=distributed_sharded requires a sharded checkpoint, "
            f"but {start_from} is model-only."
        )
    if (
        start_from
        and not start_from_model_only
        and cfg.checkpoint_load_strategy == CheckpointLoadStrategy.rank0_broadcast
    ):
        raise OLMoConfigurationError(
            "checkpoint_load_strategy=rank0_broadcast only supports unsharded checkpoints, "
            f"but {start_from} is sharded."
        )
    checkpoint_has_lora = _checkpoint_has_lora(start_from)

    # Fail fast if we would be overwriting another save directory
    if not cfg.dry_run and not is_resuming and not cfg.save_overwrite:
        save_path = join(cfg.save_folder, "config.yaml")
        if file_exists(save_path):
            raise OLMoConfigurationError(f"{save_path} already exists, use --save_overwrite to overwrite")

    barrier()

    # Init the model
    model_cfg = cfg.model
    if checkpoint_has_lora and not model_cfg.lora_enable:
        log.warning("Checkpoint %s contains LoRA adapters; enabling LoRA to resume.", start_from)
        model_cfg.lora_enable = True
    lora_needs_init = bool(model_cfg.lora_enable and not checkpoint_has_lora)
    with torch.device("meta"):
        olmo_model = model_cfg.build_model()

    # Freeze parameters depending on what we are tuning
    if not cfg.ft_connector:
        log.info(f"Freezing connector")
        for param in olmo_model.get_connector_parameters():
            param.requires_grad = False
    if not cfg.ft_vit:
        log.info(f"Freezing vision backbone")
        for param in olmo_model.get_vit_parameters():
            param.requires_grad = False
    if not cfg.ft_llm:
        log.info(f"Freezing LLM")
        for param in olmo_model.get_llm_parameters():
            param.requires_grad = False
    action_format = str(getattr(model_cfg, "action_format", "continuous")).lower()
    state_format = str(getattr(model_cfg, "state_format", "continuous")).lower()
    if not cfg.ft_action_expert or action_format == "discrete":
        log.info(f"Freezing action expert")
        for param in olmo_model.get_action_expert_parameters():
            param.requires_grad = False
    elif state_format == "discrete" and getattr(olmo_model, "action_expert", None) is not None:
        log.info("Freezing action expert state encoder for discrete-only state conditioning")
        action_expert = olmo_model.action_expert
        if hasattr(action_expert, "freeze_continuous_state_conditioning"):
            action_expert.freeze_continuous_state_conditioning()
        elif hasattr(action_expert, "state_encoder") and hasattr(action_expert, "state_norm"):
            freeze_module(action_expert.state_encoder)
            freeze_module(action_expert.state_norm)
        else:
            log.info("Action expert has no continuous state-conditioning modules to freeze")
    if cfg.ft_embedding != "all":
        freeze_wte, freeze_out, freeze_ln_f = True, True, True
        tune_added_tokens_only = False
        if cfg.ft_embedding == "ln_f":
            freeze_ln_f = False
        elif cfg.ft_embedding == "lm_head":
            freeze_ln_f = False
            freeze_out = False
        elif cfg.ft_embedding == "wte":
            freeze_wte = False
        elif cfg.ft_embedding == "added_tokens":
            # Similar to lm_head fine-tuning, but only unmask gradients for added-token rows in wte.
            freeze_ln_f = False
            freeze_out = False
            freeze_wte = False
            tune_added_tokens_only = True
        elif cfg.ft_embedding == "none":
            pass
        else:
            raise NotImplementedError(cfg.ft_embedding)
        if freeze_ln_f:
            log.info(f"Freezing LLM: ln_f")
            freeze_module(olmo_model.transformer.ln_f)
        if freeze_out and hasattr(olmo_model.transformer, "ff_out"):
            log.info(f"Freezing LLM: ff_out")
            freeze_module(olmo_model.transformer.ff_out)
        if freeze_wte:
            log.info(f"Freezing LLM: wte")
            olmo_model.transformer.wte.embedding.requires_grad = False
        elif tune_added_tokens_only:
            layout = model_llm_token_layout(olmo_model)
            if layout.added_tokens <= 0:
                raise OLMoConfigurationError(
                    "ft_embedding='added_tokens' requires a positive number of added tokens."
                )

            added_start = layout.base_tokens
            added_end = layout.base_tokens + layout.added_tokens
            olmo_model.transformer.wte.embedding.requires_grad = True
            if hasattr(olmo_model.transformer.wte, "new_embedding"):
                # Added tokens live in wte.embedding, not in additional_vocab new_embedding rows.
                olmo_model.transformer.wte.new_embedding.requires_grad = False
                log.info("Freezing LLM: wte.new_embedding")

            def _mask_non_added_token_grads(
                grad: torch.Tensor,
                start: int = added_start,
                end: int = added_end,
            ) -> torch.Tensor:
                if grad is None:
                    return grad
                masked = grad.new_zeros(grad.shape)
                s = max(min(start, grad.shape[0]), 0)
                e = max(min(end, grad.shape[0]), s)
                masked[s:e] = grad[s:e]
                return masked

            olmo_model.transformer.wte.embedding.register_hook(_mask_non_added_token_grads)
            log.info("Tuning only added-token embeddings in wte rows [%d, %d)", added_start, added_end)

    def _initialize_lora_layers(model: torch.nn.Module) -> None:
        from peft.tuners.lora.layer import LoraLayer

        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            # Keep LoRA init identical across ranks so replicated adapters stay in sync.
            torch.manual_seed(cfg.seed)
            for module in model.modules():
                if isinstance(module, LoraLayer):
                    for adapter_name in module.lora_A.keys():
                        module.reset_lora_parameters(adapter_name, "gaussian")

    cp_enabled = cfg.parallelism.context_parallel_config.degree > 1

    # Do some other model setup
    if cfg.activation_checkpointing:
        olmo_model.apply_activation_checkpointing()
    # Stops the compiler get confused due to cache modifications
    olmo_model.warmup_cache(device, cp_enabled=cp_enabled)

    if cfg.compile:
        assert cfg.parallelism.context_parallel_config.degree == 1, "Model compilation is not supported with context parallelism yet."
        olmo_model.apply_compile(**cfg.compile.compile_args())

    def _inject_lora_modules(model: torch.nn.Module) -> None:
        import peft
        from peft import LoraConfig, get_peft_model

        compat_logged = False

        def _collect_linear_leaf_names(module: torch.nn.Module) -> list[str]:
            names: list[str] = []
            seen: set[str] = set()
            for name, sub in module.named_modules():
                if isinstance(sub, torch.nn.Linear):
                    leaf = name.rsplit(".", 1)[-1]
                    if leaf not in seen:
                        names.append(leaf)
                        seen.add(leaf)
            return names

        def _build_lora_config(target_modules: list[str]) -> LoraConfig:
            return LoraConfig(
                r=model_cfg.lora_rank,
                lora_alpha=model_cfg.lora_alpha,
                target_modules=target_modules,
                lora_dropout=model_cfg.lora_dropout,
                bias=model_cfg.lora_bias,
                init_lora_weights=False,
                fan_in_fan_out=False,
            )

        def _wrap_with_lora(
            base_module: torch.nn.Module,
            *,
            target_modules: list[str],
            label: str,
        ) -> torch.nn.Module:
            nonlocal compat_logged

            with peft_fsdp2_linear_shape_compat(enabled=cfg.fsdp.fsdp2) as compat_active:
                if compat_active and not compat_logged:
                    log.info(
                        "Applying PEFT %s FSDP2 LoRA linear shape compatibility patch during post-shard adapter injection.",
                        peft.__version__,
                    )
                    compat_logged = True
                wrapped_module = get_peft_model(base_module, _build_lora_config(target_modules))

            validate_post_fsdp2_lora_shapes(wrapped_module, label)
            return wrapped_module

        if not hasattr(model, "transformer"):
            raise RuntimeError("Model has no transformer; cannot inject LoRA adapters.")

        if not hasattr(model.transformer, "peft_config"):
            transformer_targets = _collect_linear_leaf_names(model.transformer)
            if transformer_targets:
                log.info("LoRA transformer target modules: %s", transformer_targets)
                model.transformer = _wrap_with_lora(
                    model.transformer,
                    target_modules=transformer_targets,
                    label="transformer",
                )
            else:
                log.warning("No transformer linear modules found; skipping LoRA injection for transformer.")
        else:
            log.info("Transformer already has LoRA adapters; skipping reinjection.")

        if getattr(model, "vision_backbone", None) is not None:
            if not hasattr(model.vision_backbone, "peft_config"):
                vision_targets = _collect_linear_leaf_names(model.vision_backbone)
                if vision_targets:
                    log.info("LoRA vision target modules: %s", vision_targets)
                    model.vision_backbone = _wrap_with_lora(
                        model.vision_backbone,
                        target_modules=vision_targets,
                        label="vision_backbone",
                    )
                else:
                    log.warning("No vision linear modules found; skipping LoRA injection for vision backbone.")
            else:
                log.info("Vision backbone already has LoRA adapters; skipping reinjection.")

    world_mesh = None
    if not cp_enabled:
        # Shard the model, and initialize if we are not loading a checkpoint
        if cfg.fsdp and not cfg.fsdp.fsdp2:
            raise NotImplementedError()

        elif cfg.fsdp.fsdp2:
            log.info("Wrapping model with FSDP2...")
            olmo_model.apply_fsdp2(**cfg.fsdp.get_fsd2_args(cfg.autocast_precision))
            olmo_model.to_empty(device=device)
            if start_from is None:
                olmo_model.reset_with_pretrained_weights()
            elif cfg.reset_parameters:
                olmo_model.reset_parameters()
            fsdp_model = olmo_model
        else:
            raise NotImplementedError()
    else:
        parallelism_config = cfg.parallelism

        world_mesh = build_world_mesh(
            tp=parallelism_config.tensor_parallel_config,
            cp=parallelism_config.context_parallel_config,
            dp=parallelism_config.data_parallel_config,
        )
        olmo_model = parallelize_model(
            olmo_model,
            world_mesh=world_mesh,
            float8_config=None,
            cp_config=parallelism_config.context_parallel_config,
            dp_config=parallelism_config.data_parallel_config,
            tp_config=parallelism_config.tensor_parallel_config,
        )
        olmo_model.to_empty(device=device)
        if start_from is None:
            olmo_model.reset_with_pretrained_weights()
        elif cfg.reset_parameters:
            olmo_model.reset_parameters()
    fsdp_model = olmo_model
    if (
        getattr(fsdp_model, "action_expert_depth_gate", None) is not None
        and hasattr(fsdp_model, "reset_action_expert_depth_gate_parameters")
    ):
        fsdp_model.reset_action_expert_depth_gate_parameters()
 
    torch.cuda.empty_cache()
    if not model_cfg.lora_enable:
        log_model_info(fsdp_model, olmo_model)

    if isinstance(fsdp_model, FSDP):
        settings = FSDP.get_state_dict_type(fsdp_model)
        if settings is None or settings.state_dict_type is None:
            FSDP.set_state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT)

    # Construct optimizer/scheduler/checkpointer
    optim = cfg.optimizer.build_optimizer(cfg.max_grad_norm, cfg.max_grad_norm_ratio, fsdp_model)
    scheduler = cfg.scheduler.build()
    checkpointer = cfg.checkpointer_config.build(cfg.save_overwrite)

    if not cfg.data.shuffle:
        raise ValueError("Using unshuffled data for training")
    if cfg.vlm_data is not None and not cfg.vlm_data.shuffle:
        raise ValueError("Using unshuffled data for VLM training")

    # Construct data loader and evaluators
    train_loader = cfg.data.build_train_dataloader(
        model_config=cfg.model,
        mesh=world_mesh,
        global_batch_size=cfg.global_train_batch_size,
    )
    if cfg.vlm_data is not None:
        if cfg.vlm_loader_rate is None:
            raise ValueError("cfg.vlm_loader_rate must be set when cfg.vlm_data is configured.")
        vlm_loader = cfg.vlm_data.build_train_dataloader(
            model_config=cfg.model,
            mesh=world_mesh,
            global_batch_size=cfg.global_train_batch_size,
        )
    else:
        vlm_loader = None
    if cfg.eval_interval > 0 or cfg.eval_on_load:
        evaluators = [v.build_dataset_evaluator(
            model_config=cfg.model, 
            mesh=world_mesh,
            device=device) for v in cfg.evaluators]
    else:
        evaluators = None
    if cfg.inf_eval_interval > 0 or cfg.eval_on_load:
        inf_evaluators = [v.build_dataset_evaluator(
            model_config=cfg.model, 
            mesh=None,  # disable mesh for inference as it's not supported
            default_save_dir=None, 
            device=device) for v in cfg.inf_evaluators]
    else:
        inf_evaluators = None

    # Maybe build the BeakerLogger
    if "BEAKER_EXPERIMENT_ID" in os.environ and "BEAKER_TOKEN" in os.environ:
        if get_global_rank() == 0:
            experiment_id = os.environ["BEAKER_EXPERIMENT_ID"]
            client = Beaker.from_env()
            beaker_logger = BeakerLogger(client, experiment_id, cfg.beaker_log_interval)
            beaker_logger.log_init()
        else:
            beaker_logger = None
    else:
        if cfg.beaker_log_interval > 0 and "BEAKER_EXPERIMENT_ID" in os.environ:
            logging.info(f"Beaker log interval set to {cfg.beaker_log_interval}, but beaker "
                         f"token is missing, so beaker logging will turned off")
        beaker_logger = None

    # Maybe start W&B run.
    if cfg.wandb is not None and (get_global_rank() == 0 or not cfg.wandb.rank_zero_only):
        wandb_dir = Path(cfg.save_folder) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_cfg = cfg.asdict(exclude=["wandb"])

        if "BEAKER_EXPERIMENT_ID" in os.environ:
            wandb_cfg["beaker_experiment_id"] = os.environ["BEAKER_EXPERIMENT_ID"]
            if beaker_logger is not None:
                wandb_cfg["beaker_url"] = beaker_logger.get_beaker_url()

        if is_resuming:
            wandb_cfg["resuming_from"] = start_from
        if is_resuming and cfg.wandb.allow_resume and saved_config.runtime_data.wandb_id:
            run_id = saved_config.runtime_data.wandb_id
            # Use standard run resume; avoid rewind-based resume_from which requires
            # project-side rewind support in W&B.
            resume_mode = "must"
            log.info(f"Resuming W&B run with id={run_id}")
        else:
            run_id = None
            resume_mode = None

        wandb.init(
            dir=str(wandb_dir),
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            config=wandb_cfg,
            id=run_id,
            resume=resume_mode,
            settings=wandb.Settings(init_timeout=180)
        )
        wandb_url = wandb.run.get_url()
        if beaker_logger is not None:
            beaker_logger.add_wandb(wandb_url)  # add wandb url to beaker description

        if cfg.wandb.finish_on_sigterm:
            # Try to make sure wandb will always finish cleanly if we get preempted
            # This is a bit of hack, but its useful since we can't use wandb.resume if wandb
            # did not finish cleanly

            def _signal_handler(signum, frame):
                # The dataloader workers might die and send us SIGCHLD which can interrupt this
                # method, so we ignore SIGCHLD here to avoid this
                # We are exiting anyway so its probably fine?
                signal.signal(signal.SIGCHLD, signal.SIG_IGN)
                log.warning(f"Getting {signum}, finish wandb then exiting")
                if wandb.run:
                    try:
                        wandb.finish(1)
                    except Exception as e:
                        log.warning(f"Unable to finish wandb {e}")
                exit(1)
            signal.signal(signal.SIGTERM, _signal_handler)

    # Fill in some runtime data so it will be recorded when we save the config
    cfg.runtime_data = RuntimeData(
        hostname=socket.gethostname(),
        date=datetime.now().strftime("%m/%d/%Y, %H:%M"),
        world_size=get_world_size(),
        beaker_experiment_id=os.environ.get("BEAKER_EXPERIMENT_ID"),
        beaker_experiment_url=(None if beaker_logger is None else
                               beaker_logger.get_beaker_url()),
        wandb_url=wandb.run.get_url() if wandb.run else None,
        wandb_id=wandb.run.id if wandb.run else None,
        args=" ".join(sys.argv),
        resuming_from=start_from if is_resuming else None,
    )

    # Save the config in a top-level file, note if we are resuming
    # the current config will still be saved next to new checkpoints
    if not cfg.dry_run and not is_resuming:
        if get_global_rank() == 0:
            write_file(cfg.save_folder, "config.yaml",
                       OmegaConf.to_yaml(cfg, resolve=True), cfg.save_overwrite)
    barrier()

    with Trainer(
        cfg=cfg,
        mesh=world_mesh,
        epoch=cfg.epoch,
        model=olmo_model,
        fsdp_model=fsdp_model,
        checkpointer=checkpointer,
        optim=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        vlm_loader=vlm_loader,
        device=device,
        evaluators=evaluators,
        inference_evaluators=inf_evaluators,
        beaker_logger=beaker_logger,
    ) as trainer:
        lora_injected = False
        if model_cfg.lora_enable and checkpoint_has_lora:
            # We need adapters present before restore when the checkpoint already contains LoRA weights.
            log.info("Injecting LoRA adapters before checkpoint restore (checkpoint has LoRA weights).")
            _inject_lora_modules(trainer.fsdp_model)
            trainer.optim = cfg.optimizer.build_optimizer(
                cfg.max_grad_norm, cfg.max_grad_norm_ratio, trainer.fsdp_model
            )
            trainer.manual_lora_grad_sync = get_world_size() > 1
            lora_injected = True
            log_model_info(trainer.fsdp_model, trainer.model)

        if start_from:
            # Load the starting checkpoint if there is one
            t0 = time.perf_counter()
            # NOTE: for now the action expert branch could only start from unsharded checkpoints since we want to support loading VLM-only checkpoints that don't have action expert weights, and the logic for allowing missing action expert weights is currently only implemented for unsharded checkpoints. We can add support for sharded checkpoints in the future if needed.
            if start_from_model_only:
                load_strategy = cfg.checkpoint_load_strategy
                if load_strategy == CheckpointLoadStrategy.auto:
                    load_strategy = CheckpointLoadStrategy.rank0_broadcast
                log.info(
                    "Loading model-only checkpoint from %s with checkpoint_load_strategy=%s",
                    start_from,
                    load_strategy,
                )
                has_action_expert = load_unsharded_checkpoint_allowing_missing_action_expert(
                    start_from, fsdp_model
                )
                if (
                    not has_action_expert
                    and hasattr(fsdp_model, "action_expert")
                    and hasattr(fsdp_model.action_expert, "reset_parameters")
                ):
                    log.info("Checkpoint lacks action expert weights; reinitializing action expert.")
                    fsdp_model.action_expert.reset_parameters()
                    _assert_action_expert_initialized(fsdp_model)
            else:
                if reset_train and reset_opt:
                    log.info(f"Loading model from {start_from}")
                elif not reset_opt and not reset_train:
                    log.info(f"Resuming from checkpoint {start_from}")
                else:
                    log.info(f"Restoring checkpoint {start_from}, but resetting "
                             f"{'Trainer' if reset_train else 'Optimizer'}")
                trainer.restore_checkpoint(
                    start_from,
                    load_optimizer_state=not reset_opt,
                    load_trainer_state=not reset_train,
                    allow_missing_keys=cfg.reset_parameters
                )
            log.info(f"Checkpoint successfully loaded in {time.perf_counter()-t0:0.1f} seconds")
            barrier()

        if model_cfg.lora_enable and not lora_injected:
            # For base checkpoints, inject adapters after restoring base weights to avoid missing-key mismatches.
            log.info("Injecting LoRA adapters after checkpoint restore (base checkpoint/no checkpoint).")
            _inject_lora_modules(trainer.fsdp_model)
            if lora_needs_init:
                log.info("Initializing LoRA parameters (no LoRA weights found in checkpoint).")
                _initialize_lora_layers(trainer.fsdp_model)
            trainer.optim = cfg.optimizer.build_optimizer(
                cfg.max_grad_norm, cfg.max_grad_norm_ratio, trainer.fsdp_model
            )
            trainer.manual_lora_grad_sync = get_world_size() > 1
            lora_injected = True
            log_model_info(trainer.fsdp_model, trainer.model)

        barrier()


        for name, param in trainer.fsdp_model.named_parameters():
            if not torch.all(torch.isfinite(param)):
                raise ValueError(name)

        # Ready to start training
        if not cfg.dry_run:
            log.info("Starting training...")
            trainer.fit()
            log.info("Training complete")
        else:
            log.info("Dry run complete")


if __name__ == "__main__":
    prepare_torchrun_environment()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    run_trainer(cfg)
