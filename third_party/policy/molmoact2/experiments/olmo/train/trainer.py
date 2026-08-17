from __future__ import annotations

import cProfile
import dataclasses
import gc
import json
import logging
import math
import os
import random
import re
import signal
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field, replace
from datetime import timedelta
from os.path import join
from pathlib import Path
from pstats import SortKey
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from beaker import Beaker
from beaker.exceptions import BeakerError
from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions, \
    set_model_state_dict

from ..eval.evaluators import SavePredictions

try:
    from beaker.client import ExperimentClient
except ImportError:
    # for older versions of beaker py
    from beaker import Experiment as ExperimentClient

from packaging import version
from requests import RequestException
from torch import nn
from torch.nn.functional import l1_loss, mse_loss
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, \
    FullStateDictConfig
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from wandb.sdk.data_types.base_types.wb_value import WBValue

from .trainer_config import (
    SpeedMonitorConfig, CheckpointType,
    TrainConfig, BatchDivisor,
)
from .timer import TimerManager
from olmo.data.data_loader import DataLoaderConfig, KwargsMixture
from olmo.data.iterable_dataset_mixture import IterableDatasetMixture, WorkerState, \
    IterableDataMixtureCheckpoint
from olmo.eval.inf_evaluator import InfDatasetEvaluator
from olmo.eval.loss_evaluator import LossMetrics, LossDatasetEvaluator
from olmo.exceptions import OLMoConfigurationError
from olmo.extra_tokens import DEFAULT_NUM_DEPTH_TOKENS, DEPTH_END_TOKEN, DEPTH_START_TOKEN, DEPTH_TOKENS
from olmo.models.molmo.molmo import Molmo
from olmo.train.optim import Optimizer, Scheduler, SchedulerUnits
from olmo.torch_util import (
    barrier,
    gc_cuda,
    get_fs_local_rank,
    get_global_rank,
    get_world_size,
    move_to_device,
    peak_gpu_memory,
    synchronize_flag,
    synchronize_value, get_local_world_size, clip_grad_norm, save_debug_batch, )
from olmo.dist_util import get_dp_process_group
from olmo.io import PathOrStr, clear_directory, is_url, normalize_path
from olmo.train.checkpointer import Checkpointer, save_unsharded, merge_and_save_unsharded
from ..data.dynamic_packer import EXAMPLE_SUBSEGMENT_INCREMENT
from ..util import flatten_lists, format_timedelta


try:
    from megablocks.layers.moe import (
        batched_load_balancing_loss,
        clear_load_balancing_loss,
        get_load_balancing_loss,
    )
except ImportError:
    pass


log = logging.getLogger(__name__)

_ACTION_SUPERVISION_BATCH_KEYS = (
    "states",
    "actions",
    "action_horizon_is_pad",
    "action_is_pad",
    "action_dim_is_pad",
    "packed_batch_idx",
    "packed_example_ids",
    "packed_action_chunk_is_valid",
    "packed_num_chunks",
    "packed_action_chunk_cap",
    "packed_action_chunk_overflow",
)

_DEPTH_SUPERVISION_BATCH_KEYS = (
    "depth_updated_mask",
    "depth_buffer_codes",
    "packed_depth_updated_mask",
    "packed_depth_buffer_codes",
    "packed_depth_example_ids",
    "packed_num_depth_examples",
    "packed_depth_row_is_valid",
)

def build_data_mixture_checkpoint(
    worker_states: List[WorkerState],
    world_size: int,
    num_workers: int,
    next_worker_id: int,
) -> Optional[IterableDataMixtureCheckpoint]:
    expected_workers = world_size * num_workers
    if expected_workers <= 0:
        return None

    latest_by_worker: Dict[int, WorkerState] = {}
    for state in worker_states:
        current = latest_by_worker.get(state.worker_global_id)
        if current is None or current.version < state.version:
            latest_by_worker[state.worker_global_id] = state

    if len(latest_by_worker) != expected_workers:
        return None

    ordered_states = [latest_by_worker.get(worker_id) for worker_id in range(expected_workers)]
    if any(state is None for state in ordered_states):
        return None

    return IterableDataMixtureCheckpoint(  # type: ignore[arg-type]
        list(ordered_states),
        world_size,
        num_workers,
        next_worker_id=next_worker_id,
    )


def should_use_vlm_loader_for_step(seed: int, step: int, vlm_loader_rate: Optional[float]) -> bool:
    if vlm_loader_rate is None:
        return False

    rate = float(vlm_loader_rate)
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    if step < 0:
        raise ValueError(f"Expected non-negative step, found {step}")

    step_seed = (int(seed) + int(step) * 1_000_003) % (2**32)
    rng = np.random.RandomState(step_seed)
    return bool(rng.random_sample() < rate)


def strip_action_supervision_from_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    for key in _ACTION_SUPERVISION_BATCH_KEYS:
        batch.pop(key, None)
    for key in _DEPTH_SUPERVISION_BATCH_KEYS:
        batch.pop(key, None)
    return batch


def rename_vlm_train_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    renamed: Dict[str, Any] = {}
    for key, value in metrics.items():
        if key.startswith("train/"):
            metric_name = key.split("/", 1)[1]
            renamed[f"train/{metric_name}VLM"] = value
        else:
            renamed[key] = value
    return renamed


def lerobot_tag_sampling_rate_metrics(
    primary_mixture: List[KwargsMixture],
    *,
    vlm_mixture: Optional[List[KwargsMixture]] = None,
    vlm_loader_rate: Optional[float] = None,
) -> Dict[str, float]:
    tag_rates: Dict[str, float] = defaultdict(float)

    has_separate_vlm_loader = vlm_mixture is not None and vlm_loader_rate is not None
    primary_loader_weight = 1.0 - float(vlm_loader_rate) if has_separate_vlm_loader else 1.0
    vlm_loader_weight = float(vlm_loader_rate) if has_separate_vlm_loader else 0.0

    def accumulate(mixture: Optional[List[KwargsMixture]], loader_weight: float) -> None:
        if mixture is None or loader_weight <= 0.0:
            return
        for entry in mixture:
            if not entry.name or not entry.name.startswith("lerobot:"):
                continue
            bare_tag = entry.name.split(":", 1)[1]
            tag_rates[bare_tag] += 100.0 * loader_weight * float(entry.rate)

    accumulate(primary_mixture, primary_loader_weight)
    accumulate(vlm_mixture, vlm_loader_weight)

    return {
        f"sampling_rate/lerobot_tag/{tag_name}": rate
        for tag_name, rate in sorted(tag_rates.items(), key=lambda item: (-item[1], item[0]))
    }


@dataclass
class BeakerLogger:
    WANDB_REGEX = ".*( \(https://wandb.ai/.*\))$"
    beaker: Beaker
    experiment_id: str
    log_interval: int
    _workload: Any = None
    _original_description: str = None
    _is_v1 = None

    def __post_init__(self):
        self._is_v1 = hasattr(self.beaker.experiment, "get")
        if self._is_v1:
            self._workload = self.beaker.experiment.get(self.experiment_id)
            self._original_description = self._workload.description
        else:
            self._workload = self.beaker.workload.get(self.experiment_id)
            self._original_description = self._workload.experiment.description

    def get_beaker_url(self):
        if self._is_v1:
            return self.beaker.experiment.url(self._workload)
        else:
            return self.beaker.workload.url(self._workload)

    def _set_description(self, description):
        try:
            if self._is_v1:
                self.beaker.experiment.set_description(self._workload, description)
            else:
                self.beaker.workload.update(self._workload, description=description)
        except (RequestException, BeakerError) as e:
            log.warning(f"Failed to update Beaker experiment description: {e}")

    def log_init(self):
        self._set_description(f"[Init] " + self._original_description)

    def add_wandb(self, wandb_url):
        # If there is an old wandb url (such as if the run was preempted), remove it
        match = re.match(self.WANDB_REGEX, self._original_description)
        if match:
            log.info(f"Removing old wandb url {wandb_url}")
            self._original_description = self._original_description[:match.start(1)]

        self._original_description = self._original_description + " (" + wandb_url + ")"
        self._set_description(f"[Init] " + self._original_description)

    def log_progress(self, on_step, target_step, eta=None):
        if eta:
            self._set_description(f"[{100*on_step/target_step:04.1f}%; eta={eta}] " + self._original_description)
        else:
            self._set_description(f"[{100*on_step/target_step:04.1f}%] " + self._original_description)

    def log_evaluation(self, eval_name, on_step, target_step):
        self._set_description(f"[{100*on_step/target_step:04.1f}%, {eval_name}] " + self._original_description)

    def finish(self):
        self._set_description(f"[Done] " + self._original_description)


@dataclass
class BatchStatsMonitor:
    max_window_size: int = 20
    sync_nodes: bool = True
    _batch_stats: Deque[Dict[str, float]] = field(default_factory=lambda: deque([]))

    def log_batch(self, batch):
        input_ids = batch["input_ids"]
        non_masked = (input_ids >= 0).to(dtype=torch.float32)
        stats = {
            "batch/non_masked_tokens": non_masked.sum(-1).mean(),
            "batch/per_non_masked_tokens": non_masked.mean(),
            "batch/examples_truncated": non_masked[:, -1].mean(),
            "batch/per_non_masked_images": 1.0 - torch.all(batch["images"] == -1, -1).float().mean()
        }
        if "loss_masks" in batch:
            mask = (batch["loss_masks"] > 0).to(dtype=torch.float32)
            stats["batch/loss_tokens"] = mask.sum(-1).mean()
            stats["batch/per_loss_tokens"] = mask.mean()
        if "subsegment_ids" in batch:
            subsegment_ids = batch["subsegment_ids"]
            n_packed = (subsegment_ids.max(-1).values // EXAMPLE_SUBSEGMENT_INCREMENT).float().mean() + 1
            stats["batch/n_packed"] = n_packed
            n_segments = 0
            for ex_subsegment_ids in subsegment_ids:
                values = torch.unique(ex_subsegment_ids)
                # Count unique non-padding and non-image subsegments
                n_segments += ((values != -1) & (values % EXAMPLE_SUBSEGMENT_INCREMENT != 10000)).sum()
            stats["batch/n_segments"] = n_segments / len(subsegment_ids)
        else:
            stats["batch/n_packed"] = torch.ones((), device=input_ids.device)
            stats["batch/n_segments"] = torch.ones((), device=input_ids.device)
        packed_num_chunks = None
        if "packed_num_chunks" in batch:
            packed_num_chunks = batch["packed_num_chunks"].to(dtype=torch.float32)
            stats["batch/packed_action_chunks_mean_actual"] = packed_num_chunks.mean()
            stats["batch/packed_action_chunks_max_actual"] = packed_num_chunks.max()
        if packed_num_chunks is not None and "packed_action_chunk_cap" in batch:
            packed_action_chunk_cap = batch["packed_action_chunk_cap"].to(dtype=torch.float32)
            positive_cap = packed_action_chunk_cap > 0
            if positive_cap.any():
                utilization = packed_num_chunks / packed_action_chunk_cap.clamp_min(1).to(dtype=torch.float32)
                stats["batch/packed_action_chunk_cap_utilization"] = utilization.mean()
        if "packed_action_chunk_overflow" in batch:
            overflow = batch["packed_action_chunk_overflow"].to(dtype=torch.float32)
            stats["batch/packed_action_chunk_overflow_rate"] = overflow.mean()
            stats["batch/packed_action_chunk_overflow_flag"] = overflow.max()

        self._batch_stats.append(stats)
        if len(self._batch_stats) > self.max_window_size:
            self._batch_stats.popleft()

    def reset(self) -> None:
        self._batch_stats.clear()

    def check(self, device):
        stats = defaultdict(list)
        for batch in self._batch_stats:
            for k, v in batch.items():
                stats[k].append(v)

        out = {}
        for k, v in stats.items():
            v = torch.stack(v).mean()
            if self.sync_nodes:
                v = v.to(device)
                dist.all_reduce(v)
                v.div_(get_world_size())
            out[k] = v.item()
        return out


@dataclass
class SpeedMonitor:
    cfg: SpeedMonitorConfig
    global_total_tokens: int = 0
    stats: Deque[Tuple[float, int, int]] = field(default_factory=lambda: deque([]))

    def batch_start(self, global_total_tokens: int, device_batch_num_tokens: int, device_batch_num_loss_tokens: int, record: bool = True) -> None:
        self.global_total_tokens = global_total_tokens
        if record:
            if len(self.stats) >= self.cfg.window_size:
                self.stats.popleft()
            self.stats.append((
                time.monotonic(),
                device_batch_num_tokens,
                device_batch_num_loss_tokens
            ))

    def reset(self) -> None:
        self.stats.clear()

    def _window_stats(self) -> Optional[Tuple[float, int, int, int]]:
        if not self.stats:
            return None
        interval_seconds = max(time.monotonic() - self.stats[0][0], 1e-6)
        interval_batches = len(self.stats)
        interval_tokens = sum(x[1] for x in self.stats)
        interval_loss_tokens = sum(x[2] for x in self.stats)
        return interval_seconds, interval_batches, interval_tokens, interval_loss_tokens

    def batches_per_second(self) -> Optional[float]:
        window_stats = self._window_stats()
        if window_stats is None:
            return None
        interval_seconds, interval_batches, _, _ = window_stats
        return interval_batches / interval_seconds

    def check(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {"throughput/total_tokens": self.global_total_tokens}
        window_stats = self._window_stats()
        if window_stats is not None:
            interval_seconds, interval_batches, interval_tokens, interval_loss_tokens = window_stats
            metrics["throughput/device/loss_tokens_per_second"] = interval_loss_tokens / interval_seconds
            metrics["throughput/device/tokens_per_second"] = interval_tokens / interval_seconds
            metrics["throughput/device/batches_per_second"] = interval_batches / interval_seconds
        return metrics


@dataclass
class LRMonitor:
    optim: torch.optim.Optimizer

    def check(self) -> Dict[str, float]:
        group_lrs = {}
        for group in self.optim.param_groups:
            if group['group_name'] in group_lrs:
                assert group_lrs[group['group_name']] == group['lr']
            else:
                group_lrs[group['group_name']] = group['lr']
        return {f"optim/{name}_lr": lr for name, lr in group_lrs.items()}


def cross_entropy_loss(
    logits, labels, ignore_index: int = -100, reduction: str = "mean", compute_z_loss: bool = False, z_loss_scale: float = 1e-4,
):
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

    if not compute_z_loss:
        return loss, None

    z_squared = logits.logsumexp(-1).pow(2)
    if reduction == "mean":
        z_squared = (z_squared * (labels != ignore_index)).mean()
    elif reduction == "sum":
        z_squared = (z_squared * (labels != ignore_index)).sum()

    z_loss = z_loss_scale * z_squared

    return loss, z_loss


def _tensor_all_finite(tensor: Optional[torch.Tensor]) -> Optional[bool]:
    if tensor is None:
        return None
    return bool(torch.isfinite(tensor).all().item())


def _tensor_shape(tensor: Optional[torch.Tensor]) -> Optional[Tuple[int, ...]]:
    if tensor is None:
        return None
    return tuple(tensor.shape)


def _distributed_any_flag(flag: bool, device: torch.device) -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return flag

    reduce_device = device
    if dist.get_backend() == "nccl" and reduce_device.type != "cuda":
        reduce_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    flag_tensor = torch.tensor(int(flag), device=reduce_device, dtype=torch.int32)
    dist.all_reduce(flag_tensor, op=dist.ReduceOp.MAX)
    return bool(flag_tensor.item())


@dataclass
class Trainer:
    cfg: TrainConfig
    model: Molmo
    mesh: DeviceMesh
    fsdp_model: FSDP
    optim: Optimizer
    scheduler: Scheduler
    train_loader: DataLoader
    device: torch.device
    evaluators: List[LossDatasetEvaluator]
    inference_evaluators: List[InfDatasetEvaluator]
    checkpointer: Checkpointer
    vlm_loader: Optional[DataLoader] = None
    epoch: Optional[int] = None
    global_step: int = 0

    global_train_examples_seen_this_epoch: int = 0
    """Tracks the global number of training examples seen in the current epoch for the purpose of restoring
    the data loader position on restarts."""

    primary_train_examples_seen_this_epoch: int = 0
    """Tracks the primary dataloader position for restoring mixed primary/VLM training."""

    vlm_train_examples_seen_this_epoch: int = 0
    """Tracks the dedicated VLM dataloader position for restoring mixed primary/VLM training."""

    global_train_tokens_seen: int = 0
    """Tracks the global total number of tokens trained on."""

    checkpoints: List[Path] = field(default_factory=list)
    unsharded_checkpoints: List[Path] = field(default_factory=list)
    ephemeral_checkpoints: List[Path] = field(default_factory=list)
    lora_checkpoints: List[Path] = field(default_factory=list)
    merged_lora_checkpoints: List[Path] = field(default_factory=list)
    min_train_loss: float = float("inf")
    cur_train_loss: float = float("inf")
    loss_fn: Callable[..., torch.Tensor] = field(default_factory=lambda: cross_entropy_loss)  # type: ignore
    beaker_logger: BeakerLogger = None
    last_sharded_checkpoint_step: Optional[int] = None
    last_unsharded_checkpoint_step: Optional[int] = None
    _train_metrics: Any = None
    _start_time: float = 0.0
    _start_step: Optional[int] = None
    _train_start_time: Optional[float] = None
    _gc_init_state: bool = True
    _cancelled: bool = False
    _cancel_reason: Optional[str] = None
    _global_batch_size_average: List[float] = field(default_factory=list)
    manual_lora_grad_sync: bool = False
    lora_grad_sync_chunk_numel: int = 8_000_000
    _data_worker_states: Optional[Dict[int, WorkerState]] = field(default_factory=dict)
    _vlm_data_worker_states: Optional[Dict[int, WorkerState]] = field(default_factory=dict)
    _train_loader_iter: Any = None
    _vlm_loader_iter: Any = None
    _debug_tokenizer: Any = None
    _nonfinite_dump_keys: set[str] = field(default_factory=set)

    def __post_init__(self):        
        if self.cfg.enable_timing_logs:
            # Initialize timer manager for profiling every step of training. By default it's disabled.
            # The reason for this was to allow logging of detailed timing stats du. ing 
            # training as the jobs failed at random steps during training.
            self._timer_manager = TimerManager(
                synchronize=True,  # Synchronize CUDA for accurate GPU timings
                device=self.device,
                reduce_across_ranks=True,  # Average timings across all ranks
                enabled=self.cfg.enable_timing_logs  # Only enable when configured
            )

        # If save folder is a local directory, make sure we're using a shared filesystem.
        if not is_url(self.cfg.save_folder) and get_fs_local_rank() != get_global_rank():
            raise OLMoConfigurationError(
                "Checkpointing to a local directory requires a shared filesystem. "
                "If you do have a shared filesystem please set the env var 'OLMO_SHARED_FS=1' "
                "or set 'FS_LOCAL_RANK' to the global rank for each process."
            )
        
        dp_process_group = get_dp_process_group(self.mesh) if self.mesh is not None else None
        self.dp_world_size = get_world_size(dp_process_group) if dp_process_group is not None else get_world_size()
        self.cp_degree = get_world_size() // self.dp_world_size
        self.cp_enabled = self.cp_degree > 1

        self._train_metrics = LossMetrics(self.device, reduce_loss_metrics_manually=self.cp_enabled)
        self._enable_depth_reasoning = bool(getattr(self.cfg.model, "enable_depth_reasoning", False))
        self._depth_start_id: Optional[int] = None
        self._depth_end_id: Optional[int] = None
        self._depth_code_token_ids: Optional[torch.Tensor] = None
        self._depth_code_input_noise_rate = float(
            getattr(self.cfg.model, "depth_code_input_noise_rate", 0.0) or 0.0
        )
        if not 0.0 <= self._depth_code_input_noise_rate <= 1.0:
            raise OLMoConfigurationError(
                "model.depth_code_input_noise_rate must be in [0, 1], "
                f"got {self._depth_code_input_noise_rate}."
            )
        try:
            tokenizer = self.cfg.model.build_tokenizer()
            depth_start_ids = tokenizer.encode(DEPTH_START_TOKEN)
            depth_end_ids = tokenizer.encode(DEPTH_END_TOKEN)
            if len(depth_start_ids) == 1 and len(depth_end_ids) == 1:
                self._depth_start_id = int(depth_start_ids[0])
                self._depth_end_id = int(depth_end_ids[0])
                code_ids = []
                for idx in range(DEFAULT_NUM_DEPTH_TOKENS):
                    ids = tokenizer.encode(DEPTH_TOKENS.token_for_index(idx))
                    if len(ids) == 1:
                        code_ids.append(ids[0])
                if len(code_ids) == DEFAULT_NUM_DEPTH_TOKENS:
                    self._depth_code_token_ids = torch.tensor(code_ids, dtype=torch.long)
        except Exception as exc:
            log.debug("Failed to resolve depth token ids for selective loss masking: %s", exc)
        if (
            self._enable_depth_reasoning
            and self._depth_code_input_noise_rate > 0.0
            and self._depth_code_token_ids is None
        ):
            log.warning(
                "Depth code input noise was requested, but depth code token ids could not be resolved; "
                "noise will be disabled."
            )
        self._last_depth_vis: Optional[Dict[str, Any]] = None
        self._depth_vae = None
        if self._enable_depth_reasoning and self._depth_code_token_ids is not None and get_global_rank() == 0:
            try:
                self._depth_vae = self._load_depth_vae(device=torch.device("cpu"))
                log.info("Depth VAE loaded for visualization from the depth annotation pipeline.")
            except Exception as exc:
                log.warning("Failed to load depth VAE for visualization: %s", exc)

        if not self.cfg.distributed_eval:
            log.info("Setting up inference model...")
            with torch.device("cpu"):
                olmo_model = self.cfg.model.build_model()
                olmo_model.warmup_cache(self.device)
                self.model = olmo_model

        if self.evaluators:
            assert len(set(x.label for x in self.evaluators)) == len(self.evaluators), "non-unique eval labels"
        if self.inference_evaluators:
            assert len(set(x.label for x in self.inference_evaluators)) == len(self.inference_evaluators), "non-unique eval labels"

        if self.cfg.fused_loss:
            import flash_attn
            from flash_attn.ops.triton.cross_entropy import (  # type: ignore
                cross_entropy_loss,
            )

            # The `ignored_index` parameter of `cross_entropy_loss` was changed to `ignore_index` in v2.5.8 with commit https://github.com/Dao-AILab/flash-attention/commit/ec6d22143b5d375e253b2ebfc563b26a43f43684
            ce_loss_use_ignore_index_param = version.parse(flash_attn.__version__) >= version.parse("2.5.8")

            def fused_loss_fn(
                logits, labels, ignore_index: int = -100, reduction: str = "mean",
                compute_z_loss: bool = False, z_loss_scale=1
            ):
                if ce_loss_use_ignore_index_param:
                    ignore_index_kwarg = {"ignore_index": ignore_index}
                else:
                    ignore_index_kwarg = {"ignored_index": ignore_index}

                loss, z_loss = cross_entropy_loss(
                    logits,
                    labels,
                    label_smoothing=0.0,
                    logit_scale=1.0,
                    lse_square_scale=z_loss_scale if compute_z_loss else 0.0,
                    inplace_backward=False,
                    process_group=None,
                    **ignore_index_kwarg,
                )

                mask = labels != ignore_index

                if reduction == "mean":
                    loss = loss.sum() / mask.sum()
                elif reduction == "sum":
                    loss = loss.sum()
                else:
                    loss = loss

                if not compute_z_loss:
                    return loss, None

                if reduction == "mean":
                    z_loss = z_loss.sum() / mask.sum()
                elif reduction == "sum":
                    z_loss = z_loss.sum()
                else:
                    z_loss = z_loss

                return loss, z_loss

            self.loss_fn = fused_loss_fn

        if self.cfg.compile_loss:
            if torch.cuda.is_available():
                self._loss_fn = torch.compile(self.loss_fn, dynamic=self.cfg.response_logits_only)
            else:
                log.warning(
                    "compile_loss was set to True, but CUDA is not available. Compiling only works with CUDA. Ignoring."
                )

    def _looks_like_nonfinite_error(self, exc: Exception) -> bool:
        if not isinstance(exc, RuntimeError):
            return False
        message = str(exc)
        return (
            "Non-finite" in message
            or "NaN or Inf" in message
            or "NaN or Inf loss detected" in message
        )

    def _get_debug_tokenizer(self):
        if self._debug_tokenizer is not None:
            return self._debug_tokenizer
        try:
            self._debug_tokenizer = self.cfg.model.build_tokenizer()
        except Exception as exc:
            log.warning("Failed to build tokenizer for non-finite batch dump: %s", exc)
            self._debug_tokenizer = False
        return None if self._debug_tokenizer is False else self._debug_tokenizer

    def _debug_dump_root(self) -> Path:
        if self.cfg.save_folder and not is_url(self.cfg.save_folder):
            return Path(normalize_path(self.cfg.save_folder)) / "nonfinite_debug"
        return Path.cwd() / "nonfinite_debug"

    def _to_cpu_copy(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: self._to_cpu_copy(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_cpu_copy(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._to_cpu_copy(v) for v in value)
        return value

    def _parse_suspect_base_rows(self, reason: str, batch_size: Optional[int]) -> List[int]:
        suspect_rows: List[int] = []
        match = re.search(r"suspect_base_batch_rows=\[([^\]]*)\]", reason)
        if match is not None:
            suspect_rows = [
                int(part.strip())
                for part in match.group(1).split(",")
                if part.strip()
            ]
        if not suspect_rows:
            match = re.search(r"k_bad_batch_rows=\[([^\]]*)\]", reason)
            timestep_match = re.search(r"num_flow_timesteps=(\d+)", reason)
            if match is not None:
                expanded_rows = [
                    int(part.strip())
                    for part in match.group(1).split(",")
                    if part.strip()
                ]
                if timestep_match is not None:
                    num_flow_timesteps = max(1, int(timestep_match.group(1)))
                    suspect_rows = sorted({row // num_flow_timesteps for row in expanded_rows})
                else:
                    suspect_rows = expanded_rows
        if batch_size is None:
            return suspect_rows
        return [row for row in suspect_rows if 0 <= row < batch_size]

    def _decode_input_ids(self, input_ids: Optional[torch.Tensor]) -> Optional[str]:
        if input_ids is None:
            return None
        tokenizer = self._get_debug_tokenizer()
        if tokenizer is None:
            return None
        tokens = [int(t) for t in input_ids.detach().cpu().view(-1).tolist() if int(t) >= 0]
        try:
            return tokenizer.decode(tokens, truncate_at_eos=False)
        except Exception as exc:
            log.warning("Failed to decode input ids for non-finite batch dump: %s", exc)
            return None

    def _unnormalize_image_tensor_for_debug(self, image: torch.Tensor) -> torch.Tensor:
        image_cfg = getattr(getattr(self.cfg.model, "mm_preprocessor", None), "image", None)
        normalize_mode = getattr(image_cfg, "normalize", "siglip") if image_cfg is not None else "siglip"
        normalize_on_gpu = bool(getattr(image_cfg, "normalize_on_gpu", False)) if image_cfg is not None else False

        x = torch.nan_to_num(image.detach().cpu().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if normalize_on_gpu:
            if x.max().item() > 1.5 or x.dtype == torch.uint8:
                x = x / 255.0
        else:
            if normalize_mode == "siglip":
                x = (x + 1.0) / 2.0
            elif normalize_mode == "openai":
                mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), dtype=x.dtype)
                std = torch.tensor((0.26862954, 0.26130258, 0.27577711), dtype=x.dtype)
                x = x * std + mean
            elif normalize_mode == "dino":
                mean = torch.tensor((0.485, 0.456, 0.406), dtype=x.dtype)
                std = torch.tensor((0.229, 0.224, 0.225), dtype=x.dtype)
                x = x * std + mean
        return x.clamp(0.0, 1.0)

    def _tensor_to_debug_images(self, image_tensor: Optional[torch.Tensor]) -> List[np.ndarray]:
        if image_tensor is None:
            return []
        tensor = image_tensor.detach().cpu()
        if tensor.ndim < 3:
            return []
        if tensor.shape[-1] in {1, 3}:
            flat = tensor.reshape(-1, *tensor.shape[-3:])
            channel_last = True
        elif tensor.shape[-3] in {1, 3}:
            flat = tensor.reshape(-1, *tensor.shape[-3:])
            channel_last = False
        else:
            return []

        rendered: List[np.ndarray] = []
        for img in flat:
            if img.ndim != 3:
                continue
            hwc = img if channel_last else img.permute(1, 2, 0)
            if hwc.shape[-1] == 1:
                hwc = hwc.repeat(1, 1, 3)
            hwc = self._unnormalize_image_tensor_for_debug(hwc)
            rendered.append((hwc.numpy() * 255.0).round().astype(np.uint8))
        return rendered

    def _dump_nonfinite_micro_batch(
        self,
        batch: Dict[str, Any],
        *,
        reason: str,
        micro_batch_index: int,
    ) -> Optional[Path]:
        batch_size = None
        input_ids = batch.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            batch_size = int(input_ids.shape[0])
        dump_key = f"{self.global_step}:{get_global_rank()}:{micro_batch_index}:{hash(reason)}"
        if dump_key in self._nonfinite_dump_keys:
            return None
        self._nonfinite_dump_keys.add(dump_key)

        dump_root = self._debug_dump_root()
        dump_root.mkdir(parents=True, exist_ok=True)
        dump_dir = dump_root / f"step{self.global_step:08d}_rank{get_global_rank():03d}_mb{micro_batch_index:02d}_{int(time.time())}"
        dump_dir.mkdir(parents=True, exist_ok=True)

        cpu_batch = self._to_cpu_copy(batch)
        torch.save(
            {
                "batch": cpu_batch,
                "global_step": self.global_step,
                "micro_batch_index": micro_batch_index,
                "rank": get_global_rank(),
                "world_size": get_world_size(),
                "reason": reason,
            },
            dump_dir / "micro_batch.pt",
        )

        suspect_rows = self._parse_suspect_base_rows(reason, batch_size)
        metadata = {
            "global_step": self.global_step,
            "micro_batch_index": micro_batch_index,
            "rank": get_global_rank(),
            "world_size": get_world_size(),
            "reason": reason,
            "suspect_base_batch_rows": suspect_rows,
            "batch_keys": sorted(batch.keys()),
            "input_ids_shape": list(input_ids.shape) if isinstance(input_ids, torch.Tensor) else None,
        }
        (dump_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (dump_dir / "reason.txt").write_text(reason)

        if batch_size is None:
            log.warning("Saved non-finite micro-batch dump to %s", dump_dir)
            return dump_dir

        sample_rows = suspect_rows if suspect_rows else list(range(batch_size))
        for sample_idx in sample_rows:
            sample_dir = dump_dir / f"sample_{sample_idx:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_meta: Dict[str, Any] = {"sample_index": sample_idx, "suspect": sample_idx in suspect_rows}

            sample_input_ids = None
            if isinstance(input_ids, torch.Tensor):
                sample_input_ids = input_ids[sample_idx]
                torch.save(sample_input_ids.detach().cpu(), sample_dir / "input_ids.pt")
                sample_meta["input_token_count"] = int((sample_input_ids >= 0).sum().item())

            for key in [
                "attention_mask",
                "loss_masks",
                "labels",
                "subsegment_ids",
                "actions",
                "states",
                "action_horizon_is_pad",
                "action_dim_is_pad",
                "image_masks",
                "token_pooling",
                "low_res_token_pooling",
                "num_images",
                "multimodal_type",
                "num_image_starts",
            ]:
                value = batch.get(key)
                if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] > sample_idx:
                    torch.save(value[sample_idx].detach().cpu(), sample_dir / f"{key}.pt")

            images = batch.get("images")
            if isinstance(images, torch.Tensor) and images.ndim > 0 and images.shape[0] > sample_idx:
                sample_images = images[sample_idx]
                torch.save(sample_images.detach().cpu(), sample_dir / "images.pt")
                try:
                    from PIL import Image

                    for image_idx, image_arr in enumerate(self._tensor_to_debug_images(sample_images)):
                        Image.fromarray(image_arr).save(sample_dir / f"image_{image_idx:03d}.png")
                except Exception as exc:
                    log.warning("Failed to render debug images for non-finite batch dump: %s", exc)

            packed_batch_idx = batch.get("packed_batch_idx")
            if isinstance(packed_batch_idx, torch.Tensor):
                mask = packed_batch_idx == sample_idx
                sample_chunk_meta = {"num_action_chunks": int(mask.sum().item())}
                for key in [
                    "packed_batch_idx",
                    "packed_example_ids",
                    "packed_action_chunk_is_valid",
                    "packed_num_chunks",
                    "packed_action_chunk_cap",
                    "packed_action_chunk_overflow",
                ]:
                    value = batch.get(key)
                    if isinstance(value, torch.Tensor):
                        if value.shape[:1] == packed_batch_idx.shape[:1]:
                            torch.save(value[mask].detach().cpu(), sample_dir / f"{key}.pt")
                        elif value.ndim > 0 and value.shape[0] > sample_idx:
                            torch.save(value[sample_idx].detach().cpu(), sample_dir / f"{key}.pt")
                sample_meta["packed"] = sample_chunk_meta

            decoded_text = self._decode_input_ids(sample_input_ids)
            if decoded_text is not None:
                (sample_dir / "decoded_text.txt").write_text(decoded_text)
                sample_meta["decoded_text_preview"] = decoded_text[:512]

            (sample_dir / "metadata.json").write_text(json.dumps(sample_meta, indent=2))

        log.warning("Saved non-finite micro-batch dump to %s", dump_dir)
        return dump_dir

    @property
    def dataset(self) -> IterableDataset:
        return self.train_loader

    @property
    def primary_dataset(self) -> IterableDataset:
        return self.train_loader.dataset

    @property
    def vlm_dataset(self) -> Optional[IterableDataset]:
        return None if self.vlm_loader is None else self.vlm_loader.dataset

    @property
    def tokens_per_batch(self) -> int:
        return self.cfg.global_train_batch_size * self.cfg.model.max_sequence_length

    @property
    def batches_per_epoch(self) -> int:
        return self.dataset.total_size // self.cfg.global_train_batch_size

    @property
    def max_epochs(self) -> int:
        if isinstance(self.cfg.max_duration, str) and self.cfg.max_duration.endswith("ep"):
            return int(self.cfg.max_duration[:-2].strip())
        else:
            return 1

    @property
    def max_steps(self) -> int:
        if isinstance(self.cfg.max_duration, int):
            return self.cfg.max_duration
        elif isinstance(self.cfg.max_duration, str):
            if self.cfg.max_duration.endswith("T"):
                # convert to float *first* to handle scientific notation
                max_tokens = int(float(self.cfg.max_duration[:-1].strip()))
                tokens_remaining = max(max_tokens - self.global_train_tokens_seen, 0)
                steps_remaining = tokens_remaining // self.tokens_per_batch
                return self.global_step + steps_remaining
            elif self.cfg.max_duration.endswith("ep"):
                max_epochs = int(self.cfg.max_duration[:-2].strip())
                return max_epochs * self.batches_per_epoch
            else:
                # convert to float *first* to handle scientific notation
                return int(float(self.cfg.max_duration))
        else:
            raise TypeError(f"expected int or str for 'max_duration', found {type(self.cfg.max_duration)}")

    @property
    def max_tokens(self) -> int:
        if isinstance(self.cfg.max_duration, int):
            return (
                self.global_train_tokens_seen
                + max(self.cfg.max_duration - self.global_step, 0) * self.tokens_per_batch
            )
        elif isinstance(self.cfg.max_duration, str):
            if self.cfg.max_duration.endswith("T"):
                # convert to float *first* to handle scientific notation
                return int(float(self.cfg.max_duration[:-1].strip()))
            elif self.cfg.max_duration.endswith("ep"):
                max_epochs = int(self.cfg.max_duration[:-2].strip())
                return max_epochs * self.batches_per_epoch * self.tokens_per_batch
            else:
                # convert to float *first* to handle scientific notation
                return (
                    self.global_train_tokens_seen
                    + max(int(float(self.cfg.max_duration)) - self.global_step, 0) * self.tokens_per_batch
                )
        else:
            raise TypeError(f"expected int or str for 'max_duration', found {type(self.cfg.max_duration)}")

    @property
    def scheduler_current(self) -> int:
        if self.cfg.scheduler.units == SchedulerUnits.steps:
            return self.global_step
        elif self.cfg.scheduler.units == SchedulerUnits.tokens:
            return self.global_train_tokens_seen
        else:
            raise NotImplementedError(self.cfg.scheduler.units)

    @property
    def scheduler_max(self) -> int:
        if self.cfg.scheduler.units == SchedulerUnits.steps:
            return self.max_steps
        elif self.cfg.scheduler.units == SchedulerUnits.tokens:
            return self.max_tokens
        else:
            raise NotImplementedError(self.cfg.scheduler.units)

    def _build_loader_checkpoint(
        self,
        loader_cfg: Optional[DataLoaderConfig],
        worker_states: Optional[Dict[int, WorkerState]],
        examples_seen: int = 0,
    ) -> Optional[IterableDataMixtureCheckpoint]:
        if loader_cfg is None or loader_cfg.packing is None or not loader_cfg.packing.track_packing_state:
            return None

        gathered_worker_states = list((worker_states or {}).values())
        if get_world_size() > 1:
            states = [None] * get_world_size()
            dist.all_gather_object(states, gathered_worker_states)
            gathered_worker_states = flatten_lists(states)

        num_workers = loader_cfg.num_workers or 1
        loader_steps_seen = 0
        if self.cfg.global_train_batch_size > 0:
            loader_steps_seen = examples_seen // self.cfg.global_train_batch_size
        return build_data_mixture_checkpoint(
            gathered_worker_states,
            world_size=get_world_size(),
            num_workers=num_workers,
            next_worker_id=loader_steps_seen % num_workers,
        )

    def _restore_loader_state(
        self,
        loader: DataLoader,
        loader_cfg: DataLoaderConfig,
        checkpoint: Optional[IterableDataMixtureCheckpoint],
        examples_seen: int,
        label: str,
    ) -> None:
        dataset = loader.dataset
        assert isinstance(dataset, IterableDatasetMixture)

        if checkpoint is not None:
            log.info(f"{label} restoring from checkpoint...")
            dataset.resume_from = checkpoint
            if (
                checkpoint.world_size != get_world_size()
                or checkpoint.num_workers != (loader_cfg.num_workers or 1)
            ):
                log.warning(
                    "%s world size / worker count changed; future example order may differ.",
                    label,
                )
            return

        if examples_seen > 0:
            log.info(f"{label} will start at instance index {examples_seen:,d}")
            dataset.resume_from_index = examples_seen

    def _get_train_batch_from_loader(self, loader: DataLoader, *, use_vlm_loader: bool) -> Dict[str, Any]:
        iterator_attr = "_vlm_loader_iter" if use_vlm_loader else "_train_loader_iter"
        loader_iter = getattr(self, iterator_attr)
        if loader_iter is None:
            loader_iter = iter(loader)

        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        setattr(self, iterator_attr, loader_iter)
        return batch

    def _should_use_vlm_loader_for_next_step(self) -> bool:
        if self.vlm_loader is None:
            return False
        return should_use_vlm_loader_for_step(
            self.cfg.seed,
            self.global_step + 1,
            self.cfg.vlm_loader_rate,
        )

    def _record_loader_worker_states(self, batch: Dict[str, Any], *, use_vlm_loader: bool) -> None:
        if "data_worker_state" not in batch:
            return
        worker_states = self._vlm_data_worker_states if use_vlm_loader else self._data_worker_states
        assert worker_states is not None
        for state in batch.pop("data_worker_state"):
            if state.worker_global_id not in worker_states:
                worker_states[state.worker_global_id] = state
            else:
                cur_version = worker_states[state.worker_global_id].version
                if cur_version < state.version:
                    worker_states[state.worker_global_id] = state

    def trainer_state_dict(self) -> Dict[str, Any]:
        data_checkpoint = self._build_loader_checkpoint(
            loader_cfg=self.cfg.data,
            worker_states=self._data_worker_states,
            examples_seen=self.primary_train_examples_seen_this_epoch,
        )
        vlm_data_checkpoint = self._build_loader_checkpoint(
            loader_cfg=self.cfg.vlm_data,
            worker_states=self._vlm_data_worker_states,
            examples_seen=self.vlm_train_examples_seen_this_epoch,
        )

        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "global_train_examples_seen_this_epoch": self.global_train_examples_seen_this_epoch,
            "primary_train_examples_seen_this_epoch": self.primary_train_examples_seen_this_epoch,
            "vlm_train_examples_seen_this_epoch": self.vlm_train_examples_seen_this_epoch,
            "global_train_tokens_seen": self.global_train_tokens_seen,
            "world_size": get_world_size(),
            "num_workers": self.cfg.data.num_workers,
            "checkpoints": self.checkpoints,
            "unsharded_checkpoints": self.unsharded_checkpoints,
            "ephemeral_checkpoints": self.ephemeral_checkpoints,
            "lora_checkpoints": self.lora_checkpoints,
            "merged_lora_checkpoints": self.merged_lora_checkpoints,
            "data_checkpoint": data_checkpoint,
            "vlm_data_checkpoint": vlm_data_checkpoint,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
            },
        }

    def load_trainer_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Checkpoint paths.
        normalized_save_folder = normalize_path(self.cfg.save_folder)

        def _is_ours(filename):
            return normalize_path(filename).startswith(normalized_save_folder)

        self.checkpoints = [
            path for path in state_dict["checkpoints"] if _is_ours(path)]
        self.unsharded_checkpoints = [
            path for path in state_dict["unsharded_checkpoints"] if _is_ours(path)]
        self.ephemeral_checkpoints = [
            path for path in state_dict.get("ephemeral_checkpoints", []) if _is_ours(path)]
        self.lora_checkpoints = [
            path for path in state_dict.get("lora_checkpoints", []) if _is_ours(path)]
        self.merged_lora_checkpoints = [
            path for path in state_dict.get("merged_lora_checkpoints", []) if _is_ours(path)]

        # Dataset / dataloader position.
        checkpoint_epoch = state_dict.get("epoch", 0)
        self.global_step = state_dict["global_step"]
        self.global_train_examples_seen_this_epoch = state_dict["global_train_examples_seen_this_epoch"]
        self.primary_train_examples_seen_this_epoch = state_dict.get(
            "primary_train_examples_seen_this_epoch",
            self.global_train_examples_seen_this_epoch,
        )
        self.vlm_train_examples_seen_this_epoch = state_dict.get("vlm_train_examples_seen_this_epoch", 0)
        self.global_train_tokens_seen = state_dict["global_train_tokens_seen"]

        if not self.cfg.restore_dataloader:
            self.epoch = 0
            self.global_train_tokens_seen = 0
            self.global_train_examples_seen_this_epoch = 0
            self.primary_train_examples_seen_this_epoch = 0
            self.vlm_train_examples_seen_this_epoch = 0
        elif self.epoch is None:
            self.epoch = checkpoint_epoch
        elif checkpoint_epoch != self.epoch:
            log.info(f"Starting new epoch (epoch = {self.epoch})")
            self.global_train_examples_seen_this_epoch = 0
            self.primary_train_examples_seen_this_epoch = 0
            self.vlm_train_examples_seen_this_epoch = 0

        if self.cfg.fast_forward_batches:
            if self.cfg.vlm_data is not None:
                raise ValueError("fast_forward_batches is not supported with cfg.vlm_data")
            log.info(f"Fast-forwarding data loader by {self.cfg.fast_forward_batches:,d} steps")
            # Technically we don't "see" these batches that we fast-forward through, but we use
            # this variable to update the position of the dataset so we need to include them here.
            self.global_train_examples_seen_this_epoch += (
                self.cfg.fast_forward_batches * self.cfg.global_train_batch_size
            )
            self.primary_train_examples_seen_this_epoch = self.global_train_examples_seen_this_epoch
            # NOTE: on the other hand we don't add anything to 'self.global_train_tokens_seen' here because
            # that variable is meant to track the actual number of tokens trained on.

        self._restore_loader_state(
            loader=self.train_loader,
            loader_cfg=self.cfg.data,
            checkpoint=state_dict.get("data_checkpoint"),
            examples_seen=self.primary_train_examples_seen_this_epoch,
            label="Primary data loader",
        )
        if self.vlm_loader is not None and self.cfg.vlm_data is not None:
            self._restore_loader_state(
                loader=self.vlm_loader,
                loader_cfg=self.cfg.vlm_data,
                checkpoint=state_dict.get("vlm_data_checkpoint"),
                examples_seen=self.vlm_train_examples_seen_this_epoch,
                label="VLM data loader",
            )

        # RNG states.
        if "rng" in state_dict and state_dict.get("world_size", get_world_size()) == get_world_size():
            log.info("Restoring RNG states...")
            rng_state = state_dict["rng"]
            self.restore_rng_state(rng_state)
        else:
            log.warning(
                "Trainer will not restore RNG states since the RNG states in the checkpoint are missing or invalid. "
                "This typically happens when restoring from an unsharded checkpoint or a checkpoint that was saved "
                "with a different world size. If that's the case you can safely ignore this warning."
            )

    def restore_rng_state(self, rng_state: Dict[str, Any]) -> None:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        torch.cuda.set_rng_state(rng_state["cuda"])

    def _sync_lora_grads(self) -> None:
        """
        Manually synchronize replicated LoRA gradients.

        Used when adapters are injected after FSDP wrapping and therefore are
        not sharded/synchronized by FSDP itself.
        """
        if not self.manual_lora_grad_sync:
            return
        if not dist.is_available() or not dist.is_initialized():
            return
        dp_process_group = get_dp_process_group(self.mesh) if self.mesh is not None else None
        world_size = get_world_size(dp_process_group) if dp_process_group is not None else get_world_size()
        if world_size <= 1:
            return

        lora_grads: List[torch.Tensor] = []
        for name, param in self.fsdp_model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            lname = name.lower()
            if "lora_a" in lname or "lora_b" in lname:
                lora_grads.append(param.grad)

        if not lora_grads:
            return

        # Coalesce gradients into FP32 chunks to reduce collective overhead and
        # preserve small values during reduction.
        chunk: List[torch.Tensor] = []
        chunk_numel = 0

        def _flush(current: List[torch.Tensor]) -> None:
            if not current:
                return
            flat = torch.cat([g.detach().reshape(-1).to(torch.float32) for g in current], dim=0)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=dp_process_group)
            flat.div_(world_size)
            offset = 0
            for grad in current:
                n = grad.numel()
                grad.copy_(flat[offset:offset + n].to(dtype=grad.dtype).view_as(grad))
                offset += n

        for grad in lora_grads:
            n = grad.numel()
            if chunk and (chunk_numel + n > self.lora_grad_sync_chunk_numel):
                _flush(chunk)
                chunk = []
                chunk_numel = 0
            chunk.append(grad)
            chunk_numel += n
        _flush(chunk)

    def save_checkpoint(self, checkpoint_type: CheckpointType, optim=True) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        if checkpoint_type == CheckpointType.sharded:
            suffix = ""
            current_checkpoints = self.checkpoints
            num_checkpoints_to_keep = self.cfg.save_num_checkpoints_to_keep
        elif checkpoint_type == CheckpointType.sharded_ephemeral:
            suffix = ""
            current_checkpoints = self.ephemeral_checkpoints
            num_checkpoints_to_keep = 1
        else:
            raise NotImplementedError(checkpoint_type)

        self.last_sharded_checkpoint_step = self.global_step

        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)

        checkpoint_dir = join(self.cfg.save_folder, f"step{self.global_step}{suffix}")
        current_checkpoints.append(checkpoint_dir)

        # torch.distributed.checkpointing can experience weird transients errors, where one
        # process will hit "800 operation not permitted"
        # barrier/synchronize/gc to try and fix the issue
        gc_cuda()
        barrier()
        torch.cuda.synchronize(self.device)

        self.checkpointer.save(
            checkpoint_dir,
            self.fsdp_model,
            self.optim if optim else None,
            self.trainer_state_dict(),
            config=self.cfg,
        )

        self.remove_checkpoints(current_checkpoints, num_checkpoints_to_keep)
        barrier()
        gc_cuda()
        return checkpoint_dir

    def save_lora_checkpoint(self, save_and_remove: bool = True, epoch: Optional[int] = None) -> Tuple[PathOrStr, Optional[PathOrStr]]:
        suffix = "-lora"
        current_checkpoints = self.lora_checkpoints
        num_checkpoints_to_keep = self.cfg.save_num_checkpoints_to_keep

        self.optim.zero_grad(set_to_none=True)

        if not epoch:
            checkpoint_dir = join(self.cfg.save_folder, f"step{self.global_step}{suffix}")
        else:
            checkpoint_dir = join(self.cfg.save_folder, f"ep{epoch}{suffix}")

        checkpoint_dir_llm = checkpoint_dir + "-llm"
        checkpoint_dir_vision = checkpoint_dir + "-vision"


        if save_and_remove:
            current_checkpoints.append(checkpoint_dir_llm)
            current_checkpoints.append(checkpoint_dir_vision)

        barrier()
        torch.cuda.synchronize(self.device)

        rank0 = get_global_rank() == 0

        sd_options = dist_cp_sd.StateDictOptions(full_state_dict=True, cpu_offload=True)
        full_state = dist_cp_sd.get_model_state_dict(self.fsdp_model, options=sd_options)

        transformer_state = self._slice_state_dict(full_state, "transformer")
        vision_state = self._slice_state_dict(full_state, "vision_backbone")
        if rank0 and not transformer_state:
            log.warning("No transformer weights found when saving LoRA checkpoint.")
        if rank0 and not vision_state:
            log.warning("No vision backbone weights found when saving LoRA checkpoint.")

        self.fsdp_model.transformer.save_pretrained(
            checkpoint_dir_llm,
            state_dict=transformer_state,
            safe_serialization=False,
            is_main_process=rank0,
        )
        self.fsdp_model.vision_backbone.save_pretrained(
            checkpoint_dir_vision,
            state_dict=vision_state,
            safe_serialization=False,
            is_main_process=rank0,
        )

        if self.cfg.save_merged_lora_checkpoint:
            log.info("Saving merged LoRA checkpoint...")
            self._save_merged_lora_checkpoint(
                checkpoint_dir=checkpoint_dir,
                checkpoint_dir_llm=checkpoint_dir_llm,
                checkpoint_dir_vision=checkpoint_dir_vision,
                full_state=full_state,
                save_and_remove=save_and_remove,
            )

        del full_state, transformer_state, vision_state

        if save_and_remove:
            self.remove_checkpoints(current_checkpoints, num_checkpoints_to_keep * 2)
        
        barrier()
        gc_cuda()
        return checkpoint_dir

    @staticmethod
    def _slice_state_dict(full_state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        prefix = f"{prefix}."
        return {k[len(prefix):]: v for k, v in full_state.items() if k.startswith(prefix)}

    def _save_merged_lora_checkpoint(
        self,
        *,
        checkpoint_dir: str,
        checkpoint_dir_llm: str,
        checkpoint_dir_vision: str,
        full_state: Dict[str, torch.Tensor],
        save_and_remove: bool,
    ) -> None:
        if checkpoint_dir.endswith("-lora"):
            merged_checkpoint_dir = checkpoint_dir[: -len("-lora")] + "-merged"
        else:
            merged_checkpoint_dir = f"{checkpoint_dir}-merged"

        merge_and_save_unsharded(
            merged_checkpoint_dir,
            self.fsdp_model,
            self.cfg,
            checkpoint_dir_llm=checkpoint_dir_llm,
            checkpoint_dir_vision=checkpoint_dir_vision,
            overwrite=self.cfg.save_overwrite,
            full_state=full_state,
        )
        if save_and_remove:
            self.merged_lora_checkpoints.append(merged_checkpoint_dir)
            self.remove_checkpoints(self.merged_lora_checkpoints, self.cfg.save_num_checkpoints_to_keep)

    def restore_checkpoint(
        self,
        load_path: PathOrStr,
        local_cache: Optional[PathOrStr] = None,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
        allow_missing_keys: bool = False
    ):
        trainer_state = self.checkpointer.load(
            load_path, self.fsdp_model, self.optim,
            load_optimizer_state=load_optimizer_state,
            load_trainer_state=load_trainer_state,
            allow_missing_keys=allow_missing_keys
        )
        if load_trainer_state:
            self.load_trainer_state_dict(trainer_state)
            if self.global_step >= self.cfg.stop_at:
                raise ValueError(f"Checkpointed it at {self.global_step}, but stop_at is {self.cfg.stop_at}")
            if self.global_step >= self.max_steps:
                raise ValueError(f"Checkpointed it at {self.global_step}, but max steps is {self.max_steps}")
        gc_cuda()
        barrier()

    def _remove_sharded_checkpoint(self, idx: int, checkpoints: List[Path]):
        oldest_checkpoint = checkpoints.pop(idx)
        barrier()
        if get_fs_local_rank() == 0:
            clear_directory(oldest_checkpoint)
        barrier()

    def remove_checkpoints(self, current_checkpoints, num_checkpoints_to_keep):
        if num_checkpoints_to_keep > 0:
            while len(current_checkpoints) > num_checkpoints_to_keep:
                self._remove_sharded_checkpoint(0, current_checkpoints)

    def move_to_device(self, batch, device):
        return move_to_device(batch, device)

    def _log_timing(self, name: str) -> None:
        """Log the most recent timing for an operation."""
        elapsed = self._timer_manager.get_last(name)
        if elapsed is not None:
            log.info(f"[TIMING] {name}: {elapsed*1000:.2f}ms ({elapsed:.4f}s)")


    def get_labels(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Labels are just input IDs shifted to the left (first item is ignored).
        labels, label_mask, attention_mask, instance_mask = (
            batch["input_ids"].clone(),
            batch.get("label_mask"),
            batch.get("attention_mask"),
            batch.get("instance_mask"),
        )
        if label_mask is not None:
            labels.masked_fill_(~label_mask, -100)
        if attention_mask is not None:
            labels.masked_fill_(attention_mask == 0.0, -100)
        if instance_mask is not None:
            labels.masked_fill_(~instance_mask.unsqueeze(-1), value=-100)
        return labels[..., 1:].contiguous()

    def _apply_depth_code_input_noise(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, torch.Tensor]]]:
        """Replace a fraction of teacher-forced depth code inputs while keeping labels intact."""
        rate = self._depth_code_input_noise_rate
        if (
            not self._enable_depth_reasoning
            or rate <= 0.0
            or self._depth_start_id is None
            or self._depth_end_id is None
            or self._depth_code_token_ids is None
        ):
            return batch, None

        input_ids = batch.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            return batch, None

        code_ids = self._depth_code_token_ids.to(input_ids.device)
        starts_seen = (input_ids == self._depth_start_id).cumsum(dim=-1)
        ends_seen = (input_ids == self._depth_end_id).cumsum(dim=-1)
        inside_depth_span = starts_seen > ends_seen
        is_depth_code = (input_ids[..., None] == code_ids.view(1, 1, -1)).any(dim=-1)
        eligible = inside_depth_span & is_depth_code

        selected = eligible & (torch.rand(input_ids.shape, device=input_ids.device) < rate)
        eligible_count = eligible.sum().float()
        selected_count = selected.sum().float()
        stats = {
            "depth_input_noise_fraction": selected_count / eligible_count.clamp_min(1.0),
            "depth_input_noise_tokens": selected_count,
        }

        random_code_indices = torch.randint(
            low=0,
            high=code_ids.numel(),
            size=input_ids.shape,
            device=input_ids.device,
        )
        random_code_ids = code_ids[random_code_indices]
        noisy_input_ids = torch.where(selected, random_code_ids, input_ids)

        noisy_batch = dict(batch)
        noisy_batch["input_ids"] = noisy_input_ids
        return noisy_batch, stats

    def _apply_depth_loss_masking(
        self,
        *,
        batch: Dict[str, Any],
        input_ids: torch.Tensor,
        subsegment_ids: Optional[torch.Tensor],
        flat_loss_masks: torch.Tensor,
        flat_labels: torch.Tensor,
    ) -> None:
        # Depth reasoning uses full depth-token supervision whenever enabled.
        return

    def _extract_depth_predictions(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        response_mask: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, Any]]:
        if self._depth_start_id is None or self._depth_end_id is None or self._depth_code_token_ids is None:
            return None
        code_ids = self._depth_code_token_ids.to(logits.device)
        chosen_row = -1
        start = -1
        end = -1
        for row_idx in range(input_ids.shape[0]):
            row_ids = input_ids[row_idx]
            starts = (row_ids == self._depth_start_id).nonzero(as_tuple=False).flatten()
            if starts.numel() == 0:
                continue
            row_start = int(starts[0].item())
            ends = (row_ids == self._depth_end_id).nonzero(as_tuple=False).flatten()
            end_candidates = ends[ends > row_start]
            if end_candidates.numel() == 0:
                continue
            chosen_row = int(row_idx)
            start = row_start
            end = int(end_candidates[0].item())
            break
        if chosen_row < 0 or end <= start + 1:
            return None

        token_to_code = {int(code_ids[idx].item()): int(idx) for idx in range(code_ids.shape[0])}
        gt_token_ids = input_ids[chosen_row, start + 1 : end]
        gt_codes = np.asarray([token_to_code.get(int(token_id.item()), 0) for token_id in gt_token_ids], dtype=np.int64)

        if logits.dim() == 3:
            pred_logits = logits[chosen_row, start : end - 1, :]
        else:
            if response_mask is None:
                return None
            seq_len = input_ids.shape[1]
            flat_mask = response_mask.reshape(-1).long()
            cumsum = torch.zeros(flat_mask.shape[0] + 1, dtype=torch.long, device=flat_mask.device)
            cumsum[1:] = flat_mask.cumsum(0)
            flat_start = chosen_row * seq_len + start
            logits_start = int(cumsum[flat_start].item())
            pred_logits = logits[logits_start : logits_start + gt_codes.shape[0], :]
        pred_codes = pred_logits[:, code_ids].argmax(dim=-1).detach().cpu().numpy().astype(np.int64)
        return {"pred_codes": pred_codes, "gt_codes": gt_codes}

    @staticmethod
    def _load_depth_vae(*, device: torch.device):
        from olmo.depth_visualization import load_depth_vae

        return load_depth_vae(device=device)

    def _depth_codes_to_rgb(
        self,
        codes: np.ndarray,
        *,
        height: int = 10,
        width: int = 10,
        scale: int = 16,
    ) -> np.ndarray:
        total = int(height) * int(width)
        padded = np.zeros(total, dtype=np.int64)
        flat_codes = np.asarray(codes, dtype=np.int64).reshape(-1)
        padded[: min(flat_codes.shape[0], total)] = flat_codes[: min(flat_codes.shape[0], total)]

        if self._depth_vae is not None:
            grid = torch.from_numpy(padded).long().reshape(1, height, width)
            with torch.no_grad():
                depth = self._depth_vae.decode(grid).squeeze().float()
            depth_min = float(depth.min())
            depth_max = float(depth.max())
            if depth_max - depth_min > 1e-6:
                depth = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth = depth * 0.0
            gray = (depth * 255).clamp(0, 255).to(torch.uint8).numpy()
            return np.stack([gray, gray, gray], axis=-1)

        grid = padded.astype(np.float32).reshape(height, width)
        norm = np.clip(grid / float(DEFAULT_NUM_DEPTH_TOKENS - 1), 0.0, 1.0)
        red = (norm * 255).astype(np.uint8)
        blue = ((1.0 - norm) * 255).astype(np.uint8)
        green = np.minimum(red, blue)
        rgb = np.stack([red, green, blue], axis=-1)
        return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)

    def model_forward(
        self,
        batch: Dict[str, Any],
        compute_z_loss: bool = False,
    ) -> Tuple:
        # shape: (batch_size, seq_len, vocab_size)
        loss_masks = batch["loss_masks"]
        labels = batch["labels"]
        response_mask = (loss_masks > 0)
        depth_response_mask_2d = response_mask
        keys_to_exclude = ["metadata"]
        if not self.cp_enabled:
            keys_to_exclude += ["loss_masks", "labels"]
        keys_to_exclude += list(_DEPTH_SUPERVISION_BATCH_KEYS)
        model_batch, depth_noise_stats = self._apply_depth_code_input_noise(batch)

        # need to pass loss_masks and labels to model when using cp_degree > 1 in order to chunk them across cp ranks
        with torch.autocast("cuda", dtype=self.cfg.autocast_precision):
            model_out = self.fsdp_model(
                **{
                    k: v
                    for k, v in model_batch.items()
                    if k not in keys_to_exclude and not k.startswith("_")
                },
                response_mask=response_mask,
                response_logits_only=self.cfg.response_logits_only,
            )
            if depth_noise_stats is not None:
                model_metrics = dict(model_out.metrics or {})
                model_metrics.update(depth_noise_stats)
                model_out = model_out._replace(metrics=model_metrics)
            logits = model_out.logits
            if not torch.isfinite(logits).all():
                max_abs_logits = float(torch.nan_to_num(logits.detach(), nan=0.0, posinf=0.0, neginf=0.0).abs().max().item())
                raise RuntimeError(
                    "Non-finite model logits before token loss: "
                    f"logits_finite={_tensor_all_finite(logits)}, "
                    f"max_abs_logits={max_abs_logits}, "
                    f"input_ids_shape={_tensor_shape(batch.get('input_ids'))}, "
                    f"attention_mask_shape={_tensor_shape(batch.get('attention_mask'))}, "
                    f"actions_shape={_tensor_shape(batch.get('actions'))}, "
                    f"actions_finite={_tensor_all_finite(batch.get('actions'))}, "
                    f"states_shape={_tensor_shape(batch.get('states'))}, "
                    f"states_finite={_tensor_all_finite(batch.get('states'))}, "
                    f"action_horizon_is_pad_shape={_tensor_shape(batch.get('action_horizon_is_pad'))}, "
                    f"action_dim_is_pad_shape={_tensor_shape(batch.get('action_dim_is_pad'))}, "
                    f"packed_action_chunk_is_valid_shape={_tensor_shape(batch.get('packed_action_chunk_is_valid'))}."
                )
            # get the sharded loss masks from forward pass of the model instead
            loss_masks = model_out.loss_masks if model_out.loss_masks is not None else loss_masks
            loss_masks = (loss_masks * (loss_masks > 0)).view(-1)
            # get the sharded labels
            labels = model_out.labels if model_out.labels is not None else labels
            labels = labels.long().view(-1)
            response_mask = model_out.response_mask if model_out.response_mask is not None else response_mask
            labels.masked_fill_(~(loss_masks > 0), -100)
            self._apply_depth_loss_masking(
                batch=batch,
                input_ids=batch["input_ids"],
                subsegment_ids=batch.get("subsegment_ids"),
                flat_loss_masks=loss_masks,
                flat_labels=labels,
            )
            if self.should_log_this_step() and self._enable_depth_reasoning and self._depth_code_token_ids is not None:
                try:
                    vis = self._extract_depth_predictions(
                        batch["input_ids"],
                        logits.detach(),
                        response_mask=depth_response_mask_2d,
                    )
                    if vis is not None:
                        self._last_depth_vis = vis
                except Exception as exc:
                    if get_global_rank() == 0 and self.global_step <= 5:
                        log.warning("Failed to capture depth visualization logits: %s", exc)

            logits_for_loss = logits.to(torch.float32).view(-1, logits.size(-1)) # for numerical stability
            if self.cfg.response_logits_only:
                loss_masks = loss_masks[response_mask.view(-1)]
                labels = labels[response_mask.view(-1)]
            ce_loss, z_loss = self.loss_fn(
                logits_for_loss,
                labels,
                ignore_index=-100,
                reduction="none",
                compute_z_loss=compute_z_loss, 
                z_loss_scale=self.cfg.softmax_auxiliary_loss_scale,
            )
            if not torch.isfinite(ce_loss).all() or (z_loss is not None and not torch.isfinite(z_loss).all()):
                valid_label_count = int((labels != -100).sum().item())
                ignored_label_count = int((labels == -100).sum().item())
                raise RuntimeError(
                    "Non-finite token loss from loss_fn: "
                    f"ce_loss_finite={_tensor_all_finite(ce_loss)}, "
                    f"z_loss_finite={_tensor_all_finite(z_loss)}, "
                    f"logits_finite={_tensor_all_finite(logits_for_loss)}, "
                    f"loss_masks_finite={_tensor_all_finite(loss_masks)}, "
                    f"valid_label_count={valid_label_count}, "
                    f"ignored_label_count={ignored_label_count}, "
                    f"logits_shape={tuple(logits_for_loss.shape)}."
                )

        ce_loss = torch.dot(ce_loss, loss_masks)
        z_loss = torch.dot(z_loss, loss_masks) if z_loss is not None else None
        if not torch.isfinite(ce_loss).all() or (z_loss is not None and not torch.isfinite(z_loss).all()):
            raise RuntimeError(
                "Non-finite reduced token loss after loss-mask application: "
                f"ce_loss_finite={_tensor_all_finite(ce_loss)}, "
                f"z_loss_finite={_tensor_all_finite(z_loss)}, "
                f"loss_masks_finite={_tensor_all_finite(loss_masks)}, "
                f"nonzero_loss_masks={int((loss_masks > 0).sum().item())}."
            )

        return ce_loss, z_loss, model_out

    def train_batch(
        self,
        batch: Dict[str, Any],
        compute_metrics,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        # Split into micro-batches.
        micro_batches = self.split_batch(batch)
        if self.cp_enabled:
            batch["loss_masks"] = batch["loss_masks"].to(self.device)
        loss_masks = batch["loss_masks"] * (batch["loss_masks"] > 0)
        if self.cfg.batch_divisor == BatchDivisor.global_batch:
            batch_size_in_tokens = loss_masks.sum()
            dist.all_reduce(batch_size_in_tokens)
            batch_size_in_tokens.div_(get_world_size())
        elif self.cfg.batch_divisor == BatchDivisor.global_batch_average:
            batch_size_in_tokens = loss_masks.sum()
            dist.all_reduce(batch_size_in_tokens)
            batch_size_in_tokens.div_(get_world_size())
            self._global_batch_size_average.append(batch_size_in_tokens.item())
            batch_size_in_tokens = np.mean(self._global_batch_size_average)
        elif self.cfg.batch_divisor == BatchDivisor.device_batch:
            batch_size_in_tokens = loss_masks.sum()
        else:
            raise ValueError()
        if batch_size_in_tokens.item() == 0:
            # Action-only batches may have zero text loss tokens; avoid NaN from division.
            batch_size_in_tokens = torch.tensor(1.0, device=self.device)
        del batch  # in case this helps reduce memory
        assert batch_size_in_tokens > 0

        total_loss = torch.tensor(0.0, device=self.device)
        for m_b, micro_batch in enumerate(micro_batches):
            try:
                ce_loss, z_loss, model_out = self.model_forward(
                    micro_batch,
                    compute_z_loss=self.cfg.softmax_auxiliary_loss,
                )
                if compute_metrics:
                    self._train_metrics.update(micro_batch, model_out, ce_loss, z_loss)

                ce_loss = ce_loss.sum() / batch_size_in_tokens
                if not torch.isfinite(ce_loss).all():
                    raise RuntimeError(
                        "Non-finite CE loss contribution during train_batch: "
                        f"global_step={self.global_step}, "
                        f"micro_batch_index={m_b}, "
                        f"batch_size_in_tokens={float(batch_size_in_tokens.item())}, "
                        f"ce_loss_finite={_tensor_all_finite(ce_loss)}."
                    )

                # Get loss to optimize for.
                if self.cfg.softmax_auxiliary_loss:
                    z_loss = z_loss.sum() / batch_size_in_tokens
                    if not torch.isfinite(z_loss).all():
                        raise RuntimeError(
                            "Non-finite z-loss contribution during train_batch: "
                            f"global_step={self.global_step}, "
                            f"micro_batch_index={m_b}, "
                            f"batch_size_in_tokens={float(batch_size_in_tokens.item())}, "
                            f"ce_loss_finite={_tensor_all_finite(ce_loss)}, "
                            f"z_loss_finite={_tensor_all_finite(z_loss)}."
                        )
                    loss = ce_loss + z_loss
                else:
                    loss = ce_loss
                    z_loss = None
                loss = loss * self.cp_degree  # scale loss to account for gradient averaging in FSDP
                if not torch.isfinite(loss).all():
                    raise RuntimeError(
                        "Non-finite base loss before auxiliary additions: "
                        f"global_step={self.global_step}, "
                        f"micro_batch_index={m_b}, "
                        f"ce_loss_finite={_tensor_all_finite(ce_loss)}, "
                        f"z_loss_finite={_tensor_all_finite(z_loss)}, "
                        f"cp_degree={self.cp_degree}."
                    )

                if model_out.metrics is not None:
                    if "AuxLoss" in model_out.metrics:
                        aux_loss = model_out.metrics["AuxLoss"] / len(micro_batches)
                        if not torch.isfinite(aux_loss).all():
                            raise RuntimeError(
                                "Non-finite AuxLoss contribution during train_batch: "
                                f"global_step={self.global_step}, "
                                f"micro_batch_index={m_b}, "
                                f"aux_loss_finite={_tensor_all_finite(aux_loss)}, "
                                f"metric_keys={sorted(model_out.metrics.keys())}."
                            )
                        loss += aux_loss
                    if "token_losses" in model_out.metrics:
                        token_losses = model_out.metrics.pop("token_losses")
                        if not torch.isfinite(token_losses).all():
                            raise RuntimeError(
                                "Non-finite token_losses metric during train_batch: "
                                f"global_step={self.global_step}, "
                                f"micro_batch_index={m_b}, "
                                f"token_losses_finite={_tensor_all_finite(token_losses)}, "
                                f"metric_keys={sorted(model_out.metrics.keys())}."
                            )
                        token_losses = token_losses / batch_size_in_tokens
                        if not torch.isfinite(token_losses).all():
                            raise RuntimeError(
                                "Non-finite token_losses contribution during train_batch: "
                                f"global_step={self.global_step}, "
                                f"micro_batch_index={m_b}, "
                                f"token_losses_finite={_tensor_all_finite(token_losses)}."
                            )
                        loss += token_losses

                    action_flow = None
                    if model_out.internal is not None:
                        action_flow = model_out.internal.get("action_flow_loss")
                    if action_flow is not None:
                        if not torch.isfinite(action_flow).all():
                            raise RuntimeError(
                                "Non-finite action_flow_loss before aggregation: "
                                f"global_step={self.global_step}, "
                                f"micro_batch_index={m_b}, "
                                f"action_flow_finite={_tensor_all_finite(action_flow)}, "
                                f"internal_keys={sorted(model_out.internal.keys()) if model_out.internal is not None else []}."
                            )
                        action_flow = action_flow / len(micro_batches)
                        if not torch.isfinite(action_flow).all():
                            raise RuntimeError(
                                "Non-finite action_flow_loss contribution during train_batch: "
                                f"global_step={self.global_step}, "
                                f"micro_batch_index={m_b}, "
                                f"action_flow_finite={_tensor_all_finite(action_flow)}."
                            )
                        loss += action_flow

                    if self.cfg.saliency_score_loss_wt is not None:
                        if model_out.metrics.get('saliency_difference', None) is not None:
                            loss += self.cfg.saliency_score_loss_wt * model_out.metrics['saliency_difference'] / len(micro_batches)

                        elif model_out.internal.get('pred_saliency', None) is not None and model_out.internal.get('gt_saliency', None) is not None:
                            pred_saliency = model_out.internal.pop('pred_saliency')
                            gt_saliency = model_out.internal.pop('gt_saliency')

                            valid = gt_saliency != -100
                            gt_saliency[~valid] = 0.0
                            pred_saliency[~valid] = 0.0

                            saliency_loss = F.binary_cross_entropy(pred_saliency, gt_saliency, reduction="sum") / torch.sum(valid * 1.0)
                            loss += self.cfg.saliency_score_loss_wt * saliency_loss / len(micro_batches)

                    if self.cfg.frame_score_loss_wt is not None and model_out.metrics.get("embedding_scores", None) is not None:
                        assert self.cfg.frame_score_loss_target is not None
                        embedding_scores = model_out.metrics['embedding_scores']
                        target = torch.tensor(self.cfg.frame_score_loss_target, device=embedding_scores.device, dtype=embedding_scores.dtype)
                        if self.cfg.frame_score_loss_type == "l1":
                            micro_batch_loss = l1_loss(embedding_scores, target)
                        elif self.cfg.frame_score_loss_type == "mse":
                            micro_batch_loss = mse_loss(embedding_scores, target)
                        elif self.cfg.frame_score_loss_type == "rmse":
                            micro_batch_loss = torch.sqrt(mse_loss(embedding_scores, target) + 1e-8)
                        else:
                            raise ValueError(f"Unsupported frame_score_loss_type: {self.cfg.frame_score_loss_type}")
                        loss += self.cfg.frame_score_loss_wt * micro_batch_loss / len(micro_batches)
                if not torch.isfinite(loss).all():
                    raise RuntimeError(
                        "Non-finite aggregated micro-batch loss after auxiliary additions: "
                        f"global_step={self.global_step}, "
                        f"micro_batch_index={m_b}, "
                        f"ce_loss_finite={_tensor_all_finite(ce_loss)}, "
                        f"z_loss_finite={_tensor_all_finite(z_loss)}, "
                        f"metric_keys={sorted(model_out.metrics.keys()) if model_out.metrics is not None else []}, "
                        f"internal_keys={sorted(model_out.internal.keys()) if model_out.internal is not None else []}."
                    )

                del model_out

                # Run backward pass.
                loss.backward()
                total_loss += loss.detach()
            except RuntimeError as exc:
                if self._looks_like_nonfinite_error(exc):
                    self._dump_nonfinite_micro_batch(
                        micro_batch,
                        reason=str(exc),
                        micro_batch_index=m_b,
                    )
                raise
            finally:
                # In case this helps with memory utilization.
                del micro_batch

        return total_loss

    def train_step(
        self,
        batch: Dict[str, Any],
        compute_metrics: bool = True,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # Zero-gradients.
        self.optim.zero_grad(set_to_none=True)

        if not self.cp_enabled:
            # Move tensors to the right device.
            batch = self.move_to_device(batch, self.device)

        # Run forward-backward pass
        if compute_metrics:
            self._train_metrics.reset()
        loss = self.train_batch(batch, compute_metrics)

        if torch.isnan(loss) or torch.isinf(loss):
            # Log the batch into a file for debugging
            # save_debug_batch(batch, self.cfg.save_folder, self.global_step, loss.item())
            raise RuntimeError(f"NaN or Inf loss detected after train_batch aggregation at global_step={self.global_step}: {loss.item()}")

        if compute_metrics:
            metrics = {f"train/{k}": v for k, v in self._train_metrics.compute().items()}
        else:
            metrics = {}

        should_log_optim_metrics_this_step = self.should_log_optim_metrics_this_step()
        if should_log_optim_metrics_this_step:
            # No current implementation of per-parameter metrics because I am not sure the
            # old very complex one makes sense anymore
            raise NotImplementedError()

        # If LoRA adapters were injected after FSDP wrap, their grads are replicated and
        # require explicit synchronization across data-parallel ranks.
        self._sync_lora_grads()

        # Clip gradient norms, norms are clipped per group name
        # Note group name might have multiple optimizer param groups
        param_norm_groups = defaultdict(list)
        for group in self.optim.param_groups:
            param_norm_groups[group["group_name"]].append(group)
        optim_metrics = {}
        grad_norms = []

        max_grad_norm = self.scheduler.get_max_grad_norm(
            self.cfg.max_grad_norm, self.scheduler_current, self.scheduler_max)
        if self.cfg.max_grad_norm_ratio is not None:
            raise NotImplementedError()
        if max_grad_norm is not None:
            for group_name, groups in param_norm_groups.items():
                params = flatten_lists(group["params"] for group in groups)
                grad_norm = clip_grad_norm(params, max_grad_norm=max_grad_norm)
                grad_norms.append(grad_norm)
                optim_metrics[f"{group_name}_grad_norm"] = grad_norm

        nonfinite_grad_groups = [
            key.removesuffix("_grad_norm")
            for key, value in optim_metrics.items()
            if key.endswith("_grad_norm") and not bool(torch.isfinite(value).all().item())
        ]
        if _distributed_any_flag(bool(nonfinite_grad_groups), self.device):
            if get_global_rank() == 0:
                group_msg = ", ".join(nonfinite_grad_groups) if nonfinite_grad_groups else "another rank"
                log.warning(
                    "Skipping optimizer step at global_step=%s because non-finite gradient norm was detected in %s.",
                    self.global_step,
                    group_msg,
                )
            self.optim.zero_grad(set_to_none=True)
            metrics["optim/skipped_nonfinite_grad_step"] = 1.0
            self.cur_train_loss = loss.item()
            self.min_train_loss = min(self.min_train_loss, self.cur_train_loss)
            return metrics

        # Adjust the learning rate.
        initial_lr_dict = {
            "connector": self.cfg.optimizer.connector_learning_rate,
            "vit": self.cfg.optimizer.vit_learning_rate,
            "llm": self.cfg.optimizer.llm_learning_rate,
            "frame_selector": self.cfg.optimizer.frame_selector_learning_rate,
            "temporal_token_scorer": self.cfg.optimizer.temporal_token_scorer_learning_rate,
        }
        if hasattr(self.cfg.optimizer, "action_expert_learning_rate"):
            initial_lr_dict["action_expert"] = self.cfg.optimizer.action_expert_learning_rate
        for group in self.optim.param_groups:
            group_name = group["group_name"]
            if group_name in initial_lr_dict:
                group["lr"] = self.scheduler.get_lr(
                    initial_lr_dict[group_name],
                    self.scheduler_current,
                    self.scheduler_max,
                    group_name,
                )
            else:
                group["lr"] = self.scheduler.get_lr(
                    self.cfg.optimizer.learning_rate, self.scheduler_current, self.scheduler_max
                )

        # Optimizer step.
        self.optim.step()

        # Collect metrics and check for NaN loss.
        # NOTE: this involves a bunch of host-device syncs so we wait until the last moment to do this.
        for key, value in optim_metrics.items():
            metrics[f"optim/{key}"] = value.item()
        self.cur_train_loss = loss.item()
        self.min_train_loss = min(self.min_train_loss, self.cur_train_loss)
        return metrics

    def split_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        microbatch_size = self.cfg.device_train_microbatch_size
        batch_size = batch["input_ids"].shape[0]
        has_chunked_actions = "packed_batch_idx" in batch
        if batch_size <= microbatch_size:
            return [batch]
        else:
            chunk_keys = {
                "actions",
                "states",
                "action_horizon_is_pad",
                "action_is_pad",
                "action_dim_is_pad",
                "packed_batch_idx",
                "packed_example_ids",
                "packed_action_chunk_is_valid",
            }
            normal_splits: Dict[str, Any] = {}
            for key, value in batch.items():
                if has_chunked_actions and key in chunk_keys:
                    continue
                if isinstance(value, torch.Tensor):
                    normal_splits[key] = value.split(microbatch_size, dim=0)
                elif isinstance(value, list):
                    normal_splits[key] = [
                        value[microbatch_size * i : microbatch_size * i + microbatch_size]
                        for i in range(math.ceil(batch_size / microbatch_size))
                    ]
                else:
                    raise ValueError(f"unexpected item in batch: '{key}={value}'")
            n_micro = len(normal_splits["input_ids"])
            chunk_splits: List[Dict[str, torch.Tensor]] = [{} for _ in range(n_micro)]
            if has_chunked_actions:
                chunk_batch_idx = batch["packed_batch_idx"]
                for i in range(n_micro):
                    start = i * microbatch_size
                    end = min(start + microbatch_size, batch_size)
                    mask = (chunk_batch_idx >= start) & (chunk_batch_idx < end)
                    chunk_data: Dict[str, torch.Tensor] = {}
                    if mask.sum() == 0:
                        if "packed_action_chunk_cap" not in batch:
                            raise RuntimeError(f"No action chunks found for microbatch range {start}:{end}")
                        action_template = batch.get("actions")
                        if not isinstance(action_template, torch.Tensor) or action_template.ndim != 3:
                            raise RuntimeError(
                                f"No action chunks found for microbatch range {start}:{end}, "
                                "and unable to infer dummy action shape from the batch."
                            )
                        horizon, action_dim = action_template.shape[1], action_template.shape[2]
                        chunk_data["actions"] = torch.zeros(
                            1, horizon, action_dim, dtype=action_template.dtype, device=action_template.device
                        )
                        if "states" in batch:
                            state_template = batch["states"]
                            if not isinstance(state_template, torch.Tensor):
                                raise ValueError("Chunk key 'states' must be a tensor")
                            chunk_data["states"] = torch.zeros(
                                (1, *state_template.shape[1:]),
                                dtype=state_template.dtype,
                                device=state_template.device,
                            )
                        action_pad_key = None
                        if "action_horizon_is_pad" in batch:
                            action_pad_key = "action_horizon_is_pad"
                        elif "action_is_pad" in batch:
                            action_pad_key = "action_is_pad"
                        if action_pad_key is not None:
                            action_pad_template = batch[action_pad_key]
                            if not isinstance(action_pad_template, torch.Tensor):
                                raise ValueError(f"Chunk key '{action_pad_key}' must be a tensor")
                            chunk_data[action_pad_key] = torch.ones(
                                (1, *action_pad_template.shape[1:]),
                                dtype=action_pad_template.dtype,
                                device=action_pad_template.device,
                            )
                        if "action_dim_is_pad" in batch:
                            action_dim_pad_template = batch["action_dim_is_pad"]
                            if not isinstance(action_dim_pad_template, torch.Tensor):
                                raise ValueError("Chunk key 'action_dim_is_pad' must be a tensor")
                            chunk_data["action_dim_is_pad"] = torch.ones(
                                (1, *action_dim_pad_template.shape[1:]),
                                dtype=action_dim_pad_template.dtype,
                                device=action_dim_pad_template.device,
                            )
                        chunk_data["packed_batch_idx"] = torch.zeros(
                            1, dtype=torch.long, device=chunk_batch_idx.device
                        )
                        if "packed_example_ids" in batch:
                            example_ids = batch["packed_example_ids"]
                            if not isinstance(example_ids, torch.Tensor):
                                raise ValueError("Chunk key 'packed_example_ids' must be a tensor")
                            chunk_data["packed_example_ids"] = torch.full(
                                (1,), -1, dtype=example_ids.dtype, device=example_ids.device
                            )
                        if "packed_action_chunk_is_valid" in batch:
                            valid_template = batch["packed_action_chunk_is_valid"]
                            if not isinstance(valid_template, torch.Tensor):
                                raise ValueError("Chunk key 'packed_action_chunk_is_valid' must be a tensor")
                            chunk_data["packed_action_chunk_is_valid"] = torch.zeros(
                                1, dtype=valid_template.dtype, device=valid_template.device
                            )
                    else:
                        for key in chunk_keys:
                            if key not in batch:
                                continue
                            value = batch[key]
                            if not isinstance(value, torch.Tensor):
                                raise ValueError(f"Chunk key '{key}' must be a tensor")
                            sliced = value[mask]
                            if key == "packed_batch_idx":
                                sliced = sliced - start
                            chunk_data[key] = sliced
                    chunk_splits[i] = chunk_data
            return [
                {
                    **{key: value[i] for key, value in normal_splits.items()},  # type: ignore
                    **chunk_splits[i],
                }
                for i in range(n_micro)
            ]

    def system_metrics(self) -> Dict[str, float]:
        metrics = {}
        if self.global_step < 3 or self.global_step % 10 == 0:
            peak_gpu_mb = peak_gpu_memory()
            if peak_gpu_mb is not None:
                metrics["System/Peak GPU Memory (MB)"] = peak_gpu_mb
        return metrics

    def log_metrics_to_console(self, prefix: str, metrics: Dict[str, float]):
        def format_float(value: float) -> str:
            if value < 0.0001:
                return str(value)  # scientific notation
            elif value > 1000:
                return f"{int(value):,d}"
            elif value > 100:
                return f"{value:.1f}"
            elif value > 10:
                return f"{value:.2f}"
            elif value > 1:
                return f"{value:.3f}"
            else:
                return f"{value:.4f}"

        log.info(
            f"{prefix}\n"
            + "\n".join(
                [
                    f"    {name}={format_float(value)}"
                    for name, value in metrics.items()
                    # there's too many optimizer metrics
                    # also skip non-float wandb.Metrics from inference evaluators
                    if (
                        isinstance(value, (int, float)) and (
                            name == "optim/total_grad_norm"
                            or (not name.startswith("optim/") and not name.startswith("batch/"))
                    ))
                ]
            )
        )

    def should_log_optim_metrics_this_step(self) -> bool:
        if self.cfg.wandb is None:
            # We only log optimizer-specific metrics to W&B, since there are usually too many metrics
            # to log to the console.
            return False
        optim_log_interval = self.cfg.optimizer.metrics_log_interval
        if optim_log_interval is None:
            optim_log_interval = self.cfg.wandb.log_interval
        elif optim_log_interval <= 0:
            return False
        else:
            optim_log_interval = max(optim_log_interval, self.cfg.wandb.log_interval)
        return self.global_step % optim_log_interval == 0

    def should_log_this_step(self) -> bool:
        if self.global_step % self.cfg.console_log_interval == 0:
            return True
        elif self.cfg.wandb is not None and self.global_step % self.cfg.wandb.log_interval == 0:
            return True
        else:
            return False

    def inference_eval(self) -> Dict[str, Union[float, WBValue]]:
        self.optim.zero_grad(set_to_none=True)

        all_metrics = {}
        all_eval_t0 = time.perf_counter()
        if not self.cfg.distributed_eval:
            log.info(f"Setting up non-FSDP model...")
            self.model.to(self.device)
            state_dict = get_state_dict(self.fsdp_model, [], options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ))[0]
            set_model_state_dict(self.model, state_dict, options=StateDictOptions(
                broadcast_from_rank0=True, full_state_dict=True))
            model = self.model
        else:
            model = self.fsdp_model
        model.eval()

        all_eval_t0 = time.perf_counter()
        for evaluator in self.inference_evaluators:
            t0 = time.perf_counter()
            log.info(f"Running evaluation for '{evaluator.label}'...")
            if self.cfg.save_folder and self.cfg.save_inloop_predictions:
                # Add a metric that will save our predictions
                output_dir = join(self.cfg.save_folder, "inloop-predictions", f"step{self.global_step}-{evaluator.label}")
                save_inloop_metric = SavePredictions(output_dir, save_tokens=False)
                evaluator = dataclasses.replace(evaluator, evaluator=dataclasses.replace(
                    evaluator.evaluator, metrics=evaluator.evaluator.metrics + [save_inloop_metric]
                ))
            dataset_metrics = evaluator.run(
                model,
                device=self.device,
                autocast_precision=self.cfg.autocast_precision,
                is_distributed=self.cfg.distributed_eval,
                pbar=False,
            )
            self.log_metrics_to_console(f"{evaluator.label}", dataset_metrics)
            all_metrics.update({f"{evaluator.label}/{k}": v for k, v in dataset_metrics.items()})
            log.info(f"Eval for '{evaluator.label}' done in {time.perf_counter()-t0:0.1f} seconds")
        if len(self.inference_evaluators) > 1:
            log.info(f"All evals took done in {time.perf_counter()-all_eval_t0:0.1f} seconds")
        if not self.cfg.distributed_eval:
            self.model.cpu()
        return all_metrics

    def loss_eval(self) -> Dict[str, Union[float, WBValue]]:
        self.optim.zero_grad(set_to_none=True)
        self.fsdp_model.eval()
        eval_metrics = {}
        for evaluator in self.evaluators:
            t0 = time.perf_counter()
            log.info(f"Running evaluation for '{evaluator.label}'...")
            metrics = evaluator.run(
                self.fsdp_model, self.device,
                autocast_precision=self.cfg.autocast_precision,
                loss_fn=self.loss_fn,
                cp_enabled=self.cp_enabled,
            )
            eval_metrics.update({f"{evaluator.label}/{k}": v for k, v in metrics.items()})
            log.info(f"Eval for '{evaluator.label}' done in {time.perf_counter()-t0:0.1f} seconds")
            self.log_metrics_to_console(f"{evaluator.label}", metrics)
        return eval_metrics

    def _handle_interrupt(self, signalnum, stack_frame):
        del stack_frame

        signame: Optional[str] = None
        if signalnum == signal.SIGTERM:
            signame = "SIGTERM"
        elif signalnum == signal.SIGINT:
            signame = "SIGINT"

        if signame is not None:
            msg = f"{signame} received"
        else:
            msg = f"Sig({signalnum}) received"

        log.warning(msg)
        self._cancelled = True
        self._cancel_reason = msg

    def check_if_cancelled(self) -> Tuple[bool, int]:
        should_cancel = self._cancelled
        cancel_reason = self._cancel_reason
        extra_steps = self.cfg.extra_steps_after_cancel
        if get_global_rank() == 0 and not should_cancel:
            if self.cfg.time_limit is not None and time.time() - self._start_time >= self.cfg.time_limit:
                # First check if we've reached the training time limit.
                should_cancel = True
                cancel_reason = "time limit reached"

        run_canceled = synchronize_flag(should_cancel, self.device)
        if run_canceled:
            if cancel_reason is None:
                if extra_steps > 0:
                    log.warning(f"Run canceled, stopping in {extra_steps} more steps...")
                else:
                    log.warning("Run canceled")
            else:
                if extra_steps > 0:
                    log.warning(f"Run canceled due to {cancel_reason}, stopping in {extra_steps} more steps...")
                else:
                    log.warning(f"Run canceled due to {cancel_reason}")
        return run_canceled, extra_steps

    def get_eta(self) -> str:
        if self._train_start_time is None:
            return "???"
        if self.cfg.stop_at:
            steps_left = self.cfg.stop_at - self.global_step
        else:
            steps_left = self.max_steps - self.global_step
        time_passed = time.monotonic() - self._train_start_time
        seconds_per_step = time_passed / (self.global_step - self._start_step)
        seconds_left = seconds_per_step * steps_left
        # Round off to minutes to make it the string easier to parse
        minutes_left = 1 + seconds_left // 60
        return format_timedelta(timedelta(minutes=minutes_left))

    def fit(self):
        if self.cfg.stop_after is not None:
            if self.cfg.stop_at is None:
                self.cfg.stop_at = self.global_step + self.cfg.stop_after
            else:
                self.cfg.stop_at = min(self.cfg.stop_at, self.global_step + self.cfg.stop_after)

        self._start_time = time.time()
        self._gc_init_state = gc.isenabled()  # cache if garbage collection is enabled, reset on close.

        # Disable automatic garbage collection, FSDP doesn't work well with it.
        if self.cfg.gen1_gc_interval is not None:
            gc.disable()

        if self.cfg.load_path is not None and self.global_step > 0 and self.cfg.eval_on_load:
            eval_metrics = self.loss_eval()
            if wandb.run is not None:
                wandb.log(eval_metrics, step=self.global_step)

            eval_metrics = self.inference_eval()
            if wandb.run is not None:
                wandb.log(eval_metrics, step=self.global_step)
            torch.cuda.empty_cache()

        # Set model to 'train' mode.
        self.fsdp_model.train()

        # Initialize monitors.
        speed_monitor = SpeedMonitor(self.cfg.speed_monitor)
        lr_monitor = LRMonitor(self.optim)
        batch_monitor = BatchStatsMonitor()

        # Log system metrics at the start of training.
        sys_metrics = self.system_metrics()
        if sys_metrics:
            self.log_metrics_to_console("Pre-train system metrics", sys_metrics)
            if wandb.run is not None:
                wandb.log(sys_metrics, step=0)

        lerobot_sampling_metrics = lerobot_tag_sampling_rate_metrics(
            self.cfg.data.kwargs_mixture,
            vlm_mixture=None if self.cfg.vlm_data is None else self.cfg.vlm_data.kwargs_mixture,
            vlm_loader_rate=self.cfg.vlm_loader_rate,
        )
        if lerobot_sampling_metrics:
            self.log_metrics_to_console("LeRobot tag sampling rates (actual)", lerobot_sampling_metrics)
            if wandb.run is not None:
                wandb.log(lerobot_sampling_metrics, step=0)

        # Python Profiler stuff
        if self.cfg.python_profiling:
            python_profiler = cProfile.Profile()
        else:
            python_profiler = None

        # PyTorch Profiler stuff
        if self.cfg.torch_profiling and get_global_rank() == 0:
            from torch.profiler import schedule

            profiling_schedule = schedule(wait=1, warmup=5, active=3, repeat=1)

            def on_trace_ready(p):
                profiler_output_dir = Path(self.cfg.save_folder) / "profiler"
                profiler_output_dir.mkdir(exist_ok=True)

                output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=32)
                log.info(f"Profile by total GPU time at step {p.step_num}:\n{output}")
                output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=32)
                log.info(f"Profile by total CPU time at step {p.step_num}:\n{output}")

                p.export_chrome_trace(
                    str(trace_path := (profiler_output_dir / f"{p.step_num}.chrome_trace.json.gz"))
                )

            from torch.profiler import ProfilerActivity

            torch_profiler = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=True,
                schedule=profiling_schedule,
                on_trace_ready=on_trace_ready,
            )
            del profiling_schedule
        else:
            import contextlib

            torch_profiler = contextlib.nullcontext()

        # Train.
        first_batch: bool = True
        cancel_initiated: bool = False
        stop_at: Optional[int] = self.cfg.stop_at
        save_checkpoints: bool = True
        self._start_step = self.global_step
        with torch_profiler as p:
            for epoch in range(self.epoch or 0, self.max_epochs):
                while True:
                    use_vlm_loader = self._should_use_vlm_loader_for_next_step()
                    active_loader = self.vlm_loader if use_vlm_loader else self.train_loader
                    assert active_loader is not None
                    batch = self._get_train_batch_from_loader(active_loader, use_vlm_loader=use_vlm_loader)
                    if use_vlm_loader:
                        strip_action_supervision_from_batch(batch)

                    # Bookkeeping.
                    batch_size, seq_len = batch["input_ids"].shape
                    global_batch_size = batch_size * self.dp_world_size  # assumes batch size equal across ranks
                    self.global_step += 1
                    if hasattr(self.fsdp_model, "_global_step"):
                        self.fsdp_model._global_step = self.global_step
                    self.global_train_examples_seen_this_epoch += global_batch_size
                    if use_vlm_loader:
                        self.vlm_train_examples_seen_this_epoch += global_batch_size
                    else:
                        self.primary_train_examples_seen_this_epoch += global_batch_size
                    self.global_train_tokens_seen += global_batch_size * seq_len

                    speed_monitor.batch_start(
                        self.global_train_tokens_seen,
                        (batch_size * seq_len) / self.cp_degree,  # num tokens in batch for this device
                        (batch["loss_masks"] > 0).sum() / self.cp_degree,  # approximate num loss tokens in batch for this device
                        # We start monitoring speed after the first batch since the first
                        # batch might be an outlier due to compiling and other initialization overhead.
                        record=not first_batch,
                    )
                    batch_monitor.log_batch(batch)

                    if use_vlm_loader:
                        vlm_cfg = self.cfg.vlm_data
                        if vlm_cfg is not None and vlm_cfg.packing and vlm_cfg.packing.track_packing_state:
                            self._record_loader_worker_states(batch, use_vlm_loader=True)
                    elif self.cfg.data.packing and self.cfg.data.packing.track_packing_state:
                        self._record_loader_worker_states(batch, use_vlm_loader=False)

                    should_log_this_step = self.should_log_this_step()

                    if self._train_start_time is None:
                        # Start timing after the first step so we don't count warm-up
                        self._train_start_time = time.monotonic()

                    # Run train step on batch.
                    metrics = self.train_step(
                        batch,
                        compute_metrics=should_log_this_step,
                    )

                    # Maybe collect other metrics.
                    if should_log_this_step:
                        metrics.update(speed_monitor.check())
                        metrics.update(self.system_metrics())
                        metrics.update(batch_monitor.check(self.device))
                        metrics.update(lr_monitor.check())
                        metrics["batch/is_vlm_loader"] = 1.0 if use_vlm_loader else 0.0
                        if use_vlm_loader:
                            metrics.pop("train/action_flow_loss", None)
                            metrics = rename_vlm_train_metrics(metrics)

                    # Do beaker logging
                    if (
                        self.beaker_logger and
                        (
                            (self.global_step % self.beaker_logger.log_interval == 0) or
                            # Log on step 0 so we can tell the model is done initializing
                            (self.beaker_logger.log_interval == 0 and self.global_step == 1)
                        )
                    ):
                        self.beaker_logger.log_progress(self.global_step, stop_at, self.get_eta())

                    # Log metrics to console.
                    if self.global_step % self.cfg.console_log_interval == 0:
                        header = f"[step={self.global_step}/{self.max_steps}, eta={self.get_eta()}]"
                        if get_global_rank() == 0:
                            self.log_metrics_to_console(header, metrics)
                        else:
                            log.info(header)

                    # Log metrics to W&B.
                    if (
                        wandb.run is not None
                        and self.cfg.wandb is not None
                        and self.global_step % self.cfg.wandb.log_interval == 0
                    ):
                        if self._last_depth_vis is not None and get_global_rank() == 0:
                            try:
                                metrics["train/depth_gt"] = wandb.Image(
                                    self._depth_codes_to_rgb(self._last_depth_vis["gt_codes"]),
                                    caption="GT depth",
                                )
                                metrics["train/depth_pred"] = wandb.Image(
                                    self._depth_codes_to_rgb(self._last_depth_vis["pred_codes"]),
                                    caption="Predicted depth",
                                )
                            except Exception as exc:
                                log.debug("Failed to build depth visualization images: %s", exc)
                            finally:
                                self._last_depth_vis = None
                        wandb.log(metrics, step=self.global_step)

                    # Check if/when run should be canceled.
                    if not cancel_initiated and self.global_step % self.cfg.canceled_check_interval == 0:
                        cancel_initiated, extra_steps = self.check_if_cancelled()
                        if cancel_initiated:
                            stop_at = (
                                self.global_step + extra_steps
                                if stop_at is None
                                else min(self.global_step + extra_steps, stop_at)
                            )

                    # Maybe save sharded checkpoint.
                    done_training = stop_at is not None and self.global_step >= stop_at
                    if save_checkpoints and (
                        cancel_initiated
                        or (
                                (self.global_step == self.cfg.save_at
                                 or (self.global_step % self.cfg.save_interval == 0)
                            )
                            and self.cfg.save_num_checkpoints_to_keep != 0
                        )
                    ):
                        log.info("Saving checkpoint...")
                        checkpoint_path = self.save_checkpoint(
                            CheckpointType.sharded,
                            optim=(not done_training or self.cfg.save_final_optim)
                        )
                        log.info(f"Checkpoint saved to {checkpoint_path}")

                        if self.cfg.model.lora_enable:
                            log.info("Saving LoRA checkpoint...")
                            lora_checkpoint_path = self.save_lora_checkpoint()
                            log.info(f"LoRA checkpoint saved to {lora_checkpoint_path}-*")

                            if self.cfg.save_merged_lora_checkpoint:
                                log.info(f"Merged LoRA checkpoint saved to {lora_checkpoint_path}-merged")

                        # Remove any ephemeral checkpoints.
                        self.remove_checkpoints(self.ephemeral_checkpoints, 0)

                        # Reset speed monitor so that we don't count the time taken to save checkpoints.
                        speed_monitor.reset()

                        # If the run was just canceled this will be the final checkpoint.
                        if cancel_initiated:
                            save_checkpoints = False
                    elif (
                        self.cfg.save_interval_ephemeral is not None
                        and self.global_step % self.cfg.save_interval_ephemeral == 0
                    ):
                        log.info("Saving ephemeral checkpoint...")
                        checkpoint_path= self.save_checkpoint(CheckpointType.sharded_ephemeral)
                        log.info(f"Checkpoint saved to {checkpoint_path}")

                        # Reset speed monitor so that we don't count the time taken to save checkpoints.
                        speed_monitor.reset()

                    # Maybe run evaluations.
                    last_step = stop_at and (self.global_step >= stop_at)
                    if not cancel_initiated and self.cfg.eval_interval > 0 and (
                        (self.global_step % self.cfg.eval_interval == 0) or
                        (last_step and self.cfg.eval_on_last_step) or
                        (self.global_step in self.cfg.eval_on)
                    ):
                        eval_metrics = self.loss_eval()

                        # Log metrics to W&B.
                        if wandb.run is not None:
                            wandb.log(eval_metrics, step=self.global_step)

                        # Reset speed monitor so that we don't count the time taken to run evaluations.
                        speed_monitor.reset()

                        # Reset model to 'train' mode.
                        self.fsdp_model.train()

                    if not cancel_initiated and (
                        self.inference_evaluators and
                        self.cfg.inf_eval_interval > 0 and
                        ((self.global_step % self.cfg.inf_eval_interval == 0) or
                         (self.cfg.eval_on_last_step and last_step) or
                         (self.global_step in self.cfg.eval_on))
                    ):
                        eval_metrics = self.inference_eval()

                        # Log metrics to W&B.
                        if wandb.run is not None:
                            wandb.log(eval_metrics, step=self.global_step)

                        # Reset speed monitor so that we don't count the time taken to run evaluations.
                        speed_monitor.reset()

                        # Reset model to 'train' mode.
                        self.fsdp_model.train()

                    # End of batch.
                    first_batch = False
                    if p is not None:
                        p.step()

                    if stop_at is not None and self.global_step >= stop_at:
                        break

                    # Run generation 1 garbage collection.
                    if self.cfg.gen1_gc_interval is not None and self.global_step % self.cfg.gen1_gc_interval == 0:
                        gc.collect(1)

                    # Python Profiler stuff
                    # We do this now, at the bottom of this loop, so we capture the work of getting the next batch.
                    if python_profiler is not None:
                        if self.global_step == 5:
                            python_profiler.enable()
                        elif self.global_step == 8:
                            python_profiler.disable()
                            python_profiler.print_stats(sort=SortKey.CUMULATIVE)
                            python_profiler = None
                else:
                    log.info("Training epoch complete")
                    self.epoch = epoch + 1
                    self.global_train_examples_seen_this_epoch = 0
                    self.primary_train_examples_seen_this_epoch = 0
                    self.vlm_train_examples_seen_this_epoch = 0
                    self._train_loader_iter = None
                    self._vlm_loader_iter = None
                    if self.epoch < self.max_epochs:
                        self.dataset.reshuffle()
                    continue
                break

        # Save final checkpoint.
        if save_checkpoints:
            if (
                self.cfg.save_num_checkpoints_to_keep != 0
                and self.last_sharded_checkpoint_step != self.global_step
            ):
                log.info("Saving final checkpoint...")
                checkpoint_path = self.save_checkpoint(
                    CheckpointType.sharded, optim=self.cfg.save_final_optim)
                log.info(f"Checkpoint saved to {checkpoint_path}")

                if self.cfg.model.lora_enable:
                    log.info("Saving final LoRA checkpoint...")
                    lora_checkpoint_path = self.save_lora_checkpoint()
                    log.info(f"LoRA checkpoint saved to {lora_checkpoint_path}-*")

                    if self.cfg.save_merged_lora_checkpoint:
                        log.info(f"Merged LoRA checkpoint saved to {lora_checkpoint_path}-merged")

            if self.cfg.save_final_unsharded_checkpoint:
                log.info("Saving final unsharded checkpoint...")
                checkpoint_path = join(normalize_path(self.cfg.save_folder), f"step{self.global_step}-unsharded")
                save_unsharded(checkpoint_path, self.fsdp_model, None, self.cfg, self.cfg.save_overwrite)
                log.info(f"Checkpoint saved to {checkpoint_path}")

    def close(self, exit_code: int = 0) -> None:
        if wandb.run is not None:
            if exit_code != 0:
                log.info(f"Finishing wandb with exit code {exit_code}")
            wandb.finish(exit_code=exit_code, quiet=True)
        gc_cuda()
        if self._gc_init_state:
            gc.enable()
        else:
            gc.disable()
        if self.beaker_logger is not None and exit_code == 0:
            self.beaker_logger.finish()

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del exc_val, exc_tb
        self.close(0 if exc_type is None else 1)
