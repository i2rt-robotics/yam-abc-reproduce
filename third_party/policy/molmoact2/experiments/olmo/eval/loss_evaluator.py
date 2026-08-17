"""Class to build metrics for a model based on the loss"""
import dataclasses
import logging
import math
from dataclasses import dataclass, field
from itertools import islice
from typing import Dict, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
import torchmetrics
import wandb
from torch.utils.data import DataLoader
from torchmetrics import Metric, MeanMetric
from tqdm import tqdm
from wandb.sdk.data_types.base_types.wb_value import WBValue

from olmo.config import BaseConfig, D
from olmo.data.data_loader import DataLoaderConfig
from olmo.eval.save_eval_data_config import SaveEvalDataConfig
from olmo.models.molmo.molmo import MolmoConfig
from olmo.torch_util import move_to_device, get_world_size, get_global_rank

__all__ = ["LossMetrics", "LossDatasetEvaluator", "LossDatasetEvaluatorConfig"]

log = logging.getLogger(__name__)


def _is_hist(metric_name):
    return (metric_name in ["HighResSelection", "HighResVals"]) or metric_name.endswith("Hist")


class LossMetrics:
    """Aggregates loss metrics from a forward pass"""

    def __init__(self, device, collect_outputs=False, reduce_loss_metrics_manually=False):
        self.device = device
        self.eval_metrics: Dict[str, MeanMetric] = dict(
            CrossEntropyLoss=MeanMetric("error").to(device),
            ZLoss=MeanMetric("error").to(device),
            Accuracy=MeanMetric("error").to(device),
        )
        self.reduce_loss_metrics_manually = reduce_loss_metrics_manually

    def reset(self) -> None:
        if isinstance(self.eval_metrics, Metric):
            self.eval_metrics.reset()
        else:
            for metric in self.eval_metrics.values():
                metric.reset()

    def compute(self) -> Dict[str, Union[float, WBValue]]:
        metrics = {}
        if not self.reduce_loss_metrics_manually:
            for k, v in self.eval_metrics.items():
                if _is_hist(k):
                    metrics[k] = wandb.Histogram(v.compute().detach().cpu().numpy(), num_bins=100)
                else:
                    value = v.compute().item()
                    if math.isfinite(value):
                        metrics[k] = value
        else:
            # manually reduce metrics across distributed processes to avoid possible
            # deadlocks with torchmetrics
            if not dist.is_initialized():
                # Not in distributed mode, use local compute
                for k, v in self.eval_metrics.items():
                    if _is_hist(k):
                        metrics[k] = wandb.Histogram(v.compute().detach().cpu().numpy(), num_bins=100)
                    else:
                        value = v.compute().item()
                        if math.isfinite(value):
                            metrics[k] = value
                return metrics
            
            # In distributed mode, we need to ensure all ranks process the same metrics in the same order
            # First, gather all metric names from all ranks to ensure consistency
            all_metric_names = sorted(self.eval_metrics.keys())  # Sort to ensure consistent order
            
            for k in all_metric_names:
                v = self.eval_metrics[k]
                
                if _is_hist(k):
                    # Histograms need special handling - just use local compute for now
                    # TODO: Could concatenate histograms across ranks if needed
                    metrics[k] = wandb.Histogram(v.compute().detach().cpu().numpy(), num_bins=100)
                else:
                    # For MeanMetric, we need to sync the sum and count across all ranks
                    # MeanMetric stores: value (sum) and weight (count) 
                    # Get the accumulated value and weight from this rank
                    # If this rank doesn't have this metric, use zeros
                    local_value = v.mean_value if hasattr(v, 'mean_value') else torch.tensor(0.0, device=self.device)
                    local_weight = v.weight if hasattr(v, 'weight') else torch.tensor(0.0, device=self.device) 
                    # All-reduce both the value and weight across all ranks
                    # All ranks must participate in this collective operation
                    dist.all_reduce(local_value, op=dist.ReduceOp.SUM)
                    dist.all_reduce(local_weight, op=dist.ReduceOp.SUM) 
                    # Compute the global mean
                    # Only add to metrics if there's actually data (global weight > 0)
                    if local_weight > 0:
                        metrics[k] = (local_value / local_weight).item()
        
        return metrics

    def update(
        self,
        batch: Dict[str, torch.Tensor],
        model_out,
        cross_entropy_loss: torch.Tensor,
        zloss: torch.Tensor
    ) -> None:
        labels = model_out.labels if model_out.labels is not None else batch["labels"]
        pred = torch.argmax(model_out.logits, dim=-1)
        loss_masks = model_out.loss_masks if model_out.loss_masks is not None else batch["loss_masks"]
        if len(pred.shape) == 2:
            loss_masks = loss_masks * (loss_masks > 0)
            total_weight = loss_masks.sum()
            accuracy = ((pred.flatten() == labels.flatten()).float() * loss_masks.flatten()).sum().item()
        else:
            loss_masks = loss_masks.view(-1)
            valid = loss_masks > 0
            loss_masks = loss_masks[valid]
            total_weight = loss_masks.sum()
            accuracy = ((pred == labels.flatten()[valid]).float() * loss_masks).sum().item()

        self.eval_metrics["CrossEntropyLoss"].update(cross_entropy_loss/total_weight if total_weight != 0.0 else 0.0, total_weight)
        if zloss is not None:
            self.eval_metrics["ZLoss"].update(zloss/total_weight if total_weight != 0.0 else 0.0, total_weight)
        self.eval_metrics["Accuracy"].update(accuracy/total_weight if total_weight != 0.0 else 0.0, total_weight)

        if model_out.metrics is not None:
            for name, val in model_out.metrics.items():
                if _is_hist(name):
                    if name not in self.eval_metrics:
                        self.eval_metrics[name] = torchmetrics.CatMetric("error")
                    self.eval_metrics[name].update(val)
                else:
                    if name not in self.eval_metrics:
                        self.eval_metrics[name] = MeanMetric("ignore").to(cross_entropy_loss.device)
                    try:
                        if isinstance(val, tuple):
                            if val[1] > 0:
                                self.eval_metrics[name].update(val[0]/val[1], val[1])
                        else:
                            self.eval_metrics[name].update(val, 1)
                    except Exception as e:
                        e.add_note(f"Error processing metric {name}")
                        raise e


@dataclass
class LossDatasetEvaluator:
    """Evaluates a model on a dataset based on its loss and other forward-pass metrics"""
    label: str
    eval_loader: DataLoader
    evaluator: LossMetrics
    num_batches: Optional[int] = None
    console_log_interval: Optional[int] = None
    z_loss: Optional[float] = None
    save_data: Optional[SaveEvalDataConfig] = None
    response_logits_only: bool = False

    def run(
            self, 
            model, 
            device, 
            autocast_precision, 
            loss_fn=None, 
            pbar=False, 
            logger=None,
            cp_enabled=False
            ):
        # Reset metrics.
        self.evaluator.reset()
        if loss_fn is None:
            from olmo.train.trainer import cross_entropy_loss as loss_fn

        # Initialize data loader iterator.
        eval_batches = iter(self.eval_loader)

        # Adjust how many batches to evaluate on.
        num_eval_batches = self.num_batches
        if num_eval_batches > 0:
            try:
                num_eval_batches = min(num_eval_batches, len(self.eval_loader))
            except TypeError:
                # No defined length
                pass
            eval_batches = islice(eval_batches, num_eval_batches)

        # Run model over batches.
        viz_data = []
        with torch.inference_mode():
            for eval_step, batch in enumerate(tqdm(eval_batches, total=num_eval_batches, disable=not pbar)):
                if logger and eval_step % logger.log_interval == 0:
                    logger.log_evaluation(self.label, eval_step, num_eval_batches)
                batch = move_to_device(batch, device)
                response_mask = (batch["loss_masks"] > 0)
                labels = batch["labels"].long()
                loss_masks = batch["loss_masks"]
                with torch.autocast("cuda", enabled=True, dtype=autocast_precision):
                    if cp_enabled:
                        inputs = {k: v for k, v in batch.items() if k not in ["metadata"]}
                    else:
                        inputs = {k: v for k, v in batch.items() if k not in ["labels", "loss_masks", "metadata"]}
                    model_out = model(
                        **inputs,
                        response_mask=response_mask,
                        response_logits_only=self.response_logits_only
                    )
                logits = model_out.logits
                loss_masks = model_out.loss_masks if model_out.loss_masks is not None else loss_masks
                loss_masks = loss_masks * (loss_masks > 0)
                labels = model_out.labels if model_out.labels is not None else labels

                labels.masked_fill_(~(loss_masks > 0), -100)
                labels = labels.view(-1)
                loss_masks = loss_masks.view(-1)
                logits_for_loss = logits.to(torch.float32).view(-1, logits.size(-1)) # for numerical stability
                if self.response_logits_only:
                    # don't change response_mask shape directly because we will use it to save internal data
                    loss_masks = loss_masks[response_mask.view(-1)]
                    labels = labels[response_mask.view(-1)]
                ce_loss, z_loss = loss_fn(
                    logits_for_loss, labels, ignore_index=-100, reduction="none",
                    compute_z_loss=self.z_loss is not None, z_loss_scale=self.z_loss,
                )
                ce_loss = (ce_loss * loss_masks).sum()
                if z_loss is not None:
                    z_loss = (z_loss * loss_masks).sum()
                self.evaluator.update(batch, model_out, ce_loss, z_loss)

                # Maybe save internal data
                if self.save_data:
                    for i in range(len(response_mask)):
                        saved_data = {}
                        if self.save_data.example_metadata:
                            saved_data["example_metadata"] = batch["metadata"][i]
                        if self.save_data.post_processed_inputs:
                            saved_data["post_processed_inputs"] = {k: v[i].detach().cpu() for k, v in inputs.items()}
                        if self.save_data.model_internal_data:
                            saved_data["model_internal_data"] = {
                                k: (None if v is None else v[i].detach().cpu())
                                for k, v in model_out.internal.items()}
                        viz_data.append(saved_data)

                if self.console_log_interval and not pbar:
                    if eval_step + 1 == num_eval_batches or (eval_step + 1) % self.console_log_interval == 0:
                        log.info(f"[eval_step={eval_step + 1}/{num_eval_batches}]")

        if logger:
            logger.log_evaluation(self.label, num_eval_batches, num_eval_batches)
        if self.save_data:
            return self.evaluator.compute(), viz_data
        else:
            return self.evaluator.compute()


@dataclass
class LossDatasetEvaluatorConfig(BaseConfig):
    """Configuration for a loss evaluation"""

    label: Optional[str] = None
    """Label to use when logging"""

    data: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    """Data to evaluate on"""

    device_batch_size: int = 4
    """Batch size, can default to the eval batch set set in the global config"""

    subset_num_batches: Optional[int] = None
    """Number of matches to run on, if None use the entire dataset"""

    max_examples: Optional[int] = None
    """Max number of examples to run on, overrides `subset_num_batches`"""

    console_log_interval: Optional[int] = None
    """How often to log progress to console"""
    
    response_logits_only: bool = False
    """Only return logits for response tokens to save memory"""

    reduce_loss_metrics_manually: bool = False
    """Whether to manually reduce metrics across distributed processes to avoid possible deadlocks with torchmetrics"""

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        config = config.copy()
        if getattr(config, "mm_evaluator", None):
            config.generative_evaluator = LossDatasetEvaluatorConfig.update_legacy_settings(config.generative_evaluator)
        if getattr(config, "data", None):
            config.data = DataLoaderConfig.update_legacy_settings(config.data)
        return config

    def build_dataset_evaluator(
            self, 
            model_config: MolmoConfig, 
            mesh: DeviceMesh,
            device, 
            save_data: SaveEvalDataConfig=None) -> LossDatasetEvaluator:
        eval_loader = self.data.build_eval_dataloader(
            model_config=model_config, 
            mesh=mesh,
            batch_size=self.device_batch_size,
            for_inference=False,
            include_metadata=save_data and save_data.example_metadata
        )

        if self.max_examples is not None:
            num_batches = max(1, self.max_examples // (self.device_batch_size*get_world_size()))
        elif self.subset_num_batches is not None:
            num_batches = self.subset_num_batches
        else:
            num_batches = len(eval_loader)

        return LossDatasetEvaluator(
            label=self.label,
            eval_loader=eval_loader,
            evaluator=LossMetrics(device, reduce_loss_metrics_manually=self.reduce_loss_metrics_manually),
            num_batches=num_batches,
            console_log_interval=self.console_log_interval,
            save_data=save_data,
            response_logits_only=self.response_logits_only
        )
