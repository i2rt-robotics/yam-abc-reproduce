import logging
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, replace
from math import cos, pi, sqrt
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.optim import Optimizer

from olmo.config import BaseConfig, D, StrEnum
from olmo.torch_util import get_default_device, is_distributed, listinstr

try:
    from megablocks.layers.mlp import MLP, SparseMLP

    megablocks_available = True
except ImportError:
    megablocks_available = False

__all__ = [
    "LionW",
    "Scheduler",
    "CosWithWarmup",
    "LinearWithWarmup",
    "InvSqrtWithWarmup",
    "MaxScheduler",
    "ConstantScheduler",
    "BoltOnWarmupScheduler",
]


log = logging.getLogger(__name__)


class OptimizerType(StrEnum):
    lionw = "lionw"
    adamw = "adamw"


def _clean_param_name(name: str) -> str:
    return name.replace("_fsdp_wrapped_module.", "")


@dataclass
class OptimizerConfig(BaseConfig):
    PARAM_GROUP_FIELDS = ("sharded", "max_grad_norm", "max_grad_norm_ratio", "param_names", "group_name")
    name: OptimizerType = OptimizerType.lionw

    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1.0e-5
    """
    Default optimizer settings, used if the settings below are None
    """

    connector_learning_rate: Optional[float] = 1.0e-4
    vit_learning_rate: Optional[float] = 1.0e-4
    llm_learning_rate: Optional[float] = 1.0e-4
    frame_selector_learning_rate: Optional[float] = 1.0e-4
    temporal_token_scorer_learning_rate: Optional[float] = 1.0e-4
    action_expert_learning_rate: Optional[float] = 1.0e-4
    """
    Separate learning_rate values for the connector, vision backbone, and llm transformer.
    """

    connector_weight_decay: Optional[float] = 0.01
    vit_weight_decay: Optional[float] = 0.01
    llm_weight_decay: Optional[float] = 0.01
    frame_selector_weight_decay: Optional[float] = 0.01
    temporal_token_scorer_weight_decay: Optional[float] = 0.01
    action_expert_weight_decay: Optional[float] = 0.01
    """
    Separate weight decay values for the connector, vision backbone, and llm transformer.
    """

    connector_betas: Tuple[float, float] = (0.9, 0.95)
    vit_betas: Tuple[float, float] = (0.9, 0.95)
    llm_betas: Tuple[float, float] = (0.9, 0.95)
    frame_selector_betas: Tuple[float, float] = (0.9, 0.95)
    temporal_token_scorer_betas: Tuple[float, float] = (0.9, 0.95)
    action_expert_betas: Tuple[float, float] = (0.9, 0.95)
    """
    Separate betas values for the connector, vision backbone, and llm transformer.
    """

    connector_eps: Optional[float] = 1.0e-6
    vit_eps: Optional[float] = 1.0e-6
    llm_eps: Optional[float] = 1.0e-6
    frame_selector_eps: Optional[float] = 1.0e-6
    temporal_token_scorer_eps: Optional[float] = 1.0e-6
    action_expert_eps: Optional[float] = 1.0e-6
    """
    Separate weight decay values for the connector, vision backbone, and llm transformer.
    """

    metrics_log_interval: Optional[int] = -1
    """
    The interval with which to collect and log detailed parameter-specific metrics.
    This only applies when logging to W&B, since these metrics won't be logged to the console.
    If not set, defaults to the wandb `log_interval`
    """

    def __post_init__(self):
        self.betas = tuple(self.betas)  # type: ignore[assignment]
        self.connector_betas = tuple(self.connector_betas)  # type: ignore[assignment]
        self.vit_betas = tuple(self.vit_betas)  # type: ignore[assignment]
        self.llm_betas = tuple(self.llm_betas)  # type: ignore[assignment]

    def get_param_groups(self, max_grad_norm, max_grad_norm_ratio, model: nn.Module) -> List[Dict[str, Any]]:
        """
        Separate parameters into connector/vit/llm weight decay and non weight decay groups.
        """
        group_configs = [
            {
                "group_name": "llm",
                "params": [p for p in model.get_llm_parameters() if p.requires_grad],
                "lr": self.llm_learning_rate,
                "weight_decay": self.llm_weight_decay,
                "betas": self.llm_betas,
                "eps": self.llm_eps,
            },
            {
                "group_name": "connector",
                "params": [p for p in model.get_connector_parameters() if p.requires_grad],
                "lr": self.connector_learning_rate,
                "weight_decay": self.connector_weight_decay,
                "betas": self.connector_betas,
                "eps": self.connector_eps,
            },
            {
                "group_name": "vit",
                "params": [p for p in model.get_vit_parameters() if p.requires_grad],
                "lr": self.vit_learning_rate,
                "weight_decay": self.vit_weight_decay,
                "betas": self.vit_betas,
                "eps": self.vit_eps,
            },
            {
                "group_name": "frame_selector",
                "params": [p for p in model.get_frame_selection_parameters() if p.requires_grad],
                "lr": self.frame_selector_learning_rate,
                "weight_decay": self.frame_selector_weight_decay,
                "betas": self.frame_selector_betas,
                "eps": self.frame_selector_eps,
            },
            {
                "group_name": "temporal_token_scorer",
                "params": [p for p in model.get_temporal_token_scorer_parameters() if p.requires_grad],
                "lr": self.temporal_token_scorer_learning_rate,
                "weight_decay": self.temporal_token_scorer_weight_decay,
                "betas": self.temporal_token_scorer_betas,
                "eps": self.temporal_token_scorer_eps,
            },
            {
                "group_name": "action_expert",
                "params": [p for p in model.get_action_expert_parameters() if p.requires_grad],
                "lr": self.action_expert_learning_rate,
                "weight_decay": self.action_expert_weight_decay,
                "betas": self.action_expert_betas,
                "eps": self.action_expert_eps,
            },
        ]

        # Sanity check to make sure the `get_parameters` functions are doing the right thing
        param_names = {p: np for np, p in model.named_parameters() if p.requires_grad}
        params_found = set()
        for group_cfg in group_configs:
            if any(x in params_found for x in group_cfg["params"]):
                raise RuntimeError("A parameter appeared in multiple groups!")
            params_found.update(group_cfg["params"])
        for model_param, name in param_names.items():
            if model_param not in params_found:
                raise RuntimeError(f"model param {name} was not in any group")

        # Maybe split up groups if we are using weight decay on some of its params
        # Note its important to avoid creating empty optimizer groups since that will
        # trip up torch's distributed checkpointer
        non_weight_decay_params = set(model.get_non_weight_decay_params())
        param_groups = []
        for param_group in group_configs:
            params = param_group["params"]
            params.sort(key=lambda x: _clean_param_name(param_names[x]))
            if param_group["weight_decay"] == 0:
                if len(params) > 0:
                    param_groups.append(param_group)
            else:
                no_wd_params = [p for p in params if p in non_weight_decay_params]
                wd_params = [p for p in params if p not in non_weight_decay_params]
                if len(wd_params) > 0:
                    param_groups.append(dict(param_group, params=wd_params))
                if len(no_wd_params) > 0:
                    param_groups.append(dict(param_group, params=no_wd_params, weight_decay=0))
        return param_groups

    def build_optimizer(self, max_grad_norm, max_grad_norm_ratio, model: nn.Module) -> Optimizer :
        param_groups = self.get_param_groups(max_grad_norm, max_grad_norm_ratio, model)

        log.info(f"Constructing optimizer with {len(param_groups)} param groups")
        if self.name == OptimizerType.lionw:
            return LionW(
                param_groups,
                lr=self.learning_rate,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.name == OptimizerType.adamw:
            return torch.optim.AdamW(
                param_groups,
                lr=self.learning_rate,
                betas=self.betas,
                weight_decay=self.weight_decay,
                eps=self.eps,
            )
        else:
            raise NotImplementedError


class SchedulerType(StrEnum):
    cosine_with_warmup = "cosine_with_warmup"
    linear_with_warmup = "linear_with_warmup"
    inverse_sqrt_with_warmup = "inverse_sqrt_with_warmup"
    max_scheduler = "max_scheduler"
    constant = "constant"
    multimodal = "multimodal"


class SchedulerUnits(StrEnum):
    steps = "steps"
    tokens = "tokens"


class LionW(Optimizer):
    """
    Adapted from https://github.com/google/automl/blob/master/lion/lion_pytorch.py
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            group["initial_lr"] = group["lr"]
        self._update_total_dot_prod: Optional[torch.Tensor] = None
        self._update_total_norm: Optional[torch.Tensor] = None
        self._signed_update_total_norm: Optional[torch.Tensor] = None

    @torch.no_grad()
    def step(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        update_total_dot_prod = torch.tensor(0.0, dtype=torch.float32)
        update_norms = []
        signed_update_norms = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform step weight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                signed_update = torch.sign(update)
                p.add_(signed_update, alpha=-group["lr"])

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

                # Track dot product and norms of update vs signed update in order to calculate
                # their cosine similarity.
                update_total_dot_prod = update_total_dot_prod.to(update.device)
                update_total_dot_prod += torch.tensordot(update, signed_update, dims=len(update.shape))
                update_norms.append(torch.linalg.vector_norm(update, 2.0, dtype=torch.float32))
                signed_update_norms.append(torch.linalg.vector_norm(signed_update, 2.0, dtype=torch.float32))

        # Compute cosine similarity between update and signed update.
        self._update_total_dot_prod = update_total_dot_prod.to(get_default_device())
        self._update_total_norm = torch.linalg.vector_norm(
            torch.stack(update_norms),
            2.0,
            dtype=torch.float32,
        ).to(get_default_device())
        self._signed_update_total_norm = torch.linalg.vector_norm(
            torch.stack(signed_update_norms),
            2.0,
            dtype=torch.float32,
        ).to(get_default_device())


@dataclass
class Scheduler(metaclass=ABCMeta):
    # NOTE: these fields are not given default values because otherwise dataclasses complains
    # about how the scheduler subclasses are defined.
    grad_clip_warmup_steps: Optional[int]
    grad_clip_warmup_factor: Optional[float]
    warmup_min_lr: Optional[float]

    @abstractmethod
    def get_lr(self, initial_lr: float, step: int, max_steps: int, group_name: str=None) -> float:
        raise NotImplementedError

    def _get_max_grad_norm_coeff(
        self, initial_value: Optional[float], step: int, max_steps: int
    ) -> Optional[float]:
        del max_steps  # might need this in the future, but for now I just wanted to match the API of `get_lr()`.
        if initial_value is None:
            return None
        elif (
            self.grad_clip_warmup_steps is None
            or self.grad_clip_warmup_factor is None
            or step > self.grad_clip_warmup_steps
        ):
            return initial_value
        else:
            return self.grad_clip_warmup_factor * initial_value

    def get_max_grad_norm(
        self, initial_max_grad_norm: Optional[float], step: int, max_steps: int
    ) -> Optional[float]:
        return self._get_max_grad_norm_coeff(initial_max_grad_norm, step, max_steps)

    def get_max_grad_norm_ratio(
        self, initial_max_grad_norm_ratio: Optional[float], step: int, max_steps: int
    ) -> Optional[float]:
        return self._get_max_grad_norm_coeff(initial_max_grad_norm_ratio, step, max_steps)

    def _linear_warmup(self, initial_lr: float, step: int, warmup_steps: int = 2000) -> float:
        warmup_min_lr = self.warmup_min_lr if self.warmup_min_lr is not None else initial_lr * 0.10
        assert 0 <= warmup_min_lr < initial_lr
        return warmup_min_lr + (initial_lr - warmup_min_lr) * min(step, warmup_steps) / warmup_steps


@dataclass
class CosWithWarmup(Scheduler):
    warmup_steps: int
    alpha_f: float = 0.1
    t_max: Optional[int] = None

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f
        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        elif step >= max_steps:
            return eta_min
        else:
            step = step - self.warmup_steps
            max_steps = max_steps - self.warmup_steps
            return eta_min + (initial_lr - eta_min) * (1 + cos(pi * step / max_steps)) / 2


@dataclass
class LinearWithWarmup(Scheduler):
    warmup_steps: int
    alpha_f: float = 0.1
    t_max: Optional[int] = None

    def get_lr(self, initial_lr: float, step: int, max_steps: int, group_name=None) -> float:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f
        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        elif step >= max_steps:
            return eta_min
        else:
            step = step - self.warmup_steps
            max_steps = max_steps - self.warmup_steps
            return initial_lr - (initial_lr - eta_min) * (step / max_steps)


@dataclass
class InvSqrtWithWarmup(Scheduler):
    warmup_steps: int

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        del max_steps
        return initial_lr * sqrt(self.warmup_steps / max(self.warmup_steps, step))


@dataclass
class MaxScheduler(Scheduler):
    sched1: Scheduler
    sched2: Scheduler

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        return max(
            self.sched1.get_lr(initial_lr, step, max_steps), self.sched2.get_lr(initial_lr, step, max_steps)
        )


@dataclass
class BoltOnWarmupScheduler(Scheduler):
    inner: Scheduler
    warmup_start: int
    warmup_end: int

    @classmethod
    def wrap(cls, scheduler: Scheduler, warmup_start: int, warmup_end: int) -> "BoltOnWarmupScheduler":
        return cls(
            grad_clip_warmup_steps=None,
            grad_clip_warmup_factor=None,
            inner=scheduler,
            warmup_start=warmup_start,
            warmup_end=warmup_end,
            warmup_min_lr=None,
        )

    def get_lr(self, initial_lr: float, step: int, max_steps: int, group_name=None) -> float:
        if step < self.warmup_start:
            return 0.0
        if step < self.warmup_end:
            lr_at_intercept = self.inner.get_lr(initial_lr, self.warmup_end, max_steps, group_name)
            return lr_at_intercept * (step - self.warmup_start) / (self.warmup_end - self.warmup_start)
        else:
            return self.inner.get_lr(initial_lr, step, max_steps, group_name)

    def _get_max_grad_norm_coeff(
        self, initial_value: Optional[float], step: int, max_steps: int
    ) -> Optional[float]:
        return self.inner._get_max_grad_norm_coeff(initial_value, step, max_steps)


@dataclass
class ConstantScheduler(Scheduler):
    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        del step, max_steps
        return initial_lr


@dataclass
class MultimodalScheduler(Scheduler):
    connector_scheduler: Scheduler
    vit_scheduler: Scheduler
    llm_scheduler: Scheduler
    frame_selector_scheduler: Scheduler
    temporal_token_scorer_scheduler: Scheduler
    action_expert_scheduler: Scheduler

    def get_lr(self, initial_lr: float, step: int, max_steps: int, group_name: str) -> float:
        if group_name.startswith("connector"):
            return self.connector_scheduler.get_lr(initial_lr, step, max_steps)
        elif group_name.startswith("vit"):
            return self.vit_scheduler.get_lr(initial_lr, step, max_steps)
        elif group_name.startswith("llm"):
            return self.llm_scheduler.get_lr(initial_lr, step, max_steps)
        elif group_name.startswith("frame_selector"):
            return self.frame_selector_scheduler.get_lr(initial_lr, step, max_steps)
        elif group_name.startswith("temporal_token_scorer"):
            return self.temporal_token_scorer_scheduler.get_lr(initial_lr, step, max_steps)
        elif group_name.startswith("action_expert"):
            return self.action_expert_scheduler.get_lr(initial_lr, step, max_steps)
        else:
            raise ValueError(f"Unknown group name: {group_name}")


@dataclass
class SchedulerConfig(BaseConfig):
    name: SchedulerType = SchedulerType.cosine_with_warmup
    units: SchedulerUnits = SchedulerUnits.steps
    t_warmup: Union[int, float] = 100
    t_max: Optional[Union[int, float]] = None
    alpha_f: float = 0.1

    connector_t_warmup: Union[int, float] = 200
    vit_t_warmup: Union[int, float] = 200
    llm_t_warmup: Union[int, float] = 200
    frame_selector_t_warmup: Union[int, float] = 200
    temporal_token_scorer_t_warmup: Union[int, float] = 200
    action_expert_t_warmup: Union[int, float] = 200
    """
    Per-parameter group warmups
    """

    grad_clip_warmup_steps: Optional[Union[int, float]] = None
    """
    The warmup period for which the max grad norm (or norm ratio) will be set to its
    warmup value of `max_grad_norm * grad_clip_warmup_factor`.
    """

    grad_clip_warmup_factor: Optional[float] = None
    """
    The ratio of the max allowed gradient norm (or norm ratio) for clipping during the warmup period
    vs after the warmup period.
    """

    warmup_min_lr: Optional[float] = None
    """
    The starting LR during the warmup period. If not set this defaults to 10% of
    the target LR.
    """

    def build(self) -> MultimodalScheduler:
        connector_sched = CosWithWarmup(
            grad_clip_warmup_steps=None
            if self.grad_clip_warmup_steps is None
            else int(self.grad_clip_warmup_steps),
            grad_clip_warmup_factor=self.grad_clip_warmup_factor,
            warmup_steps=int(self.connector_t_warmup),
            alpha_f=self.alpha_f,
            t_max=None if self.t_max is None else int(self.t_max),
            warmup_min_lr=self.warmup_min_lr,
        )
        vit_sched = CosWithWarmup(
            grad_clip_warmup_steps=None
            if self.grad_clip_warmup_steps is None
            else int(self.grad_clip_warmup_steps),
            grad_clip_warmup_factor=self.grad_clip_warmup_factor,
            warmup_steps=int(self.vit_t_warmup),
            alpha_f=self.alpha_f,
            t_max=None if self.t_max is None else int(self.t_max),
            warmup_min_lr=self.warmup_min_lr,
        )
        llm_sched = CosWithWarmup(
            grad_clip_warmup_steps=None
            if self.grad_clip_warmup_steps is None
            else int(self.grad_clip_warmup_steps),
            grad_clip_warmup_factor=self.grad_clip_warmup_factor,
            warmup_steps=int(self.llm_t_warmup),
            alpha_f=self.alpha_f,
            t_max=None if self.t_max is None else int(self.t_max),
            warmup_min_lr=self.warmup_min_lr,
        )
        frame_selector_scheduler = CosWithWarmup(
            grad_clip_warmup_steps=None if self.grad_clip_warmup_steps is None else int(self.grad_clip_warmup_steps),
            grad_clip_warmup_factor=self.grad_clip_warmup_factor,
            warmup_steps=int(self.frame_selector_t_warmup),
            alpha_f=self.alpha_f,
            t_max=None if self.t_max is None else int(self.t_max),
            warmup_min_lr=self.warmup_min_lr,
        )
        temporal_token_scorer_scheduler = CosWithWarmup(
            grad_clip_warmup_steps=None if self.grad_clip_warmup_steps is None else int(self.grad_clip_warmup_steps),
            grad_clip_warmup_factor=self.grad_clip_warmup_factor,
            warmup_steps=int(self.temporal_token_scorer_t_warmup),
            alpha_f=self.alpha_f,
            t_max=None if self.t_max is None else int(self.t_max),
            warmup_min_lr=self.warmup_min_lr,
        )
        action_expert_scheduler = CosWithWarmup(
            grad_clip_warmup_steps=None if self.grad_clip_warmup_steps is None else int(self.grad_clip_warmup_steps),
            grad_clip_warmup_factor=self.grad_clip_warmup_factor,
            warmup_steps=int(self.action_expert_t_warmup),
            alpha_f=self.alpha_f,
            t_max=None if self.t_max is None else int(self.t_max),
            warmup_min_lr=self.warmup_min_lr,
        )

        return MultimodalScheduler(
            grad_clip_warmup_steps=None if self.grad_clip_warmup_steps is None else int(self.grad_clip_warmup_steps),
            grad_clip_warmup_factor=self.grad_clip_warmup_factor,
            warmup_min_lr=self.warmup_min_lr,
            connector_scheduler=connector_sched,
            vit_scheduler=vit_sched,
            llm_scheduler=llm_sched,
            frame_selector_scheduler=frame_selector_scheduler,
            temporal_token_scorer_scheduler=temporal_token_scorer_scheduler,
            action_expert_scheduler=action_expert_scheduler,
        )


