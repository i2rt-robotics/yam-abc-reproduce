from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.optim.optimizers import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.constants import ACTION


@PreTrainedConfig.register_subclass("molmoact2")
@dataclass
class MolmoAct2Config(PreTrainedConfig):
    """
    Lightweight config wrapper for running MolmoAct2 policies inside LeRobot eval.

    Only inference is supported. We delegate action generation to an external MolmoAct2
    checkpoint (from the MolmoAct2 codebase). Required fields:
      - `checkpoint_path`: path to a MolmoAct2 checkpoint directory (unsharded preferred).

    Optional:
      - `seq_len`: max token length for the Molmo tokenizer/collator.
      - `num_steps`: flow-matching integration steps for action generation.
    """

    checkpoint_path: str = "allenai/MolmoAct2"
    seq_len: Optional[int] = None
    num_steps: Optional[int] = None
    # Inference action mode:
    # - "continuous": flow-matching action expert (generate_actions)
    # - "discrete": autoregressive token generation + discrete action decode
    inference_action_mode: str = "continuous"
    # Required when inference_action_mode="discrete".
    discrete_action_tokenizer: Optional[str] = "allenai/MolmoAct2-FAST-Tokenizer"
    enable_depth_reasoning: bool = False
    enable_inference_cuda_graph: bool = True
    num_depth_tokens_per_image: Optional[int] = None
    verbose: bool = False
    norm_tag: str = ""

    # Provide minimal feature metadata to satisfy the policy factory. These will be
    # overridden at runtime by `make_policy` if dataset/env features are available.
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=[7])}
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.inference_action_mode = str(self.inference_action_mode or "continuous").strip().lower()
        if self.inference_action_mode not in {"continuous", "discrete"}:
            raise ValueError(
                f"Unsupported inference_action_mode={self.inference_action_mode!r}. "
                "Expected one of {'continuous', 'discrete'}."
            )
        if self.seq_len is not None and self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1 or None, got {self.seq_len}.")

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return None

    @property
    def reward_delta_indices(self):
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError("MolmoAct2 config is inference-only.")

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    def validate_features(self) -> None:
        # Nothing to validate for inference-only wrapper.
        return

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.checkpoint_path)
