"""Run this script with 'torchrun'."""
import dataclasses
import json
import logging
import os
import pickle
import re
import sys
from copy import deepcopy
from datetime import datetime
from os.path import dirname, join
from typing import Optional, List, Tuple, Dict

import omegaconf
import torch
import wandb
from beaker import Beaker
from omegaconf import OmegaConf as om
from torch import distributed as dist

from olmo.config import BaseConfig
from olmo.data.data_loader import DataLoaderConfig
from olmo.eval.inf_evaluator import InfDatasetEvaluator, EvaluatorConfig, \
    InfDatasetEvaluatorConfig
from olmo.eval.loss_evaluator import LossDatasetEvaluatorConfig, LossDatasetEvaluator, \
    SaveEvalDataConfig, _is_hist
from olmo.exceptions import OLMoCliError
from olmo.io import file_exists, write_file, get_bytes_range, read_file
from olmo.model_configs import DEBUG_MOLMO
from olmo.models.model_config import BaseModelConfig
from olmo.models.molmo.molmo_preprocessor import MolmoPreprocessorConfig
from olmo.models.molmo2.molmo2 import Molmo2Config
from olmo.models.molmo2.molmo2_preprocessor import Molmo2PreprocessorConfig
from olmo.nn.beam_search import SamplingConfig
from olmo.nn.image_vit import VitConfig
from olmo.nn.llm import LlmConfig
from olmo.models.molmo.molmo import Molmo, MolmoConfig
from olmo.nn.vision_backbone import MolmoVisionBackboneConfig
from olmo.tokenizer import TokenizerConfig
from olmo.torch_util import (
    barrier,
    get_global_rank,
    get_local_rank,
    peak_gpu_memory,
    seed_all, get_world_size, )
from olmo.train.checkpointer import load_model_state
from olmo.train.trainer import BeakerLogger
from olmo.train.trainer_config import FSDPConfig
from olmo.util import (
    resource_path, log_metrics_to_console, prepare_torchrun_environment, clean_opt, flatten_lists,
)

log = logging.getLogger(__name__)


def get_float_dtype_by_name(dtype):
    return {
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
        'fp16': torch.float16,
        'float16': torch.float16,
        'fp32': torch.float32,
        'float32': torch.float32,
        'fp64': torch.float64,
        'float64': torch.float64,
    }[dtype]


def cast_float_dtype(t: torch.Tensor, dtype: str):
    if t.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        t = t.to(get_float_dtype_by_name(dtype))
    return t


def get_gcs_url(output_file):
    assert output_file.startswith("gs://")
    return f"https://storage.cloud.google.com/{output_file[5:]}l"


@dataclasses.dataclass
class DatasetEvaluatorConfig(BaseConfig):
    """Configuration for an offline dataset evaluation, it could be for loss or generation"""

    label: str = omegaconf.MISSING

    data: DataLoaderConfig = dataclasses.field(default_factory=DataLoaderConfig)
    """Data to evaluate on"""

    device_batch_size: Optional[int] = None
    """Batch size"""

    subset_num_batches: Optional[int] = None
    """Number of matches to run on, if None use the entire dataset"""

    max_examples: Optional[int] = None
    """Max number of examples to run on, overrides `subset_num_batches`"""

    max_new_tokens: Optional[int] = 448
    """Max number of tokens to generate"""

    sampling: SamplingConfig = dataclasses.field(default_factory=SamplingConfig)

    generative_evaluator: Optional[EvaluatorConfig] = None
    """Specifies how to compute metrics and save the predictions if doing a generative eval"""

    save_data: Optional[SaveEvalDataConfig] = None
    """
    Save low-level inputs/outputs, these can be used to make visualizations, but can also occupy 
    a lot of disk space
    """

    model_config_override: Optional[BaseModelConfig] = None

    eval_name: Optional[str] = None
    
    response_logits_only: bool = False

    enable_profiling: bool = False
    """Enable detailed profiling of the generation process"""

    @property
    def generative(self):
        return self.generative_evaluator is not None

    def build_evaluator(
            self, 
            model_config, 
            device, 
            default_save_dir, 
            console_log_interval, 
            include_image=False,
            mesh=None
        ):
        if self.generative:
            if self.save_data:
                raise NotImplementedError()
            cfg = InfDatasetEvaluatorConfig(
                self.label, self.data, self.generative_evaluator,
                max_new_tokens=self.max_new_tokens,
                device_batch_size=self.device_batch_size,
                subset_num_batches=self.subset_num_batches,
                max_examples=self.max_examples,
                console_log_interval=console_log_interval,
                include_image=include_image,
                sampling=self.sampling,
            )
            return cfg.build_dataset_evaluator(
                model_config=model_config,
                mesh=mesh,
                device=device,
                default_save_dir=default_save_dir
            )
        else:
            cfg = LossDatasetEvaluatorConfig(
                self.label, self.data,
                device_batch_size=self.device_batch_size,
                subset_num_batches=self.subset_num_batches,
                max_examples=self.max_examples,
                console_log_interval=console_log_interval,
                response_logits_only=self.response_logits_only,
            )
            return cfg.build_dataset_evaluator(model_config=model_config, device=device, save_data=self.save_data, mesh=None)


@dataclasses.dataclass
class EvalConfig(BaseConfig):
    """Configuration for an offline dataset evaluation, it could be for loss or generation"""

    evaluations: List[DatasetEvaluatorConfig] = dataclasses.field(default_factory=list)
    """Inference Evaluation configurations."""

    load_path: str = ""
    """The directory to load the model from"""

    model: Optional[BaseModelConfig] = None
    """Model config to use, load the one in the `load_path`` if None"""
    console_log_interval: int = 10
    """How often to log what step we are on to console"""

    fsdp: Optional[FSDPConfig] = None
    """Runs models with FSPD, needed for large models or large sequence lengths to reduce memory"""

    precision: Optional[str] = "fp32"
    """Autocase precision"""

    pbar: bool = True
    """Whether to show a tqdm progress bar"""

    seed: int = 6198
    """Random seed to torch/numpy, typically will not effect results"""

    save_to_checkpoint_dir: Optional[bool] = False
    """Use the checkpoint directory as `self.save_dir`"""

    eval_name: Optional[str] = None
    """Name to post-fix the evaluation outputs with"""

    skip_if_metrics_cached: bool = True
    """Skip a the metric file already exists in the save location, otherwise override it"""

    save_dir: Optional[str] = None
    """Where to save prediction, metrics, and visualizations"""

    include_image: bool = False
    """Include the image in the evaluation outputs"""

    beaker_log_interval: Optional[int] = 10

    load_bf16: bool = False
    """Load the model in bfloat16 if possible, this is usually safe for eval"""

    @property
    def autocast_precision(self) -> torch.dtype:
        if self.precision == "amp_bf16":
            return torch.bfloat16
        elif self.precision == "amp_fp16":
            return torch.float16
        elif self.precision == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Unexpected precision type '{self.precision}'")

    def build(self) -> 'ModelEvaluator':
        return ModelEvaluator(self)


@dataclasses.dataclass
class ModelEvaluator:
    """Evaluates a model on multiple datasets"""
    config: 'EvalConfig'

    def get_save_dir(self, cfg: 'DatasetEvaluatorConfig') -> Optional[str]:
        """Get directory to save the eval results"""
        if not self.config.save_dir and not self.config.save_to_checkpoint_dir:
            return None

        if self.config.save_to_checkpoint_dir:
            base = dirname(self.config.load_path)
        else:
            base = self.config.save_dir

        # If the load path has a step indicator, use it in the save dir name
        step_match = re.match(".*/(?:step|bk)([0-9]+).*",  self.config.load_path)
        if step_match is not None:
            step = int(step_match.group(1))
        else:
            step = None
        mixture_or_task_name = cfg.data.dataset
        if '=' in mixture_or_task_name: # remove filename, usually followed by '='
            mixture_or_task_name = mixture_or_task_name.split('=')[0]
        split = cfg.data.split
        if cfg.generative:
            name = f"predictions"
        else:
            name = f"loss"
        if step is not None:
            name = f"{name}-ck{step}-{mixture_or_task_name}-{split}"
        else:
            name = f"{name}-{mixture_or_task_name}-{split}"
        if cfg.eval_name:
            name += "-" + cfg.eval_name
        elif self.config.eval_name:
            name += "-" + self.config.eval_name
        default_prediction_dir = join(base, name)
        return default_prediction_dir

    def get_metric_file(self, cfg: 'DatasetEvaluatorConfig'):
        save_dir = self.get_save_dir(cfg)
        if save_dir:
            return join(self.get_save_dir(cfg), "metrics.json")
        else:
            return None

    def initialize_and_load_model(self) -> Molmo:
        cfg = self.config
        torch.cuda.set_device(f"cuda:{get_local_rank()}")
        device = torch.device("cuda")

        if cfg.load_path == "debug":
            assert not self.config.fsdp
            logging.warning("Loading debugging model")
            if (model_cfg := self.config.model) is None:
                model_cfg = MolmoConfig(
                    LlmConfig(
                        d_model=128,
                        n_heads=2,
                        n_layers=1,
                        max_sequence_length=4096,
                        additional_vocab_size=128,
                        vocab_size=50280,
                        embedding_size=50304,
                        rope=True,
                        weight_tying=False,
                        tokenizer=TokenizerConfig(
                            identifier='allenai/OLMoE-1B-7B-0924'
                        )
                    ),
                    vision_backbone=MolmoVisionBackboneConfig(
                        vit=VitConfig(
                            image_num_layers=1, image_emb_dim=128,
                            image_num_heads=2, image_head_dim=64,
                            image_num_key_value_heads=2
                        )
                    ),
                    mm_preprocessor=MolmoPreprocessorConfig(crop_mode="resize")
                )
            with torch.device("meta"):
                model = model_cfg.build_model()
            model.to_empty(device=device)
            model.reset_parameters()
        elif cfg.load_path == "debug_video":
            assert not self.config.fsdp
            logging.warning("Loading debugging model")
            model_cfg = Molmo2Config(
                llm=DEBUG_MOLMO.llm,
                vision_backbone=MolmoVisionBackboneConfig(
                    vit=VitConfig(
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
                    )
                ),
                data_formatter=DEBUG_MOLMO.data_formatter,
                mm_preprocessor=Molmo2PreprocessorConfig(
                    pooling_h=2,
                    pooling_w=2,
                    max_frames=4,
                    max_crops=1
                )
            )
            with torch.device("meta"):
                model = model_cfg.build_model()
            model.to_empty(device=device)
            model.reset_parameters()
        else:
            if self.config.model:
                model_cfg = self.config.model
            else:
                model_cfg_path = resource_path(cfg.load_path, "config.yaml")
                model_cfg = MolmoConfig.load(model_cfg_path, key="model", validate_paths=False)
            with torch.device("meta"):
                model: Molmo = model_cfg.build_model()

            if self.config.fsdp:
                assert self.config.fsdp.fsdp2
                model.apply_fsdp2(**self.config.fsdp.get_fsd2_args(self.config.autocast_precision))

            model.to_empty(device=device)
            load_model_state(cfg.load_path, model)
            model.eval()
            if self.config.load_bf16:
                model.to(torch.bfloat16)
                logging.info("Converted model to bfloat16")
            torch.cuda.empty_cache()

        # Just in case the model is doing randomization even during eval
        seed_all(cfg.seed)

        dtype = model.transformer.wte.embedding.dtype
        log.info(f"Model weight dtype: {dtype}")
        log.info(f"Total number of parameters: {model.num_params():,d}")
        log.info(f"Number of non-embedding parameters: {model.num_params(include_embedding=False):,d}")
        log.info(f"Peak GPU Memory (MB) before FSDP: {int(peak_gpu_memory() or 0)}")
        barrier()
        return model, device

    def run(self):
        config = self.config
        assert len(config.evaluations) > 0

        # Load any metrics that were cached
        cfg_to_metrics = {}
        for cfg in config.evaluations:
            if self.config.skip_if_metrics_cached:
                metric_file = self.get_metric_file(cfg)
                if metric_file and file_exists(metric_file):
                    logging.info(f"Loading pre-computed metrics for {cfg.label} from {metric_file}")
                    if get_global_rank() == 0:
                        cfg_to_metrics[cfg.label] = json.loads(read_file(metric_file))["metrics"]
                    else:
                        # Still set with a empty dict to mark that this eval can can be skipped
                        cfg_to_metrics[cfg.label] = {}

        # Possibly return early if everything was cached
        if all(x.label in cfg_to_metrics for x in config.evaluations):
            logging.info("All metrics cached, checkpoint will not be loaded")
            all_metrics = {}
            for name, metrics in cfg_to_metrics.items():
                all_metrics.update({f"{name}/{k}": v for k, v in metrics.items()})
            to_print = {k: v for k, v in all_metrics.items() if isinstance(v, (int, float, str))}
            log_metrics_to_console("all-metrics", to_print)
            return all_metrics

        # Initialize the model
        model, device = self.initialize_and_load_model()

        # Maybe initialize beaker progress tracker
        beaker_logger = None
        if self.config.beaker_log_interval > 0 and "BEAKER_EXPERIMENT_ID" in os.environ:
            if "BEAKER_TOKEN" in os.environ:
                if get_global_rank() == 0:
                    experiment_id = os.environ["BEAKER_EXPERIMENT_ID"]
                    client = Beaker.from_env()
                    beaker_logger = BeakerLogger(client, experiment_id, self.config.beaker_log_interval)
                    beaker_logger.log_init()
            else:
                if "BEAKER_EXPERIMENT_ID" in os.environ:
                    logging.info(f"Beaker log interval set to {self.config.beaker_log_interval}, but beaker "
                                 f"token is missing, so beaker logging will be turned off")

        all_metrics = {}
        for eval_ix, evaluation in enumerate(config.evaluations):
            if evaluation.label in cfg_to_metrics:
                continue

            if evaluation.model_config_override:
                prev_config = deepcopy(model.config)
                model.config = evaluation.model_config_override

            if len(config.evaluations) == 1:
                logging.info(f"Starting inference {evaluation.label}")
            else:
                logging.info(f"Starting inference {evaluation.label} ({eval_ix+1}/{len(config.evaluations)})")

            metrics_file = self.get_metric_file(evaluation)
            if metrics_file and file_exists(metrics_file):
                if self.config.skip_if_metrics_cached:
                    raise ValueError(f"{metrics_file} exists but skip_if_metrics_cached is True")
                logging.warning(f"{metrics_file} already exists! File will be overwritten")

            save_dir = self.get_save_dir(evaluation)
            if evaluation.generative:
                evaluator: InfDatasetEvaluator = evaluation.build_evaluator(
                    model.config, device, save_dir, self.config.console_log_interval,
                    self.config.include_image)
                metrics = evaluator.run(
                    model, device,
                    autocast_precision=self.config.autocast_precision,
                    is_distributed=self.config.fsdp is not None,
                    pbar=self.config.pbar,
                    logger=beaker_logger
                )
            else:
                evaluator: LossDatasetEvaluator = evaluation.build_evaluator(
                    model.config, device, save_dir, self.config.console_log_interval)
                metrics = evaluator.run(
                    model, device,
                    autocast_precision=self.config.autocast_precision,
                    pbar=self.config.pbar,
                    logger=beaker_logger
                )

            if evaluation.save_data:
                metrics, saved_data = metrics
                if get_global_rank() == 0:
                    out = [None for _ in range(get_world_size())]
                    dist.gather_object(saved_data, out)
                    selection_data = flatten_lists(out)
                    if save_dir is not None:
                        logging.info(f"Saving eval data to {save_dir}")
                        os.makedirs(save_dir, exist_ok=True)
                        data = pickle.dumps(selection_data)
                        write_file(save_dir, "eval_data.pkl", data, True)
                else:
                    dist.gather_object(saved_data)
                    selection_data = None

            # Post-process the metrics by saving the wandb.Html outputs to disk
            if save_dir and get_global_rank() == 0:
                for k, v in list(metrics.items()):
                    if _is_hist(k):
                        # FIXME Ideally we would save the histogram as a PNG
                        del metrics[k]
                        continue
                    if isinstance(v, wandb.Html):
                        filename = f"{evaluation.label}-{k}.html"
                        write_file(save_dir, filename, v.html, True)
                        if save_dir.startswith("gs://"):
                            metrics[k] = get_gcs_url(join(save_dir, filename))
                        else:
                            metrics[k] = join(save_dir, filename)

            to_print = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
            if metrics_file and get_global_rank() == 0:
                to_save = dict(
                    metrics=metrics,
                    beaker_experiment_id=os.environ.get("BEAKER_EXPERIMENT_ID"),
                    date=datetime.now().strftime("%m/%d/%Y, %H:%M"),
                    eval_config=om.to_container(om.create(evaluation)),
                    model_config=om.to_container(om.create(model.config)),
                )
                write_file(save_dir, "metrics.json", json.dumps(to_save, indent=2), True)
            log_metrics_to_console(evaluation.label, to_print)
            cfg_to_metrics[evaluation.label] = metrics

            if evaluation.model_config_override:
                model.config = prev_config

        all_metrics = {}
        for name, metrics in cfg_to_metrics.items():
            all_metrics.update({f"{name}/{k}": v for k, v in metrics.items()})

        if len(config.evaluations) > 1:   # print aggregated metrics if doing multiple evaluations
            to_print = {k: v for k, v in all_metrics.items() if isinstance(v, (int, float, str))}
            log_metrics_to_console("all-metrics", to_print)
        return all_metrics


if __name__ == "__main__":
    prepare_torchrun_environment()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = EvalConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    cfg.build().run()
