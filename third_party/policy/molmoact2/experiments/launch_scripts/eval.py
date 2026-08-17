"""Evals a checkpoint on a task, run this script with 'torchrun'."""
import argparse
import logging
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omegaconf import OmegaConf

from olmo.eval.eval_utils import get_default_max_tokens, get_evaluator
from olmo.eval.save_eval_data_config import SaveEvalDataConfig
from olmo.models.molmo.molmo import MolmoConfig
from olmo.train.trainer_config import FSDPConfig
from olmo.data.data_loader import DataLoaderConfig
from olmo.util import clean_opt, prepare_torchrun_environment, select_checkpoint, resource_path
from olmo.eval.model_evaluator import ModelEvaluator, EvalConfig, DatasetEvaluatorConfig

log = logging.getLogger(__name__)


def main():
    prepare_torchrun_environment()

    parser = argparse.ArgumentParser(prog="Script to generate dense captions")
    parser.add_argument("checkpoint")
    parser.add_argument("--task", default="dense_caption_eval")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seq_len", default=None, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--max_examples", default=None, type=int)
    parser.add_argument("--device_batch_size", default=1, type=int)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--save_eval_data", action="store_true",
                        help="Save detailed inputs/intermediate model data for use in visualizations")
    parser.add_argument("--loss", action="store_true",
                        help="Compute loss/accuracy metrics instead of doing inference")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Override max new tokens, otherwise use task-specific default")
    parser.add_argument("--response_logits_only", action="store_true")
    args, other_args = parser.parse_known_args()

    checkpoint_dir = select_checkpoint(args.checkpoint)
    if args.fsdp:
        if args.seq_len is None:
            raise ValueError("Sequence length is required if using FSDP")

    if args.loss:
        max_new_tokens = None
    elif args.max_new_tokens:
        max_new_tokens = args.max_new_tokens
    else:
        max_new_tokens = get_default_max_tokens(args.task)
        log.info(f"Using default of {max_new_tokens} max tokens for task {args.task}")

    eval_config = DatasetEvaluatorConfig(
        data=DataLoaderConfig(
            args.task, split=args.split,
            sequence_length=args.seq_len,
            drop_last=False, seed=6198,
            shuffle=True if args.max_examples else False,
            pad=None,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
        device_batch_size=args.device_batch_size,
        max_new_tokens=max_new_tokens,
        generative_evaluator=None if args.loss else get_evaluator(args.task),
        save_data=SaveEvalDataConfig() if args.save_eval_data else None,
        label=args.task,
        max_examples=args.max_examples,
        response_logits_only=args.response_logits_only,
    )

    # Explicitly set the model config so model settings can be overriden by CLI args
    model_cfg_path = resource_path(select_checkpoint(checkpoint_dir), "config.yaml")
    model_cfg = MolmoConfig.load(model_cfg_path, key="model", validate_paths=False)

    cfg = EvalConfig(
        pbar=False,
        model=model_cfg,
        evaluations=[eval_config],
        load_path=checkpoint_dir,
        console_log_interval=10,
        fsdp=FSDPConfig(fsdp2=True) if args.fsdp else None,
        save_to_checkpoint_dir=args.save_dir is None,
        save_dir=args.save_dir,
    )

    config = OmegaConf.create(cfg)
    config.merge_with_dotlist([clean_opt(arg) for arg in other_args])
    cfg = OmegaConf.to_object(config)
    ModelEvaluator(cfg).run()


if __name__ == '__main__':
    main()
