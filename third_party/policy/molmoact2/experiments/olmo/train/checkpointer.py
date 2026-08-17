"""
Model checkpointer

Mostly from  https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/train/checkpoint.py
"""
from __future__ import annotations

import dataclasses
import logging
import os
import re
import tempfile
import time
from contextlib import contextmanager
from os.path import join
from pathlib import Path
from typing import Generator, Optional, Dict, Any, Union, ClassVar, Tuple, Callable

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from cached_path import cached_path
from omegaconf import OmegaConf
from torch import nn, nn as nn
from torch.distributed.checkpoint import state_dict as dist_cp_sd

from olmo.config import BaseConfig
from olmo.io import PathOrStr, dir_is_empty, normalize_path, is_url, clear_directory, upload, \
    list_directory, resource_path, write_file, file_exists
from olmo.torch_util import barrier, get_fs_local_rank, get_global_rank
from olmo.train.distributed_checkpointing import save_model_and_optim_state, \
    load_model_and_optim_state
from olmo.train.optim import Optimizer
from olmo.util import wait_for


log = logging.getLogger(__name__)
MODEL_FILENAME = "model.pt"
OPT_FILENAME = "optim.pt"


@torch.no_grad()
def load_model_state_unsharded(dir: PathOrStr, model: nn.Module):
    """
    Load model state in-place for unsharded chechpoint saved in `dir`,
    works for sharded and unsharded models
    """
    if get_global_rank() == 0:
        state_dict = torch.load(resource_path(dir, MODEL_FILENAME),
                                map_location="cpu", weights_only=True)
    else:
        state_dict = {}

    if not torch.distributed.is_initialized():
        model.load_state_dict(state_dict)
        return
    
    dist_cp_sd.set_model_state_dict(
        model=model,
        model_state_dict=state_dict,
        options=dist_cp_sd.StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
    )


def save_unsharded(dir: PathOrStr, model: nn.Module, optim: Optimizer,
                   config: BaseConfig, overwrite: bool = False):
    """
    Save model, optim, and other training state to a local or remote directory unsharded
    :warning This can be very slow if saving to a remote directory
    """
    sd_options = dist_cp_sd.StateDictOptions(full_state_dict=True, cpu_offload=True)
    state_dict = dist_cp_sd.get_model_state_dict(model, options=sd_options)
    if get_fs_local_rank() == 0:
        write_file(dir, MODEL_FILENAME, lambda f: torch.save(state_dict, f), overwrite)
        del state_dict
    barrier()
    if optim is not None:
        optim_dict = dist_cp_sd.get_optimizer_state_dict(model, optim, options=sd_options)
        if get_fs_local_rank() == 0:
            write_file(dir, OPT_FILENAME, lambda f: torch.save(optim_dict, f), overwrite)
            del optim_dict
        barrier()
    if get_fs_local_rank() == 0:
        write_file(dir, Checkpointer.CONFIG_FILENAME, OmegaConf.to_yaml(config, resolve=True), overwrite)
    return dir


def _slice_state_dict(full_state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    prefix = f"{prefix}."
    return {k[len(prefix):]: v for k, v in full_state.items() if k.startswith(prefix)}


def merge_and_save_unsharded(
    dir: PathOrStr,
    model: nn.Module,
    config: BaseConfig,
    checkpoint_dir_llm: PathOrStr,
    checkpoint_dir_vision: Optional[PathOrStr] = None,
    overwrite: bool = False,
    full_state: Optional[Dict[str, torch.Tensor]] = None,
) -> PathOrStr:
    """
    Merge LoRA adapters (saved separately for LLM/vision) into a full unsharded model checkpoint.

    This function must be called on all ranks. It performs distributed state-dict collection
    before rank0-only merge/write to keep collective ordering consistent.
    """
    from peft import PeftModel

    if full_state is None:
        sd_options = dist_cp_sd.StateDictOptions(full_state_dict=True, cpu_offload=True)
        full_state = dist_cp_sd.get_model_state_dict(model, options=sd_options)

    rank0 = get_global_rank() == 0
    if rank0:
        if not hasattr(config, "model") or not hasattr(config.model, "build_model"):
            raise ValueError("merge_and_save_unsharded requires config.model.build_model().")

        with torch.device("meta"):
            merged_model = config.model.build_model()
        merged_model.to_empty(device=torch.device("cpu"))

        transformer_state = _slice_state_dict(full_state, "transformer")
        vision_state = _slice_state_dict(full_state, "vision_backbone")
        non_vlm_state = {
            k: v
            for k, v in full_state.items()
            if not k.startswith("transformer.") and not k.startswith("vision_backbone.")
        }
        _, unexpected = merged_model.load_state_dict(non_vlm_state, strict=False)
        if unexpected:
            log.warning("Unexpected non-VLM keys while building merged LoRA checkpoint: %s", unexpected[:10])

        if not transformer_state:
            log.warning("No transformer weights found when saving merged LoRA checkpoint.")
        else:
            transformer_with_lora = PeftModel.from_pretrained(merged_model.transformer, checkpoint_dir_llm)
            transformer_with_lora.load_state_dict(transformer_state, strict=False)
            merged_model.transformer = transformer_with_lora.merge_and_unload()

        if getattr(merged_model, "vision_backbone", None) is not None and checkpoint_dir_vision is not None:
            if not os.path.exists(checkpoint_dir_vision):
                log.warning("Missing vision LoRA adapter dir for merged checkpoint: %s", checkpoint_dir_vision)
            elif not vision_state:
                log.warning("No vision backbone weights found when saving merged LoRA checkpoint.")
            else:
                vision_with_lora = PeftModel.from_pretrained(
                    merged_model.vision_backbone,
                    checkpoint_dir_vision,
                )
                vision_with_lora.load_state_dict(vision_state, strict=False)
                merged_model.vision_backbone = vision_with_lora.merge_and_unload()

        merged_state_dict = merged_model.state_dict()
        write_file(dir, MODEL_FILENAME, lambda f: torch.save(merged_state_dict, f), overwrite)
        write_file(dir, Checkpointer.CONFIG_FILENAME, OmegaConf.to_yaml(config, resolve=True), overwrite)
        del merged_state_dict
        del merged_model

    barrier()
    return dir


def is_unsharded_checkoint(dir: PathOrStr) -> bool:
    return file_exists(join(dir, MODEL_FILENAME))


def load_model_state(dir: PathOrStr, model: nn.Module, cfg: CheckpointerConfig = None):
    """
    Load model state in-place from `dir`

    Works for any combination of sharded/unshared checkpoints and sharded/unshared model
    """
    t0 = time.perf_counter()
    if is_unsharded_checkoint(dir):
        log.info(f"Loading model state from unsharded checkpoint {dir}...")
        load_model_state_unsharded(dir, model)
    else:
        log.info(f"Loading model state from sharded checkpoint {dir}...")
        Checkpointer(cfg or CheckpointerConfig()).load(
            dir, model,
            optim=None,
            load_optimizer_state=False,
            load_trainer_state=False
        )
    log.info(f"Done in {time.perf_counter()-t0:0.1f} seconds")


@dataclasses.dataclass
class CheckpointerConfig(BaseConfig):
    """Config for loading/saving sharded checkpoints"""

    save_thread_count: Optional[int] = None
    load_thread_count: Optional[int] = None
    pre_download: bool = False
    work_dir: Optional[str] = None
    throttle_uploads: bool = False

    def build(self, save_overwrite: bool):
        return Checkpointer(**self.asdict(), save_overwrite=save_overwrite)


@dataclasses.dataclass
class Checkpointer:
    CONFIG_FILENAME: ClassVar[str] = "config.yaml"
    CHECKPOINT_DIR: ClassVar[str] = "step{step}"

    save_overwrite: bool = False
    save_thread_count: Optional[int] = None
    load_thread_count: Optional[int] = None
    pre_download: bool = False
    work_dir: Optional[str] = None
    throttle_uploads: bool = False

    @classmethod
    def find_checkpoints(cls, dir: PathOrStr) -> Generator[Tuple[int, str], None, None]:
        """
        Find sharded checkpoints within a directory.
        """
        dir = normalize_path(dir)
        for path in list_directory(dir):
            name = os.path.basename(path)
            if (m := re.match("^" + cls.CHECKPOINT_DIR.format(step=r"(\d+)$"), name)) is not None:
                step = int(m.group(1))
                yield step, path

    @classmethod
    def contains_checkpoint(cls, dir: PathOrStr) -> bool:
        """
        Check if a directory is a sharded checkpoint directory or contains a child checkpoint directory.
        """
        try:
            next(cls.find_checkpoints(dir))
            return True
        except (StopIteration, FileNotFoundError):
            return False

    @classmethod
    def latest_checkpoint(cls, dir: PathOrStr) -> str:
        """
        Find the latest (sharded) checkpoint in a directory of checkpoints.

        :raises FileNotFoundError: If no checkpoints are found.
        """
        dir = normalize_path(dir)
        latest_step: Optional[int] = None
        latest_checkpoint: Optional[str] = None
        for step, path in cls.find_checkpoints(dir):
            if latest_step is None or step > latest_step:
                latest_step = step
                latest_checkpoint = path

        if latest_checkpoint is None:
            raise FileNotFoundError(f"No checkpoints found in '{dir}'")
        else:
            return latest_checkpoint

    def save(self, dir: PathOrStr, model: nn.Module, optim: Optimizer, train_state: Dict[str, Any],
             config: BaseConfig = None):
        """
        Save model, optim, and other training state to a local or remote directory.
        """
        dir = normalize_path(dir)
        with self._temporary_wd(dir) as wd:
            # Save trainer state.
            self._save_train_state(dir, wd, train_state)

            # Save model and optim state.
            model_and_optim_dir = (
                f"{dir}/model_and_optim" if is_url(dir) else wd / "model_and_optim"
            )
            save_model_and_optim_state(
                model_and_optim_dir,
                model,
                optim,
                save_overwrite=self.save_overwrite,
                thread_count=self.save_thread_count,
                throttle_uploads=self.throttle_uploads,
            )
        if get_fs_local_rank() == 0 and config is not None:
            self.write_file(dir, self.CONFIG_FILENAME, OmegaConf.to_yaml(config, resolve=True))

    def write_file(self, dir: PathOrStr, fname: str, contents: Union[str, bytes, Callable]) -> PathOrStr:
        return write_file(dir, fname, contents, self.save_overwrite)

    def load(
        self,
        dir: PathOrStr,
        model: nn.Module,
        optim: Optimizer = None,
        *,
        load_optimizer_state: Optional[bool] = None,
        load_trainer_state: Optional[bool] = None,
        key_mapping: Optional[Dict[str, str]] = None,
        allow_missing_keys: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Load model, optim, and other training state from a local or remote checkpoint directory
        created via :meth:`save()` or :meth:`save_async()`.
        """
        dir = normalize_path(dir)

        # Maybe load trainer state.
        trainer_state: Optional[Dict[str, Any]] = None
        if load_trainer_state is not False:
            # Try loading the given rank's state first, then fall back to rank 0 train state if it
            # doesn't exist, which can happen when we're restoring a checkpoint with a different world size.
            for path in (f"{dir}/train/rank{get_global_rank()}.pt", f"{dir}/train/rank0.pt"):
                try:
                    trainer_state = torch.load(cached_path(path, quiet=True), weights_only=False)
                    break
                except FileNotFoundError:
                    pass

            if load_trainer_state is True and trainer_state is None:
                raise FileNotFoundError(f"Missing trainer state in checkpoint dir '{dir}'")

        # Load model and optimizer state.
        model_and_optim_dir: str = f"{dir}/model_and_optim"
        load_model_and_optim_state(
            model_and_optim_dir,
            model,
            optim if load_optimizer_state else None,
            process_group=None,
            key_mapping=key_mapping,
            pre_download=is_url(dir) and self.pre_download,
            work_dir=self.work_dir,
            thread_count=self.load_thread_count,
            allow_missing_keys=allow_missing_keys
        )
        return trainer_state

    def _save_train_state(self, dir: PathOrStr, wd: Path, train_state: Dict[str, Any]):
        train_dir = wd / "train"
        # NOTE: if 'dir' is a URL, the 'wd' will be a different temp dir for each rank.
        if is_url(dir) or get_fs_local_rank() == 0:
            train_dir.mkdir(exist_ok=True, parents=True)
        wait_for(train_dir.exists, description=f"Waiting for '{train_dir}' to be created...")
        torch.save(train_state, train_dir / f"rank{get_global_rank()}.pt")

    def _get_tmp_dir(self, dir: PathOrStr) -> Path:
        # Prepare temporary directory.
        tmp_dir: Path
        if is_url(dir):
            tmp_dir = Path(tempfile.mkdtemp(dir=self.work_dir))
        else:
            tmp_dir = Path(dir).with_name(Path(dir).name + "-tmp")
            if get_fs_local_rank() == 0:
                clear_directory(tmp_dir)
                tmp_dir.mkdir(exist_ok=True, parents=True)

        # In the cases where we're using a shared NFS drive between ranks to save checkpoints,
        # creating the temp directory from rank 0 might not be immediately
        # realized in the file systems of the other ranks.
        # So we wait here across all ranks until that tmp checkpoint directory is visible.
        wait_for(lambda: tmp_dir.exists(), "Waiting for checkpoint directory", timeout=10.0)
        barrier()
        return tmp_dir

    def _prepare_dir(self, dir: PathOrStr, ensure_exists: bool = True) -> str:
        dir = normalize_path(dir)

        # Make sure checkpoint directory doesn't exist unless it's okay to overwrite it.
        if not dir_is_empty(dir):
            if self.save_overwrite:
                if get_fs_local_rank() == 0:
                    clear_directory(dir)
            else:
                raise FileExistsError(dir)

        if ensure_exists and not is_url(dir):
            if get_fs_local_rank() == 0:
                Path(dir).mkdir(exist_ok=True, parents=True)
            # Ensure the dir exists for all ranks before continuing. This might take a second if we're
            # saving to an NFS drive or something like that.
            wait_for(Path(dir).exists, description=f"Waiting on '{dir}' to be created...")

        barrier()
        return dir

    def _teardown_tmp_dir(self, dir: PathOrStr, tmp_dir: Path):
        if not is_url(dir):
            # Replace the temporary directory with the actual checkpoint directory.
            if get_fs_local_rank() == 0:
                # Replace temp directory with target checkpoint directory.
                try:
                    tmp_dir.replace(str(dir))
                except FileNotFoundError:
                    # Caught when another (file-system) local rank 0 has already replaced the tmp directory.
                    # This can happen when nodes are saving to a common NFS drive but otherwise have distinct
                    # file-systems.
                    if not Path(dir).exists():
                        raise

            # In the cases where we're using a shared NFS drive between ranks to save checkpoints,
            # replacing the temp directory with the final directory from rank 0 might not be immediately
            # realized in the file systems of the other ranks.
            # So we wait here across all ranks until that final checkpoint directory is visible.
            wait_for(lambda: Path(dir).exists(), "Waiting for checkpoint directory", timeout=10.0)
        else:
            # NOTE: each rank will have its own tmp dir
            # Upload files to final location.
            for path in tmp_dir.glob("**/*"):
                if not path.is_file():
                    continue
                upload(
                    path,
                    f"{dir}/{path.relative_to(tmp_dir)}",
                    save_overwrite=self.save_overwrite,
                )

            # Then remove the temp dir.
            clear_directory(tmp_dir)

    @contextmanager
    def _temporary_wd(self, dir: PathOrStr) -> Generator[Path, None, None]:
        # No need to mkdir here since we'll directly replace the temporary directory with
        # this directory below.
        dir = self._prepare_dir(dir, ensure_exists=False)

        tmp_dir = self._get_tmp_dir(dir)

        yield tmp_dir

        barrier()

        self._teardown_tmp_dir(dir, tmp_dir)
