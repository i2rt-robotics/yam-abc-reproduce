from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import tempfile
from os.path import join
from typing import Dict, List, Optional

from olmo.torchcodec_utils import verify_torchcodec_runtime

log = logging.getLogger(__name__)

_LEROBOT_ENV_FILE_NAMES = (
    "LEROBOT_STATS_BY_TAG",
    "LEROBOT_REPO_TO_TAG",
    "LEROBOT_TAG_METADATA",
)
_SUBPROCESS_ENV_STRIP_KEYS = (
    "BEAKER_EXPERIMENT_SPEC",
    "BEAKER_JOB_SPEC",
    "BEAKER_TASK_SPEC",
    "BEAKER_TASK_CONTEXT",
)
_SUBPROCESS_ENV_SIZE_FALLBACK_BYTES = 128 * 1024
_LEROBOT_ENV_DIR: Optional[str] = None

def _estimate_environment_size_bytes(env: Optional[Dict[str, str]] = None) -> int:
    source = os.environ if env is None else env
    return sum(len(str(key)) + len(str(value)) + 2 for key, value in source.items())


def _get_lerobot_env_dir() -> str:
    global _LEROBOT_ENV_DIR
    if _LEROBOT_ENV_DIR is None:
        _LEROBOT_ENV_DIR = tempfile.mkdtemp(prefix="lerobot_env_")
        atexit.register(shutil.rmtree, _LEROBOT_ENV_DIR, ignore_errors=True)
    return _LEROBOT_ENV_DIR


def _store_env_json_in_file(name: str, payload: object) -> str:
    env_dir = _get_lerobot_env_dir()
    path = join(env_dir, f"{name.lower()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.environ.pop(name, None)
    os.environ[f"{name}_PATH"] = path
    return path


def _prepare_subprocess_environment_for_training() -> None:
    removed: List[str] = []
    for key in _SUBPROCESS_ENV_STRIP_KEYS:
        value = os.environ.get(key)
        if value:
            removed.append(f"{key}({len(value)}B)")
            os.environ.pop(key, None)
    if removed:
        log.info("Removed large env vars before subprocess launch: %s", ", ".join(removed))

    env_size = _estimate_environment_size_bytes()
    if env_size > _SUBPROCESS_ENV_SIZE_FALLBACK_BYTES and "TORCHINDUCTOR_COMPILE_THREADS" not in os.environ:
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        log.warning(
            "Environment still large after trimming (%d bytes); setting TORCHINDUCTOR_COMPILE_THREADS=1 "
            "to avoid subprocess spawn failures during torch.compile.",
            env_size,
        )


def _run_torchcodec_preflight(model_cfg) -> None:
    loading_method = str(getattr(model_cfg.mm_preprocessor, "loading_method", "") or "")
    lerobot_backend = os.environ.get("LEROBOT_VIDEO_BACKEND", "pyav").strip() or "pyav"
    if not (loading_method.startswith("torchcodec") or lerobot_backend == "torchcodec"):
        return

    info = verify_torchcodec_runtime()
    log.info("TorchCodec preflight passed: %s", json.dumps(info, sort_keys=True))


def _set_lerobot_environment_from_args(args) -> None:
    os.environ["LEROBOT_N_OBS_STEPS"] = str(args.n_obs_steps)
    os.environ["LEROBOT_MAX_ACTION_HORIZON"] = str(args.max_action_horizon)
    os.environ.pop("LEROBOT_ACTION_HORIZON", None)
    max_action_dim = getattr(args, "max_action_dim", getattr(args, "action_dim", 7))
    os.environ["LEROBOT_MAX_ACTION_DIM"] = str(max_action_dim)
    os.environ["LEROBOT_IMAGE_RESIZE"] = str(args.img_resize)
    os.environ["LEROBOT_NORM_MODE"] = str(args.norm_mode)
    os.environ["LEROBOT_ACTION_FORMAT"] = str(args.action_format)
    os.environ["LEROBOT_STATE_FORMAT"] = str(args.state_format)
    os.environ["LEROBOT_DISCRETE_ACTION_TOKENIZER"] = str(args.discrete_action_tokenizer or "")
    os.environ["LEROBOT_ENABLE_DEPTH_REASONING"] = "1" if getattr(args, "enable_depth_reasoning", False) else "0"
    os.environ["LEROBOT_NUM_DEPTH_TOKENS_PER_IMAGE"] = str(
        int(getattr(args, "num_depth_tokens_per_image", 100))
    )
    os.environ["LEROBOT_ADD_DEPTH_TOKENS"] = "1" if getattr(args, "add_depth_tokens", False) else "0"
    os.environ["LEROBOT_ADD_SETUP_TOKENS"] = "1" if getattr(args, "add_setup_tokens", False) else "0"
    os.environ["LEROBOT_ADD_CONTROL_TOKENS"] = "1" if getattr(args, "add_control_tokens", False) else "0"
    os.environ["LEROBOT_RANDOM_CAMERA_ORDER"] = str(args.random_camera_order)
    os.environ["LEROBOT_USE_ANNOTATED_TASK"] = "1" if args.use_annotated_task else "0"
    os.environ["LEROBOT_SAMPLE_ANNOTATED_TASK"] = "1" if args.sample_annotated_task else "0"
