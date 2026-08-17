import hashlib
# import ujson as json
import json
import logging
import os
import re
import tarfile
import socket
import sys
import math
import threading
import uuid

import torch.multiprocessing as mp
import time
import warnings
from datetime import datetime, timedelta
from os.path import abspath, join, basename, dirname
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, List, Tuple

import numpy as np
import rich
from datasets import disable_progress_bar as disable_hf_datasets_progress_bar
from rich.console import Console, ConsoleRenderable
from rich.highlighter import NullHighlighter
from rich.progress import Progress
from rich.text import Text
from rich.traceback import Traceback

from .config import StrEnum
from .exceptions import (
    OLMoCliError,
    OLMoEnvironmentError,
    OLMoError,
)
from .io import add_cached_path_clients, PathOrStr, file_exists, dir_is_empty, list_directory
from .torch_util import get_global_rank, get_local_rank, get_node_rank, is_distributed, \
    barrier, init_process_group, synchronize_flag

try:
    from functools import cache, wraps
except ImportError:
    from functools import lru_cache as cache


OLMO_NUM_THREADS_ENV_VAR = "OLMO_NUM_THREADS"


def get_hf_access_token() -> Optional[str]:
    for env_var in ("HF_ACCESS_TOKEN", "HF_TOKEN"):
        token = os.environ.get(env_var)
        if token is not None:
            token = token.strip()
        if token:
            return token
    return None


def ensure_multiple_of(x: int, of: int) -> int:
    return of * math.ceil(x / of)


def compute_hash(data: str) -> str:
    """Computes the hash of a string."""
    if isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    else:
        return hashlib.sha256(data.encode("utf-8")).hexdigest()


def load_json(file):
    with open(file, "r") as f:
        return json.load(f)


_log_extra_fields: Dict[str, Any] = {}
log = logging.getLogger(__name__)


class LogFilterType(StrEnum):
    rank0_only = "rank0_only"
    local_rank0_only = "local_rank0_only"
    all_ranks = "all_ranks"


def log_extra_field(field_name: str, field_value: Any) -> None:
    global _log_extra_fields
    if field_value is None:
        if field_name in _log_extra_fields:
            del _log_extra_fields[field_name]
    else:
        _log_extra_fields[field_name] = field_value


def setup_logging(log_filter_type: LogFilterType = LogFilterType.rank0_only) -> None:
    """
    :param rank0_only: INFO and below messages will only be emitted on the rank0 process.
    """
    log_extra_field("hostname", socket.gethostname())
    if is_distributed():
        log_extra_field("node_rank", get_node_rank())
        log_extra_field("local_rank", get_local_rank())
        log_extra_field("global_rank", get_global_rank())
    else:
        log_extra_field("node_rank", 0)
        log_extra_field("local_rank", 0)
        log_extra_field("global_rank", 0)

    old_log_record_factory = logging.getLogRecordFactory()

    def log_record_factory(*args, **kwargs) -> logging.LogRecord:
        record = old_log_record_factory(*args, **kwargs)
        for field_name, field_value in _log_extra_fields.items():
            setattr(record, field_name, field_value)
        return record

    logging.setLogRecordFactory(log_record_factory)

    handler: logging.Handler
    if not is_interactive():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s\t%(hostname)s:%(local_rank)s\t%(name)s:%(lineno)s\t%(levelname)s\t%(message)s"
        )
        formatter.default_time_format = "%Y-%m-%d %H:%M:%S"
        formatter.default_msec_format = "%s.%03d"
        handler.setFormatter(formatter)
    else:
        handler = RichHandler()

    def rank0_filter(record: logging.LogRecord) -> int:
        if record.levelno > logging.INFO:
            return 1
        if getattr(record, "global_rank", 0) == 0:
            return 1
        else:
            return 0

    def local_rank0_filter(record: logging.LogRecord) -> int:
        if record.levelno > logging.INFO:
            return 1
        if getattr(record, "local_rank", 0) == 0:
            return 1
        else:
            return 0

    if log_filter_type == LogFilterType.rank0_only:
        filter = rank0_filter
    elif log_filter_type == LogFilterType.local_rank0_only:
        filter = local_rank0_filter  # type: ignore
    elif log_filter_type == LogFilterType.all_ranks:
        filter = None
    else:
        raise ValueError(log_filter_type)

    if filter is not None:
        handler.addFilter(filter)  # type: ignore
    # torch 2.6 will try setup some loggers of its own when we import, so
    # use `force` to use our logging settings instead
    logging.basicConfig(handlers=[handler], level=logging.INFO, force=True)

    file_handler = logging.FileHandler("debug.log")
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s\t%(hostname)s:%(local_rank)s\t%(name)s:%(lineno)s\t%(levelname)s\t%(message)s"
    )
    formatter.default_time_format = "%Y-%m-%d %H:%M:%S"
    formatter.default_msec_format = "%s.%03d"
    file_handler.setFormatter(formatter)
    logging.root.addHandler(file_handler)
    logging.captureWarnings(True)

    if os.environ.get("HF_DATASETS_OFFLINE"):
        # Stop HF warning us about every single dataset
        logging.getLogger("datasets.load").setLevel(logging.ERROR)
        logging.getLogger("datasets.packaged_modules.cache.cache").setLevel(logging.ERROR)
    # Calm down the requests/url notices
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger("google.resumable_media._helpers").setLevel(logging.WARNING)


def excepthook(exctype, value, traceback):
    """
    Used to patch `sys.excepthook` in order to log exceptions.
    """
    if issubclass(exctype, KeyboardInterrupt):
        sys.__excepthook__(exctype, value, traceback)
    elif issubclass(exctype, OLMoCliError):
        rich.get_console().print(f"[yellow]{value}[/]", highlight=False)
    elif issubclass(exctype, OLMoError):
        rich.get_console().print(Text(f"{exctype.__name__}:", style="red"), value, highlight=False)
    else:
        log.critical("Uncaught %s: %s", exctype.__name__, value, exc_info=(exctype, value, traceback))


def install_excepthook():
    sys.excepthook = excepthook


def filter_warnings():
    # Filter internal deprecation warnings from torch
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message="torch.distributed.*_base is a private function and will be deprecated.*",
    )
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message="TypedStorage is deprecated.*",
    )
    warnings.filterwarnings(
        action="ignore",
        category=FutureWarning,
        message="Please use DTensor instead.*",
    )
    warnings.filterwarnings(
        action="ignore",
        category=FutureWarning,
        message="You are using `torch.load` with `weights_only=False`.*",
    )
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message="`_get_pg_default_device` will be deprecated,",
    )
    warnings.filterwarnings(
        'ignore',
        category=UserWarning,
        module='google.protobuf.runtime_version'
    )
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message="The video decoding and encoding capabilities of torchvision are deprecated.*",
        module=r"torchvision\.io\._video_deprecation_warning",
    )


def flatten_lists(xss):
    return [x for xs in xss for x in xs]


def split_into_groups(lst, max_group_size):
    """ partition `lst` into that the mininal number of groups that as evenly sized
    as possible  and are at most `max_group_size` in size """
    if max_group_size is None:
        return [lst]
    if max_group_size == 1:
        return [[x] for x in lst]
    n_groups = (len(lst) + max_group_size - 1) // max_group_size
    per_group = len(lst) // n_groups
    remainder = len(lst) % n_groups
    groups = []
    ix = 0
    for _ in range(n_groups):
        group_size = per_group
        if remainder > 0:
            remainder -= 1
            group_size += 1
        groups.append(lst[ix:ix + group_size])
        ix += group_size
    return groups


def set_env_variables():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    setup_gcp_credentials()
    setup_s3_credentials()


def prepare_torchrun_environment():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")
    log.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")

    # This needs to happen first so `prepare_cli_environment` can access the global rank,
    # which is used for logging filters/fields
    init_process_group()

    if "TORCH_LOGS_RANK0" in os.environ:
        if get_local_rank() == 0:
            cur = os.environ.get("TORCH_LOGS")
            os.environ["TORCH_LOGS"] = (cur+"," if cur else "") + os.environ["TORCH_LOGS_RANK0"]

    add_cached_path_clients()
    prepare_cli_environment()
    log.info(f"Set up torchrun environment")


def is_interactive() -> bool:
    return not (
        os.environ.get("OLMo_NONINTERACTIVE", False)
        or os.environ.get("DEBIAN_FRONTEND", None) == "noninteractive"
        or not sys.stdout.isatty()
    )


def prepare_cli_environment(log_filter_type: Optional[LogFilterType] = None):
    if log_filter_type is None:
        log_filter_type = LogFilterType(os.environ.get("LOG_FILTER_TYPE", "rank0_only"))
    if (get_global_rank() != 0) or not is_interactive():
        disable_hf_datasets_progress_bar()

    rich.reconfigure(width=max(rich.get_console().width, 180), soft_wrap=True)
    setup_logging(log_filter_type=log_filter_type)
    install_excepthook()
    filter_warnings()
    set_env_variables()


def clean_opt(arg: str) -> str:
    if "=" not in arg:
        arg = f"{arg}=True"
    name, val = arg.split("=", 1)
    name = name.strip("-").replace("-", "_")
    return f"{name}={val}"


class RichHandler(logging.Handler):
    """
    A simplified version of rich.logging.RichHandler from
    https://github.com/Textualize/rich/blob/master/rich/logging.py
    """

    def __init__(
        self,
        *,
        level: Union[int, str] = logging.NOTSET,
        console: Optional[Console] = None,
        markup: bool = False,
    ) -> None:
        super().__init__(level=level)
        self.console = console or rich.get_console()
        self.highlighter = NullHighlighter()
        self.markup = markup

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if hasattr(record.msg, "__rich__") or hasattr(record.msg, "__rich_console__"):
                self.console.print(record.msg)
            else:
                msg: Any = record.msg
                if isinstance(record.msg, str):
                    msg = self.render_message(record=record, message=record.getMessage())
                renderables = [
                    self.get_time_text(record),
                    self.get_level_text(record),
                    self.get_location_text(record),
                    msg,
                ]
                if record.exc_info is not None:
                    tb = Traceback.from_exception(*record.exc_info)  # type: ignore
                    renderables.append(tb)
                self.console.print(*renderables)
        except Exception:
            self.handleError(record)

    def render_message(self, *, record: logging.LogRecord, message: str) -> ConsoleRenderable:
        use_markup = getattr(record, "markup", self.markup)
        message_text = Text.from_markup(message) if use_markup else Text(message)

        highlighter = getattr(record, "highlighter", self.highlighter)
        if highlighter:
            message_text = highlighter(message_text)

        return message_text

    def get_time_text(self, record: logging.LogRecord) -> Text:
        log_time = datetime.fromtimestamp(record.created)
        time_str = log_time.strftime("[%Y-%m-%d %X]")
        return Text(time_str, style="log.time", end=" ")

    def get_level_text(self, record: logging.LogRecord) -> Text:
        level_name = record.levelname
        level_text = Text.styled(level_name.ljust(8), f"logging.level.{level_name.lower()}")
        level_text.style = "log.level"
        level_text.end = " "
        return level_text

    def get_location_text(self, record: logging.LogRecord) -> Text:
        name_and_line = f"{record.name}:{record.lineno}" if record.name != "root" else "root"
        text = f"[{name_and_line}, rank={record.local_rank}]"  # type: ignore
        return Text(text, style="log.path")


def wait_for(condition: Callable[[], bool], description: str, timeout: float = 10.0):
    """Wait for the condition function to return True."""
    start_time = time.monotonic()
    while not condition():
        time.sleep(0.5)
        if time.monotonic() - start_time > timeout:
            raise TimeoutError(f"{description} timed out")


def is_url(path: PathOrStr) -> bool:
    return re.match(r"[a-z0-9]+://.*", str(path)) is not None


def get_progress_bar() -> Progress:
    from cached_path import get_download_progress

    return get_download_progress()


class ResultThread(threading.Thread):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self._result = None

    def run(self):
        self._result = self.fn(*self.args, **self.kwargs)


def rank0_resource_path(device, folder=None, fname=None, local_cache=None,
                        progress=None, cache_dir=None, sleep_time=2):
    """
    Call `resource_path` with the given args for the rank0 process, other ranks will
    wait until rank0 is done downloading.

    Should be called on all ranks, rank0 will get a path, other ranks will get None

    This can be useful if just using `barrier` might timeout because the file being
    downloaded is huge
    """
    if get_global_rank() == 0:
        thread = ResultThread(
            fn=resource_path, fname=fname, folder=folder, local_cache=local_cache, progress=progress,
            cache_dir=cache_dir, quiet=True)
        thread.start()
    else:
        thread = None
    while synchronize_flag(thread.is_alive() if thread else True, device):
        time.sleep(sleep_time)
    if thread:
        return thread._result
    else:
        return None


def resource_path(
    folder: PathOrStr, fname: str=None, local_cache: Optional[PathOrStr] = None,
    progress: Optional[Progress] = None, cache_dir=None, quiet=False
) -> Path:
    if fname is None:
        fname = basename(folder)
        folder = dirname(folder)
    if local_cache is not None and (local_path := Path(local_cache) / fname).is_file():
        log.info(f"Found local cache of {fname} at {local_path}")
        return local_path
    else:
        from cached_path import cached_path

        return cached_path(f"{str(folder).rstrip('/')}/{fname}", progress=progress,
                           cache_dir=cache_dir, quiet=quiet)


def get_default_thread_count() -> int:
    """
    Get the default maximum number of threads allowed.
    """
    env_val = os.environ.get(OLMO_NUM_THREADS_ENV_VAR)
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            raise OLMoEnvironmentError(
                f"Invalid value for {OLMO_NUM_THREADS_ENV_VAR} environment variable ('{env_val}')"
            )
    else:
        return min(16, (os.cpu_count() or 1) + 4)


def split_dict_of_list(batch, split_size):
    out = None
    for key, val in batch.items():
        parts = split_list(val, split_size)
        if out is None:
            out = [{key: part} for part in parts]
        else:
            assert len(out) == len(parts)
            for out_dict, part in zip(out, parts):
                out_dict[key] = part
    return out


def split_list(lst, split_size):
    assert len(lst) % split_size == 0
    n = len(lst) // split_size
    return [lst[i*split_size:(i+1)*split_size] for i in range(n)]


def get_all_keys(dicts):
    keys = set(dicts[0])
    for x in dicts[1:]:
        keys.update(x)
    return keys

def flatten_list(lst):
    return [x for xs in lst for x in xs]


def transpose_dict_of_lists(data: Dict[Any, List]) -> List[Dict]:
    n = len(next(iter(data.values())))
    return [{k: v[i] for k, v in data.items()} for i in range(n)]


def log_metrics_to_console(prefix: str, metrics: Dict[str, float]):
    # FiXME repeats code from trainer
    def format_value(value: float) -> str:
        if isinstance(value, str):
            return value
        if value < 0.0001:
            return str(value)  # scientific notation
        elif value > 1000:
            return f"{int(value):,d}"
        elif value > 100:
            return f"{value:.2f}"
        elif value > 10:
            return f"{value:.3f}"
        elif value > 1:
            return f"{value:.4f}"
        else:
            return f"{value:.5f}"

    logging.info(
        f"{prefix}\n"
        + "\n".join(
            [
                f"    {name}={format_value(value)}"
                for name, value in metrics.items()
                if not name.startswith("optim/")  # there's too many optimizer metrics
            ]
        )
    )


def get_absolute_coordinates(point, w, h) -> list:
    """Convert normalized coordinates to absolute pixel coordinates."""
    if np.max(point) > 100:
        raise ValueError("Point coordinates should be in the range [0, 100]")
    point = np.array(point, dtype=np.float32)
    point /= 100.0
    point = point * np.array([w, h])
    return point.tolist()


def setup_gcp_credentials():
    if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ and "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        credentials = "gcp_credentials.json"
        if get_local_rank() == 0:
            log.info("Writing GCP credentials to credentials.json")
            with open(credentials, "w") as f:
                f.write(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = abspath(credentials)
        barrier()


def setup_s3_credentials():
    if "AWS_CREDENTIALS" in os.environ and "AWS_SHARED_CREDENTIALS_FILE" not in os.environ:
        credentials = "aws_credentials"
        if get_local_rank() == 0:
            log.info("Writing AWS credentials to %s", credentials)
            with open(credentials, "w") as f:
                f.write(os.environ["AWS_CREDENTIALS"])
        os.environ["AWS_SHARED_CREDENTIALS_FILE"] = abspath(credentials)
        barrier()


def _format(val):
    if val is None:
        return ""
    if isinstance(val, int):
        return str(val)
    elif isinstance(val, str):
        return val
    else:
        return f"{val:0.2f}"


def list_of_dict_to_string(table: List[Dict[str, Union[str, int, float]]], filler="", rows=None) -> str:
    keys = dict()
    for row in table:
        keys.update(row)
    if rows is not None:
        keys = [k for k in rows if k in keys] + [k for k in keys if k not in rows]
    raw_table = [list(keys)]
    raw_table += [[_format(row.get(key, filler)) for key in keys] for row in table]
    return table_string(raw_table)


def table_string(table: List[List[str]]) -> str:
    """Table as listoflists to evenly spaces string"""
    # print while padding each column to the max column length
    if len(table) == 0:
        return ""
    col_lens = [0] * len(table[0])
    for row in table:
        for i, cell in enumerate(row):
            col_lens[i] = max(len(cell), col_lens[i])

    formats = ["{0:<%d}" % x for x in col_lens]
    out = []
    for row in table:
        out.append(" ".join(formats[i].format(row[i]) for i in range(len(row))))
    return "\n".join(out)


def generate_uuid() -> str:
    """
    Generate a unique ID.
    """
    return str(uuid.uuid4())


def resolve_hf_checkpoint_ref(path: PathOrStr) -> str:
    return str(path)


def looks_like_hf_model_id(path: PathOrStr) -> bool:
    """Return true for Hugging Face model ids such as ``allenai/MolmoAct2``."""
    checkpoint = resolve_hf_checkpoint_ref(str(path).strip())
    if not checkpoint or "://" in checkpoint:
        return False
    if checkpoint.startswith(("/", "./", "../", "~")):
        return False
    parts = checkpoint.split("/")
    return len(parts) == 2 and all(parts)


def is_hf_checkpoint_ref(path: PathOrStr) -> bool:
    checkpoint = resolve_hf_checkpoint_ref(path)
    local_path = Path(checkpoint).expanduser()
    if local_path.exists():
        if (local_path / "model.safetensors.index.json").is_file():
            return True
        if (local_path / "pytorch_model.safetensors.index.json").is_file():
            return True
        return any(local_path.glob("*.safetensors"))
    return looks_like_hf_model_id(checkpoint)


def _checkpoint_archive_cache_dir(archive: PathOrStr) -> Path:
    cache_root = os.environ.get("MOLMOACT2_CHECKPOINT_CACHE")
    if not cache_root:
        cache_root = join(os.environ.get("HF_HOME", str(Path.home() / ".cache")), "molmoact2", "checkpoints")
    archive_id = compute_hash(str(archive))[:16]
    return Path(cache_root).expanduser() / archive_id


def _safe_extract_tar(archive_path: Path, extract_dir: Path) -> None:
    extract_root = extract_dir.resolve()
    with tarfile.open(archive_path) as tar:
        for member in tar.getmembers():
            member_path = (extract_root / member.name).resolve()
            if not str(member_path).startswith(str(extract_root)):
                raise OLMoCliError(f"Refusing to extract unsafe checkpoint archive member: {member.name}")
        tar.extractall(extract_root)


def _find_extracted_checkpoint_dir(extract_dir: Path) -> Path:
    candidates = [extract_dir]
    candidates.extend(path for path in extract_dir.iterdir() if path.is_dir())
    for candidate in candidates:
        if (candidate / "model.pt").is_file() or (candidate / "config.yaml").is_file() or (candidate / "model.yaml").is_file():
            return candidate
    return extract_dir


def _resolve_checkpoint_archive(checkpoint: PathOrStr, quiet: bool = False) -> str:
    checkpoint_str = str(checkpoint)
    if not checkpoint_str.endswith((".tar", ".tar.gz", ".tgz")):
        return checkpoint_str

    extract_dir = _checkpoint_archive_cache_dir(checkpoint_str)
    marker = extract_dir / ".complete"
    lock_path = extract_dir.with_suffix(".lock")
    extract_dir.parent.mkdir(parents=True, exist_ok=True)

    import fcntl

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if not marker.is_file():
            extract_dir.mkdir(parents=True, exist_ok=True)
            for child in extract_dir.iterdir():
                if child == marker:
                    continue
                if child.is_dir():
                    import shutil

                    shutil.rmtree(child)
                else:
                    child.unlink()
            if not quiet:
                log.info("Downloading checkpoint archive %s", checkpoint_str)
            archive_path = resource_path(checkpoint_str, quiet=quiet)
            if not quiet:
                log.info("Extracting checkpoint archive %s to %s", archive_path, extract_dir)
            _safe_extract_tar(Path(archive_path), extract_dir)
            marker.write_text(str(archive_path))
        fcntl.flock(lock_file, fcntl.LOCK_UN)

    resolved = _find_extracted_checkpoint_dir(extract_dir)
    if not quiet:
        log.info("Resolved checkpoint archive %s to %s", checkpoint_str, resolved)
    return str(resolved)


def select_checkpoint(checkpoint, prefer_unsharded=False, quiet=False):
    """
    returns the latest is checkpoint directory in `checkpoint`, returns `checkpoint`
    if it is already a checkpoint dir
    """
    checkpoint = _resolve_checkpoint_archive(checkpoint, quiet=quiet)

    if is_hf_checkpoint_ref(checkpoint):
        checkpoint = resolve_hf_checkpoint_ref(checkpoint)
        if not quiet:
            log.info("Using Hugging Face checkpoint %s", checkpoint)
        return checkpoint

    if file_exists(join(checkpoint, "model.pt")) or not dir_is_empty(join(checkpoint, "model_and_optim")):
        return checkpoint

    # This might be a model save directory, check for checkpoints based on the filename
    candidates = []
    for file in list_directory(checkpoint, include_files=False):
        match = re.match(".*/step([0-9]+)(-unsharded)?.*", file)
        if match:
            sharded_val = bool(match.group(2))
            if not prefer_unsharded:
                sharded_val = not sharded_val
            candidates.append((file, int(match.group(1)), sharded_val))
    if len(candidates) == 0:
        raise FileNotFoundError(f"{checkpoint} is not a checkpoint, and does not contain any checkpoints")
    oldest = max(candidates, key=lambda x: x[1:])[0]
    if not quiet:
        log.info(f"Selected {oldest} as oldest checkpoint in {checkpoint}")
    return oldest


def format_timedelta(td: timedelta) -> str:
    def format_value_and_unit(value: int, unit: str) -> str:
        if value == 1:
            return f"{value} {unit}"
        else:
            return f"{value} {unit}s"

    parts = []
    seconds = int(td.total_seconds())
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        parts.append(format_value_and_unit(days, "day"))
    if hours:
        parts.append(format_value_and_unit(hours, "hour"))
    if minutes:
        parts.append(format_value_and_unit(minutes, "minute"))
    if seconds:
        parts.append(format_value_and_unit(seconds, "second"))
    return ", ".join(parts)


def  interpolate_frame_scores(scores: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Interpolate frame scores to a target number of frames using bilinear interpolation.

    Args:
        scores: Original scores array, sampled at a specific rate (e.g., 0.5 fps)
        target_frames: Number of frames to interpolate to

    Returns:
        Interpolated scores array with length equal to target_frames
    """
    # Create source and target indices
    source_indices = np.linspace(0, len(scores) - 1, len(scores))
    target_indices = np.linspace(0, len(scores) - 1, target_frames)

    # Perform linear interpolation
    interpolated_scores = np.interp(target_indices, source_indices, scores)

    return interpolated_scores


def parse_timestamp(time_value: Union[str, float]) -> float:
    if isinstance(time_value, str):
        if ':' in time_value:  # MM:SS.FF format
            try:
                time_obj = datetime.strptime(time_value, "%M:%S.%f")
                return time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1000000
            except ValueError:
                pass
            try:
                time_obj = datetime.strptime(time_value, "%H:%M:%S.%f")
                return time_obj.hour * 60 * 60 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1000000
            except ValueError:
                logging.warning(f"Unable to parse timestamp: {time_value}")
                return time_value  # Return as-is if can't parse
        else:
            return float(time_value)
    else:
        return time_value


def normalize_timestamps_and_points(
    triplets: List[Tuple[float, float, float]],
    video_duration: float,
    video_h: float,
    video_w: float,
    upper_bound: int = 100,
    num_decimals: int = 1,
) -> List[Tuple[float, float, float]]:
    """
    Normalize the timestamps and points in the triplets to a range of [0, 100].
    """
    norm_triplets = []
    for ts, x, y in triplets:
        norm_ts = round(ts / video_duration * upper_bound, num_decimals)
        norm_x = round(x / video_w * upper_bound, num_decimals)
        norm_y = round(y / video_h * upper_bound, num_decimals)
        norm_triplets.append((norm_ts, norm_x, norm_y))
    return norm_triplets


def set_example_style(ex: Dict, style: str) -> Dict:
    if "message_list" in ex:
        return dict(
            ex,
            message_list=[dict(x, style=style) for x in ex["message_list"]]
        )
    else:
        return dict(ex, style=style)
