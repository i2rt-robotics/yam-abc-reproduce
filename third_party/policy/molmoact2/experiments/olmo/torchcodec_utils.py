from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional


def _module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception:
        return "unavailable"
    return str(getattr(module, "__version__", "unknown"))


def _build_ffmpeg_subprocess_env() -> dict[str, str]:
    allowed_keys = {
        "HOME",
        "TMPDIR",
        "LD_LIBRARY_PATH",
        "PATH",
        "LANG",
        "LC_ALL",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
    }
    return {
        key: value
        for key, value in os.environ.items()
        if key in allowed_keys and value
    }


def _create_smoke_video(video_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg binary was not found on PATH.")

    try:
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=32x32:rate=4",
                "-t",
                "1",
                "-pix_fmt",
                "yuv420p",
                str(video_path),
            ],
            env=_build_ffmpeg_subprocess_env(),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed while creating TorchCodec smoke video: {exc.stderr.strip()}"
        ) from exc


def collect_torchcodec_runtime_info() -> dict[str, Any]:
    import torch

    return {
        "torch": torch.__version__,
        "torchvision": _module_version("torchvision"),
        "torchaudio": _module_version("torchaudio"),
        "torchcodec": _module_version("torchcodec"),
        "lerobot": _module_version("lerobot"),
        "cuda": str(getattr(torch.version, "cuda", None)),
        "ffmpeg": shutil.which("ffmpeg") or "unavailable",
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
    }


def verify_torchcodec_runtime(video_path: Optional[str] = None) -> dict[str, Any]:
    info = collect_torchcodec_runtime_info()
    tmpdir: Optional[tempfile.TemporaryDirectory[str]] = None
    path: Path
    if video_path is None:
        tmpdir = tempfile.TemporaryDirectory(prefix="torchcodec_smoke_")
        path = Path(tmpdir.name) / "smoke.mp4"
        _create_smoke_video(path)
        info["video_source"] = "synthetic"
    else:
        path = Path(video_path).expanduser()
        if not path.exists():
            raise RuntimeError(f"TorchCodec preflight video does not exist: {path}")
        info["video_source"] = "user"
    info["video_path"] = str(path)

    try:
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(str(path), seek_mode="exact", num_ffmpeg_threads=1)
        metadata = decoder.metadata
        frames = decoder.get_frames_at(indices=[0])
        decoded_frames = int(frames.data.shape[0])
        if decoded_frames != 1:
            raise RuntimeError(f"Expected one decoded frame, got {decoded_frames}.")
        info["num_frames"] = None if metadata.num_frames is None else int(metadata.num_frames)
        info["average_fps"] = None if metadata.average_fps is None else float(metadata.average_fps)
        info["decoded_frames"] = decoded_frames
        return info
    except Exception as exc:
        raise RuntimeError(
            "TorchCodec preflight failed. "
            f"runtime={json.dumps(info, sort_keys=True)}"
        ) from exc
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()
