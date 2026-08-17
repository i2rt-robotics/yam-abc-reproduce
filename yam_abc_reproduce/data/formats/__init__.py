"""Format registry + the convert_episode entry point.

The default format is registered eagerly (lightweight). LeRobot and ABC are
imported lazily so their heavy/optional deps are only required when used.
"""

from __future__ import annotations

from pathlib import Path

from .base import DatasetWriter, EpisodeWriter, FormatReader, get_reader, get_writer, register
from .default_format import DefaultFormat

# Eager registration of the canonical format (reader + writer).
register("default", DefaultFormat)

__all__ = [
    "DatasetWriter",
    "EpisodeWriter",
    "FormatReader",
    "get_reader",
    "get_writer",
    "register",
    "convert_episode",
]


def _ensure_registered(name: str) -> None:
    if name == "lerobot":
        from . import lerobot_format  # noqa: F401  (registers on import)
    elif name == "abc":
        from . import abc_format  # noqa: F401


def convert_episode(src: str | Path, to: str, repo_id: str, out: str | None = None) -> None:
    """Convert a default-format episode (or a directory of them) to ``to`` format."""
    _ensure_registered(to)
    src = Path(src)
    reader = get_reader("default")
    writer = get_writer(to)

    # A directory containing the completeness flag is a single episode; otherwise
    # treat it as a parent directory of episodes.
    from ..schema import WRITE_COMPLETE_FLAG

    if (src / WRITE_COMPLETE_FLAG).exists():
        episodes = [src]
    else:
        episodes = sorted(p for p in src.iterdir() if (p / WRITE_COMPLETE_FLAG).exists())
    if not episodes:
        raise FileNotFoundError(f"no completed episodes under {src}")

    writer.begin(repo_id=repo_id, out=out)
    for ep in episodes:
        meta, buffers = reader.read_episode(ep)
        writer.add_episode(meta, buffers)
    writer.finalize()
