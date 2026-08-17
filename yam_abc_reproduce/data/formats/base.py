"""Format protocols + a tiny name->class registry.

Two writer roles exist:

* **EpisodeWriter** (``write_episode``) — used by the recorder to write ONE
  episode into a folder. Implemented by the canonical ``default`` format.
* **DatasetWriter** (``begin`` / ``add_episode`` / ``finalize``) — used by the
  converter to accumulate many episodes into a target dataset (e.g. LeRobot).

A **FormatReader** (``read_episode``) turns a stored episode back into
``(EpisodeMeta, buffers)``. ``buffers`` is a ``dict[str, list]`` keyed by schema
keys: state/action keys map to lists of 1-D arrays, camera image keys map to
lists of HxWx3 frames, and ``<role>-timestamp`` maps to a list of floats.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..schema import EpisodeMeta


@runtime_checkable
class EpisodeWriter(Protocol):
    def write_episode(self, episode_dir, meta: EpisodeMeta, buffers: dict) -> None: ...


@runtime_checkable
class FormatReader(Protocol):
    def read_episode(self, episode_dir) -> tuple[EpisodeMeta, dict]: ...


@runtime_checkable
class DatasetWriter(Protocol):
    def begin(self, repo_id: str, out: str | None = None) -> None: ...
    def add_episode(self, meta: EpisodeMeta, buffers: dict) -> None: ...
    def finalize(self) -> None: ...


_REGISTRY: dict[str, type] = {}


def register(name: str, cls: type) -> None:
    _REGISTRY[name] = cls


def _get(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"unknown format {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def get_writer(name: str):
    """Return a writer instance (EpisodeWriter for default, DatasetWriter otherwise)."""
    return _get(name)


def get_reader(name: str) -> FormatReader:
    return _get(name)
