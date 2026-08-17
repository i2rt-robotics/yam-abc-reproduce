"""Live progress for a running ``yam-abc-convert`` job.

The converter has no progress protocol: it emits no ``@metric`` lines, and its tqdm bars do
not survive the pipe. So progress is read from what it writes under ``data/`` instead, which
also means it works for a conversion the GUI did not start.

What counts as "done" differs per format, and only one of the two is obvious:

- **abc** writes one ``episode_NNNNNN.mcap`` per finished episode, so the file count is the
  progress.
- **lerobot** (v3) does *not* write per-episode files. ``data/chunk-000/file-000.parquet`` and
  the per-camera ``videos/.../file-000.mp4`` are aggregates that accumulate every episode, so
  counting them tells you nothing. ``meta/info.json`` holds the real counters, and it is only
  rewritten when an episode commits -- progress therefore advances in steps, minutes apart,
  while raw frames are extracted and encoded in between.

Episode counts make a poor progress bar here: takes in one corpus ranged from 88 to 3981
frames, so "4/33 episodes" is only loosely tied to the work done. Frames are linear in it, and
the per-episode frame count can be read from the source timestamps without decoding anything.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..data.schema import WRITE_COMPLETE_FLAG

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LEROBOT_HOME = _REPO_ROOT / "data" / "lerobot"
_ABC_OUT = _REPO_ROOT / "data" / "abc"
_EPISODES = _REPO_ROOT / "data" / "episodes"

# A float64 .npy is a 128-byte header then 8 bytes per sample, so a timestamp file's size
# gives its frame count without reading it. Cross-checked against LeRobot's own
# meta/info.json total_frames on a 33-episode corpus: exact.
_NPY_HEADER, _F64 = 128, 8

_io_cache: dict[int, tuple[float, int]] = {}     # pid -> (when, write_bytes) for the rate


def _episode_frames(d: Path) -> int:
    """Frame count for one recorded episode.

    metadata.json's num_frames is the recorder's own count and is what the converter uses as
    its reference timeline. Falling back to *any* timestamp file rather than a hardcoded
    ``top-`` one matters: the camera roster is configurable (a station can run left+wrist with
    no ``top`` at all), and a missing denominator would pin the bar at 0% for the whole run.
    """
    try:
        n = json.loads((d / "metadata.json").read_text()).get("num_frames")
        if isinstance(n, int) and n > 0:
            return n
    except (OSError, ValueError, AttributeError):
        pass
    for ts in sorted(d.glob("*-timestamp.npy")):
        return max(0, (ts.stat().st_size - _NPY_HEADER) // _F64)
    return 0


def _source_totals(src: Path) -> tuple[int, int]:
    """(episode count, frame count) for a source dir.

    Only dirs carrying WRITE_COMPLETE_FLAG count, mirroring the converter's own predicate
    (data/formats/__init__.py) -- a half-written take left by a hard kill would otherwise
    inflate the denominator so the bar could never reach 100%.

    Deliberately not cached. Caching by path went stale the moment an episode was recorded or
    pruned between two conversions in one GUI session, which showed up as "4/3 episodes" and a
    bar stuck at 100%. It is a listdir plus one read per episode, which is nothing at 1 Hz.
    """
    if not src.exists():
        return 0, 0
    eps = [d for d in src.glob("*") if (d / WRITE_COMPLETE_FLAG).exists()]
    return len(eps), sum(_episode_frames(d) for d in eps)


def _proc_tree(pid: int) -> list[int]:
    """The pid and every descendant. LeRobot forks encoder workers that do most of the IO."""
    seen, stack = [], [pid]
    while stack:
        p = stack.pop()
        seen.append(p)
        try:
            for task in Path(f"/proc/{p}/task").iterdir():
                stack += [int(c) for c in (task / "children").read_text().split()]
        except (OSError, ValueError):
            pass
    return seen


def _write_rate(pid: int | None) -> float | None:
    """MiB/s written by the job tree since the last call, from the kernel's own counters.

    /proc/<pid>/io is free to read, unlike du over a staging tree of tens of thousands of
    PNGs. None until there are two samples to difference.
    """
    if not pid:
        return None
    total = 0
    for p in _proc_tree(pid):
        try:
            for line in Path(f"/proc/{p}/io").read_text().splitlines():
                if line.startswith("write_bytes:"):
                    total += int(line.split()[1])
                    break
        except (OSError, ValueError, IndexError):
            pass  # a worker that exited between the walk and the read
    now = time.time()
    prev = _io_cache.get(pid)
    _io_cache[pid] = (now, total)
    if prev is None or now <= prev[0] or total < prev[1]:
        return None  # first sample, or the tree turned over and the counter went backwards
    return (total - prev[1]) / (now - prev[0]) / (1024 * 1024)


def _read_counter(info: Path, key: str) -> int:
    """One integer out of LeRobot's meta/info.json without importing json for a partial write."""
    try:
        text = info.read_text()
    except OSError:
        return 0
    marker = f'"{key}"'
    i = text.find(marker)
    if i < 0:
        return 0
    digits = ""
    for ch in text[i + len(marker):]:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    return int(digits) if digits else 0


def snapshot(params: dict, pid: int | None, started: float) -> dict:
    """Progress for one convert job: episodes, frames, write rate, ETA.

    ``started`` is the job's launch time; rate is cumulative work over total elapsed, which
    averages across whole episodes and so rides out the extract/encode sawtooth. A short
    window would read zero for minutes at a time, since progress only moves at commits.
    """
    task = (params.get("task") or params.get("dataset") or "").strip()
    src_arg = params.get("src") or (f"data/episodes/{task}" if task else "")
    src = Path(src_arg) if Path(src_arg).is_absolute() else _REPO_ROOT / src_arg
    if not src_arg:
        src = _EPISODES
    fmt = params.get("to", "lerobot")
    repo_id = (params.get("repo_id") or task or "").strip()

    ep_total, frame_total = _source_totals(src)
    if fmt == "abc":
        # Two ways a plain glob overcounts. A killed run leaves its .mcap files behind
        # (ABCFormat.begin only mkdirs and resets its index), which would read as instant
        # progress -- so only count what THIS job wrote. And the newest file is the one being
        # written right now, minutes of encoding on a real episode, so it is not done yet.
        mcaps = [p for p in (_ABC_OUT / repo_id).glob("episode_*.mcap")
                 if p.stat().st_mtime >= started] if repo_id else []
        ep_done = len(mcaps)
        if ep_done and pid and Path(f"/proc/{pid}").exists():
            ep_done -= 1        # the in-flight one; app.js snaps the bar to 100% on exit
        frame_done = frame_total = 0
        done, total = ep_done, ep_total
    else:
        info = _LEROBOT_HOME / repo_id / "meta" / "info.json"
        ep_done = _read_counter(info, "total_episodes")
        frame_done = _read_counter(info, "total_frames")
        done, total = frame_done, frame_total

    frac = min(1.0, done / total) if total else 0.0
    elapsed = max(1e-6, time.time() - started)
    eta = (total - done) * elapsed / done if done and total > done else None

    return {
        "fmt": fmt,
        "episodes_done": ep_done,
        "episodes_total": ep_total,
        "frames_done": frame_done,
        "frames_total": frame_total,
        "write_mib_s": _write_rate(pid),
        "frac": frac,
        "eta_s": eta,
    }
