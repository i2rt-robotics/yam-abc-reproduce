"""Post-collection sanity check: reuse LeRobot's Rerun-based dataset viewer.

We deliberately do NOT build our own viewer. ``lerobot.scripts.visualize_dataset``
already shows synchronized video + action/state curves per episode.
"""

from __future__ import annotations

import subprocess
import sys


def visualize_lerobot(repo_id: str, root: str | None = None, episode_index: int = 0) -> int:
    """Shell out to LeRobot's visualizer for one episode. Returns the exit code."""
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.visualize_dataset",
        "--repo-id",
        repo_id,
        "--episode-index",
        str(episode_index),
    ]
    if root:
        cmd += ["--root", root]
    print("running:", " ".join(cmd))
    return subprocess.call(cmd)
