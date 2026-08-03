#!/usr/bin/env python3
"""Compute ABC ``norm_stats.json`` from an exported training cache.

ABC normalizes state/action with ``(x - mean) / (std + 1e-6)`` (see
``abc_minimal/preprocess.py``). The upstream ships a pre-baked ``norm_stats.json``
tied to its own data; when you train/fine-tune on a *different* dataset the stats
must be recomputed from that data, or dimensions the reference held constant
(std -> 0) blow the normalized values (and the loss) up to millions.

This reads every ``states_actions.bin`` under ``<cache>/<train_dir>`` (columns are
``[state(state_dim) | action(action_dim)]`` per row, written by
``third_party/policy/abc/export_mcap.py``), computes per-dimension mean/std, and
floors near-constant dimensions to std=1.0 so they normalize to ~0 instead of
exploding. Writes ``<cache>/norm_stats.json`` in the schema the trainer loads.
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

STATE_DIM = 14  # [left arm6, left ee1, right arm6, right ee1]
STD_FLOOR = 1e-2  # dims with std below this are treated as constant -> std=1.0


def load_rows(cache: Path, train_dir: str) -> np.ndarray:
    rows = []
    for d in sorted(glob.glob(str(cache / train_dir / "*" / ""))):
        meta = json.loads((Path(d) / "episode_metadata.json").read_text())
        n = meta["num_steps"]
        rows.append(np.fromfile(Path(d) / "states_actions.bin").reshape(n, -1))
    if not rows:
        raise SystemExit(f"no episodes under {cache/train_dir}")
    return np.concatenate(rows, axis=0)


def stats(x: np.ndarray) -> dict:
    mean = x.mean(0)
    std = x.std(0)
    floored = std < STD_FLOOR
    std = np.where(floored, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist(), "_floored_dims": np.where(floored)[0].tolist()}


@dataclass
class ComputeAbcNormStatsArgs:
    cache: str
    """ABC cache root (contains train_real/)"""
    train_dir: str = "train_real"
    """split to compute stats over"""
    state_dim: int = STATE_DIM
    out: str | None = None
    """output path; defaults to <cache>/norm_stats.json"""


def main() -> None:
    args = tyro.cli(
        ComputeAbcNormStatsArgs,
        description="Compute ABC norm_stats.json from an export cache",
    )

    cache = Path(args.cache)
    rows = load_rows(cache, args.train_dir)
    state = stats(rows[:, : args.state_dim])
    actions = stats(rows[:, args.state_dim :])
    for name, s in (("state", state), ("actions", actions)):
        fl = s.pop("_floored_dims")
        print(f"{name}: dims={len(s['mean'])}  std=[{min(s['std']):.4g}..{max(s['std']):.4g}]"
              + (f"  floored dims (constant)->1.0: {fl}" if fl else "  (no floored dims)"))

    out = Path(args.out) if args.out else cache / "norm_stats.json"
    out.write_text(json.dumps({"state": state, "actions": actions}, indent=2))
    print(f"\nrows={len(rows)}  wrote {out}")


if __name__ == "__main__":
    main()
