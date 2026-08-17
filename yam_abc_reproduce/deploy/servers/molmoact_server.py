"""MolmoAct2-BimanualYAM policy server on the YAM-ABC-Reproduce wire protocol.

Needs a molmoact2 group: `molmoact2` mirrors the upstream server env (torch 2.5.1+cu121 /
transformers 4.57); the GUI uses `molmoact2-train`, whose torch carries the sm_120 kernels
cu121 lacks. REUSES the release server's ``Policy`` from ``examples/yam/host_server_yam.py``
-- so its bf16 / snapshot-dir / tokenizer / device-cast workarounds apply verbatim -- and
only swaps the FastAPI+json_numpy shell for the websocket one.

YAM-ABC-Reproduce obs -> MolmoAct mapping:
    images["top"|"left"|"right"] -> ordered [top, left, right] PIL list
    state (14,)                  -> state
    prompt                       -> task/instruction
Response: {"actions": (num_steps, 14) float32}

Run (cd third_party/policy/molmoact2 for its sources; python from the repo's .venv):
    python <yam_abc_reproduce>/yam_abc_reproduce/deploy/servers/molmoact_server.py --port 8202
Extra deps: `websockets` and the openpi-client package (for msgpack_numpy) -- neither
molmoact2 group carries them, so add `--extra deploy` or use PYTHONPATH; see _wire.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import _wire  # noqa: E402  (sibling module, no yam_abc_reproduce dependency)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("yam_abc_reproduce.deploy.molmoact")

# Camera order MUST match training; molmoact YAM expects [top, left, right].
CAMERA_ORDER = ("top", "left", "right")


def _load_host_server_module(molmoact_repo: Path):
    """Import the release server module to reuse its ``Policy`` (with all the
    bf16/snapshot workarounds) without copying any of that logic here."""
    path = molmoact_repo / "examples" / "yam" / "host_server_yam.py"
    if not path.is_file():
        raise SystemExit(
            f"could not find host_server_yam.py at {path}\n"
            "pass --molmoact-repo pointing at the molmoact2 checkout"
        )
    spec = importlib.util.spec_from_file_location("host_server_yam", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def build_infer(policy, num_steps: int):
    def infer(obs: dict) -> dict:
        images = obs["images"]
        missing = [c for c in CAMERA_ORDER if c not in images]
        if missing:
            raise ValueError(f"missing camera roles {missing}; got {list(images)}")
        actions = policy.predict(
            top_cam=images["top"],
            left_cam=images["left"],
            right_cam=images["right"],
            instruction=str(obs.get("prompt", "")),
            state=np.asarray(obs["state"], dtype=np.float32),
            num_steps=num_steps,
        )
        return {"actions": np.asarray(actions, dtype=np.float32)}

    return infer


def main() -> None:
    p = argparse.ArgumentParser(description="MolmoAct2-YAM YAM-ABC-Reproduce websocket server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8202)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument(
        "--checkpoint",
        default=None,
        help="local HF-format checkpoint dir (e.g. a YAM-ABC-Reproduce-finetuned model "
        "converted via convert_molmoact2_to_hf.py). Defaults to the release repo id.",
    )
    p.add_argument(
        "--norm-tag",
        default=None,
        help="normalization tag in the checkpoint's norm_stats.json. Auto-detected "
        "from a local checkpoint (its single metadata_by_tag key) when omitted; "
        "a YAM-ABC-Reproduce-finetuned model uses 'yam_abc_reproduce_yam', the release model "
        "'yam_dual_molmoact2'.",
    )
    p.add_argument(
        "--molmoact-repo",
        default=str(Path(__file__).resolve().parents[3] / "third_party" / "policy" / "molmoact2"),
        help="path to the molmoact2 checkout (default: sibling submodule)",
    )
    args = p.parse_args()

    host_mod = _load_host_server_module(Path(args.molmoact_repo))
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        args.dtype
    ]
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    checkpoint = args.checkpoint or host_mod.REPO_ID

    # The release server hardcodes NORM_TAG="yam_dual_molmoact2", but a YAM-ABC-Reproduce
    # finetune's norm_stats.json carries a different tag (e.g. "yam_abc_reproduce_yam").
    # predict_action reads the module global at call time, so overriding it here
    # is enough. Auto-detect from a local checkpoint when --norm-tag is omitted.
    norm_tag = args.norm_tag
    if norm_tag is None and os.path.isdir(checkpoint):
        try:
            import json
            with open(os.path.join(checkpoint, "norm_stats.json")) as f:
                tags = list(json.load(f).get("metadata_by_tag", {}))
            if len(tags) == 1:
                norm_tag = tags[0]
        except (OSError, ValueError):
            pass
    if norm_tag:
        host_mod.NORM_TAG = norm_tag
        log.info("using norm_tag=%s", norm_tag)
    policy = host_mod.Policy(repo_id=checkpoint, device=args.device, dtype=dtype)
    host_mod.warmup(policy)

    metadata = {
        "backend": "molmoact2",
        "repo_id": checkpoint,
        "state_dim": host_mod.STATE_DIM,
        "camera_order": list(CAMERA_ORDER),
    }
    _wire.serve(build_infer(policy, args.num_steps), args.host, args.port, metadata)


if __name__ == "__main__":
    main()
