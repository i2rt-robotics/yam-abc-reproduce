"""ABC (DiT) policy server on the YAM-ABC-Reproduce wire protocol.

Needs the `abc-policy` group (uv sync --extra deploy --group abc-policy). Loads
``SimPolicy`` (abc_minimal) from a checkpoint.

YAM-ABC-Reproduce obs -> ABC mapping:
    images[role] (H,W,3 uint8 RGB) -> transposed to (3,H,W) per camera_keys
    state (14,)                    -> state
    prompt                         -> encoded into the CLIP task vector

NOTE on prompt: ABC bakes the instruction into a CLIP task vector at load time
(``SimPolicy.infer`` ignores ``obs["prompt"]``), so --prompt is fixed at launch.
To switch tasks, restart the server -- or re-encode ``self.task_vec`` here.

Response: {"actions": (chunk_length, action_dim) float32}, e.g. (30, 14).

Run (cd third_party/policy/abc so abc_minimal is importable; python from the repo's .venv):
    python <yam_abc_reproduce>/yam_abc_reproduce/deploy/servers/abc_server.py \
        --checkpoint /path/to/ckpt.pt --prompt "put the bottle in the bin" --port 8300
Extra deps: `websockets` + openpi-client (msgpack_numpy) -- the `abc-policy` group has
neither, so add `--extra deploy` or put openpi-client on PYTHONPATH (see docs/deploy.md).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import _wire 
from imageproc import letterbox_224

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("yam_abc_reproduce.deploy.abc")


def _maybe_merge_lora(ckpt_path: str) -> str:
    """A LoRA-trained ABC checkpoint stores base/lora_a/lora_b per wrapped Linear,
    which the plain DiT can't load. If detected, fold the adapters into the base
    weights (W = base + lora_b@lora_a, scale=1.0 as apply_lora uses no alpha),
    preserve norm_stats, write a sibling ``*_merged.pt`` and return its path.
    Non-LoRA checkpoints pass through unchanged."""
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    prefixes = sorted({k[: -len(".base.weight")] for k in sd if k.endswith(".base.weight")})
    if not prefixes:
        return ckpt_path  # already a plain checkpoint

    merged_path = os.path.splitext(ckpt_path)[0] + "_merged.pt"
    if os.path.isfile(merged_path):
        log.info("using existing merged checkpoint %s", merged_path)
        return merged_path

    log.info("LoRA checkpoint detected (%d wrapped layers); merging -> %s",
             len(prefixes), merged_path)
    merged, consumed = {}, set()
    for p in prefixes:
        bw = sd[f"{p}.base.weight"]
        la, lb = sd.get(f"{p}.lora_a.weight"), sd.get(f"{p}.lora_b.weight")
        w = bw.float()
        if la is not None and lb is not None:
            w = w + (lb.float() @ la.float())  # scale = 1.0
        merged[f"{p}.weight"] = w.to(bw.dtype)
        consumed |= {f"{p}.base.weight", f"{p}.lora_a.weight", f"{p}.lora_b.weight"}
        bb = sd.get(f"{p}.base.bias")
        if bb is not None:
            merged[f"{p}.bias"] = bb
            consumed.add(f"{p}.base.bias")
    for k, v in sd.items():
        if k not in consumed:
            merged[k] = v
    out = {"model": merged}
    if isinstance(ckpt, dict):
        for k, v in ckpt.items():  # keep norm_stats etc., drop the huge optimizer
            if k not in ("model", "optimizer"):
                out[k] = v
    torch.save(out, merged_path)
    return merged_path


def _to_chw_uint8(img: np.ndarray) -> np.ndarray:
    """YAM-ABC-Reproduce sends HWC uint8; ABC's preprocess expects CHW uint8."""
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] != 3 and a.shape[2] == 3:
        a = np.transpose(a, (2, 0, 1))
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def build_infer(policy, camera_keys):
    def infer(obs: dict) -> dict:
        images = obs["images"]
        missing = [c for c in camera_keys if c not in images]
        if missing:
            raise ValueError(f"missing camera roles {missing}; got {list(images)}")
        abc_obs = {
            "state": np.asarray(obs["state"], dtype=np.float32),
            # Letterbox to 224 mirroring export_mcap.py's training resize (imageproc).
            "images": {c: _to_chw_uint8(letterbox_224(images[c])) for c in camera_keys},
            "prompt": str(obs.get("prompt", "")),  # ignored by infer; task_vec is fixed
        }
        # RTC: if the client sends an action prefix, condition the chunk on it
        # (sample_actions_rtc) so real-time chunks connect smoothly.
        ap = obs.get("action_prefix")
        if ap is not None:
            actions = policy.infer(
                abc_obs, action_prefix=np.asarray(ap, dtype=np.float32),
                prefix_length=int(obs.get("prefix_length", 4)),
            )
        else:
            actions = policy.infer(abc_obs)  # (chunk_length, action_dim), unnormalized
        actions = np.array(actions, dtype=np.float32) 
        # Clip the per-arm gripper channels to [0,1] (their normalized range), as the
        # reference deployment does — unclipped grippers over-drive the real hardware.
        if actions.ndim == 2 and actions.shape[1] == 14:
            actions[:, [6, 13]] = np.clip(actions[:, [6, 13]], 0.0, 1.0)
        return {"actions": actions}

    return infer


def main() -> None:
    p = argparse.ArgumentParser(description="ABC DiT YAM-ABC-Reproduce websocket server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8300)
    p.add_argument("--checkpoint", required=True, help="checkpoint path or s3:// uri")
    p.add_argument("--prompt", required=True, help="fixed task instruction (baked at load)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from abc_minimal.config import SimEvalConfig
    from abc_minimal.eval_policy import SimPolicy, local_checkpoint

    # Override model config here if trained checkpoint differs from the sim default (DiTConfig:
    # action_dim=14, chunk_length=30, camera_keys=("top","left","right")).
    from abc_minimal.dit import infer_dit_shape

    config = SimEvalConfig(checkpoint=args.checkpoint, prompt=args.prompt)
    ckpt_path = local_checkpoint(config.checkpoint)
    ckpt_path = _maybe_merge_lora(ckpt_path)  # fold LoRA adapters if present
    # Match the model shape to the checkpoint.
    shape = infer_dit_shape(ckpt_path)
    for k, v in shape.items():
        setattr(config.model, k, v)
    if shape:
        log.info("auto-matched DiT shape to checkpoint: %s", shape)
    policy = SimPolicy(ckpt_path, config, device=args.device)

    camera_keys = tuple(config.model.camera_keys)
    metadata = {
        "backend": "abc",
        "state_dim": int(config.model.state_dim),
        "action_dim": int(config.model.action_dim),
        "chunk_length": int(config.model.chunk_length),
        "camera_order": list(camera_keys),
        "prompt": args.prompt,
    }
    _wire.serve(build_infer(policy, camera_keys), args.host, args.port, metadata)


if __name__ == "__main__":
    main()
