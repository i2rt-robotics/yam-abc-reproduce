"""MolmoAct2-BimanualYAM inference server.

Mirrors `host_server.py` but for the bimanual YAM checkpoint:

  * 3 cameras in fixed order [top, left, right]
  * raw robot state is shape (14,)  (per-arm 7-D, two arms)
  * norm_tag = "yam_dual_molmoact2"

Wire protocol:

    GET  /act        -> health check, returns {"status": "ok", ...}
    POST /act        -> action inference
        request body  (json_numpy):
            {
              "top_cam":     ndarray(H, W, 3) uint8 RGB,
              "left_cam":    ndarray(H, W, 3) uint8 RGB,
              "right_cam":   ndarray(H, W, 3) uint8 RGB,
              "instruction": str,
              "state":       ndarray(14,) float32,
              "timestamp":   float (optional),
              "num_steps":   int   (optional, default 10),
              "enable_cuda_graph": bool (optional),
            }
        response body (json_numpy):
            {"actions": ndarray(N, D) float32, "dt_ms": float}

Run:

    uv run python host_server_yam.py --host 0.0.0.0 --port 8202
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from typing import Any

import json_numpy
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Patches the stdlib `json` module so np.ndarray round-trips through JSON.
# Must be called before any json.dumps/loads we rely on.
json_numpy.patch()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("molmoact2.yam.server")


REPO_ID = "allenai/MolmoAct2-BimanualYAM"
NORM_TAG = "yam_dual_molmoact2"
STATE_DIM = 14
NUM_CAMERAS = 3
DEFAULT_NUM_STEPS = 10


def _patch_modeling_for_bf16(local_dir: str) -> None:
    """Same idempotent patches as the DROID server. The dtype needle may no
    longer match newer revisions of `modeling_molmoact2.py` (and will warn
    rather than fail); `_to_array` is still required for bf16.
    """
    patches = [
        (
            "device=device,\n            dtype=torch.float32,\n            generator=generator,",
            "device=device,\n"
            "            dtype=source_tensor.dtype,  # patched_bf16_dtype\n"
            "            generator=generator,",
            "patched_bf16_dtype",
        ),
        (
            "return value.detach().cpu().numpy().astype(np.float32, copy=False)",
            "return value.detach().cpu().float().numpy().astype(np.float32, copy=False)  # patched_bf16_to_array",
            "patched_bf16_to_array",
        ),
    ]
    candidates = [os.path.join(local_dir, "modeling_molmoact2.py")]
    modules_root = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules"
    )
    if os.path.isdir(modules_root):
        for sub in os.listdir(modules_root):
            p = os.path.join(modules_root, sub, "modeling_molmoact2.py")
            if os.path.isfile(p):
                candidates.append(p)
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
        except OSError:
            continue
        new_src = src
        applied: list[str] = []
        for needle, replacement, marker in patches:
            if marker in new_src:
                continue
            if needle not in new_src:
                log.warning("patch %s: needle not found in %s", marker, path)
                continue
            new_src = new_src.replace(needle, replacement, 1)
            applied.append(marker)
        if new_src != src:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_src)
            log.info("Applied patches %s in %s", applied, path)


class Policy:
    """Holds the loaded model + processor and serializes inference calls."""

    def __init__(
        self,
        repo_id: str,
        device: str,
        dtype: torch.dtype,
        enable_cuda_graph: bool = False,
    ) -> None:
        self.default_cuda_graph = enable_cuda_graph
        # `predict_action` reads `norm_stats.json` from `config._name_or_path`.
        # Always resolve to the local snapshot dir so that lookup works. Accept a
        # local directory directly (e.g. a FreeMani-finetuned checkpoint converted
        # to HF format) instead of only HF repo ids.
        local_dir = repo_id if os.path.isdir(repo_id) else snapshot_download(repo_id=repo_id)
        log.info("Resolved snapshot dir: %s", local_dir)

        _patch_modeling_for_bf16(local_dir)

        log.info("Loading processor")
        # `tokenizer_config.json` ships `extra_special_tokens` as a list, which
        # transformers >=4.46 rejects. The model code only uses these via
        # `convert_tokens_to_ids`, so an empty dict is safe.
        self.processor = AutoProcessor.from_pretrained(
            local_dir, trust_remote_code=True, extra_special_tokens={}
        )

        log.info("Loading model (dtype=%s, device=%s)", dtype, device)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                local_dir,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            .to(device)
            .eval()
        )
        self.device = device

        # Upstream `_move_inputs_to_device` only moves tensors; it does not
        # cast floats to the model dtype. With bf16 weights the processor's
        # fp32 `pixel_values` then trips `mat1 and mat2 must have the same
        # dtype`. Replace the bound method per-instance.
        target_dtype = next(self.model.parameters()).dtype

        def _move_and_cast(
            inputs: Any, dev: Any, _target: torch.dtype = target_dtype
        ) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    value = value.to(dev)
                    if value.is_floating_point() and value.dtype != _target:
                        value = value.to(_target)
                out[key] = value
            return out

        self.model._move_inputs_to_device = _move_and_cast
        # CUDA-graph capture in the action expert is not safe under concurrent
        # calls; coarse-grained serialization is fine at ~5 Hz robot poll.
        self._lock = threading.Lock()

    @torch.inference_mode()
    def predict(
        self,
        top_cam: np.ndarray,
        left_cam: np.ndarray,
        right_cam: np.ndarray,
        instruction: str,
        state: np.ndarray,
        num_steps: int = DEFAULT_NUM_STEPS,
        enable_cuda_graph: bool = False,
    ) -> np.ndarray:
        # Camera order must match training: [top, left, right].
        images = [_to_pil(top_cam), _to_pil(left_cam), _to_pil(right_cam)]
        state_f32 = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_f32.shape != (STATE_DIM,):
            raise ValueError(
                f"state must be shape ({STATE_DIM},), got {state_f32.shape}"
            )

        with self._lock:
            out = self.model.predict_action(
                processor=self.processor,
                images=images,
                task=instruction,
                state=state_f32,
                norm_tag=NORM_TAG,
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=num_steps,
                normalize_language=True,
                enable_cuda_graph=enable_cuda_graph,
            )
        raw = out.actions
        if torch.is_tensor(raw):
            raw = raw.detach().to(dtype=torch.float32, device="cpu").numpy()
        actions = np.asarray(raw, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return actions


def _to_pil(arr: Any) -> Image.Image:
    if isinstance(arr, Image.Image):
        return arr.convert("RGB")
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got shape {a.shape}")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(a, mode="RGB")


def build_app(policy: Policy) -> FastAPI:
    app = FastAPI(title="MolmoAct2-BimanualYAM server", version="0.1.0")

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "repo_id": REPO_ID,
                "norm_tag": NORM_TAG,
                "device": policy.device,
                "dtype": str(policy.model.dtype),
                "num_cameras": NUM_CAMERAS,
                "state_dim": STATE_DIM,
            }
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/act")
    async def act(request: Request) -> Response:
        raw = await request.body()
        try:
            payload = json_numpy.loads(raw.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            return _error_response(400, f"failed to decode json_numpy body: {e}")

        try:
            top_cam = payload["top_cam"]
            left_cam = payload["left_cam"]
            right_cam = payload["right_cam"]
            instruction = str(payload["instruction"])
            state = payload["state"]
        except KeyError as e:
            return _error_response(400, f"missing required field: {e}")

        num_steps = int(payload.get("num_steps", DEFAULT_NUM_STEPS))
        enable_cuda_graph = bool(
            payload.get("enable_cuda_graph", policy.default_cuda_graph)
        )

        t0 = time.perf_counter()
        try:
            actions = policy.predict(
                top_cam=top_cam,
                left_cam=left_cam,
                right_cam=right_cam,
                instruction=instruction,
                state=state,
                num_steps=num_steps,
                enable_cuda_graph=enable_cuda_graph,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("inference failed")
            return _error_response(500, f"inference failed: {e}")
        dt_ms = (time.perf_counter() - t0) * 1000.0

        body = json_numpy.dumps({"actions": actions, "dt_ms": dt_ms})
        return Response(content=body, media_type="application/json")

    return app


def _error_response(status: int, message: str) -> Response:
    body = json_numpy.dumps({"error": message})
    return Response(content=body, status_code=status, media_type="application/json")


def warmup(policy: Policy) -> None:
    log.info("Warming up model with dummy frames (cuda_graph=%s) ...",
             policy.default_cuda_graph)
    dummy_img = np.zeros((180, 320, 3), dtype=np.uint8)
    dummy_state = np.zeros(STATE_DIM, dtype=np.float32)
    t0 = time.perf_counter()
    try:
        policy.predict(
            top_cam=dummy_img,
            left_cam=dummy_img,
            right_cam=dummy_img,
            instruction="warmup",
            state=dummy_state,
            num_steps=DEFAULT_NUM_STEPS,
            enable_cuda_graph=policy.default_cuda_graph,
        )
    except Exception:  # noqa: BLE001
        log.exception("warmup inference failed (server will still start)")
        return
    log.info("Warmup OK (%.1f ms)", (time.perf_counter() - t0) * 1000.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MolmoAct2-BimanualYAM inference server")
    p.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8202, help="bind port (default: 8202)")
    p.add_argument("--repo-id", default=REPO_ID, help=f"HF repo id (default: {REPO_ID})")
    p.add_argument("--device", default="cuda:0", help="torch device (default: cuda:0)")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="model dtype (default: bfloat16; fp32 needs ~26 GB)",
    )
    p.add_argument("--no-warmup", action="store_true", help="skip warmup pass")
    p.add_argument(
        "--cuda-graph",
        action="store_true",
        help="enable CUDA graph capture for action expert (faster but ~2 GB more VRAM)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    policy = Policy(
        repo_id=args.repo_id,
        device=args.device,
        dtype=dtype,
        enable_cuda_graph=args.cuda_graph,
    )
    if not args.no_warmup:
        warmup(policy)

    app = build_app(policy)

    import uvicorn

    log.info("Listening on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
