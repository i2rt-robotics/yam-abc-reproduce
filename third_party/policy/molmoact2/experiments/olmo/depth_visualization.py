from __future__ import annotations

from typing import Any

import numpy as np
import torch

from olmo.extra_tokens import DEFAULT_NUM_DEPTH_TOKENS


def load_depth_vae(
    *,
    device: torch.device,
    checkpoint_path: str | None = None,
    image_size: int = 320,
) -> Any:
    """Load the released depth VQ-VAE used by scripts/generate_depth_annotation.py."""
    from scripts.generate_depth_annotation import DEFAULT_VAE_CKPT, VQVAE, resolve_checkpoint

    resolved_checkpoint = resolve_checkpoint(checkpoint_path, DEFAULT_VAE_CKPT)
    vae = VQVAE(image_size=image_size).to(device)
    ckpt = torch.load(resolved_checkpoint, map_location="cpu")["weights"]
    if next(iter(ckpt)).startswith("module."):
        ckpt = {key.removeprefix("module."): value for key, value in ckpt.items()}
    vae.load_state_dict(ckpt)
    vae.eval()
    return vae


def depth_codes_to_rgb(
    codes: np.ndarray,
    *,
    depth_vae: Any | None = None,
    height: int = 10,
    width: int = 10,
    scale: int = 16,
    num_depth_tokens: int = DEFAULT_NUM_DEPTH_TOKENS,
) -> np.ndarray:
    total = int(height) * int(width)
    padded = np.zeros(total, dtype=np.int64)
    flat_codes = np.asarray(codes, dtype=np.int64).reshape(-1)
    flat_codes = np.clip(flat_codes, 0, int(num_depth_tokens) - 1)
    padded[: min(flat_codes.shape[0], total)] = flat_codes[: min(flat_codes.shape[0], total)]

    if depth_vae is not None:
        device = next(depth_vae.parameters()).device
        grid = torch.from_numpy(padded).long().reshape(1, height, width).to(device)
        with torch.no_grad():
            depth = depth_vae.decode(grid).squeeze().float().detach().cpu()
        depth_min = float(depth.min())
        depth_max = float(depth.max())
        if depth_max - depth_min > 1e-6:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = depth * 0.0
        gray = (depth * 255).clamp(0, 255).to(torch.uint8).numpy()
        return np.stack([gray, gray, gray], axis=-1)

    grid = padded.astype(np.float32).reshape(height, width)
    norm = np.clip(grid / float(int(num_depth_tokens) - 1), 0.0, 1.0)
    red = (norm * 255).astype(np.uint8)
    blue = ((1.0 - norm) * 255).astype(np.uint8)
    green = np.minimum(red, blue)
    rgb = np.stack([red, green, blue], axis=-1)
    return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
