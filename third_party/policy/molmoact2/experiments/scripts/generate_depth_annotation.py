#!/usr/bin/env python
"""Generate MolmoAct2 adaptive-depth annotations for a LeRobot dataset.

The output is a standalone LeRobot dataset next to the source dataset, named
``<dataset-name>_depth`` by default. The data parquet files are copied with
three extra per-frame columns:

- ``depth_codes``: VQ-VAE codes for the current frame depth map.
- ``buffer_codes``: depth token buffer after patch updates.
- ``depth_updated_mask``: boolean mask of updated depth patches.

This script intentionally does not import the original MolmoAct2 preprocessing
scripts. The Depth-Anything-V2 and VQ-VAE modules below match the small subset
needed to load the released checkpoints and generate the same annotations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import deque
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as nn_func
from huggingface_hub import hf_hub_download, snapshot_download
from torch import Tensor, nn
from torch.nn.init import trunc_normal_

# LeRobot rejects the deprecated LEROBOT_HOME env var during import. Preserve
# compatibility with older shells by translating it before importing lerobot modules.
if "HF_LEROBOT_HOME" not in os.environ and "LEROBOT_DATA_ROOT" in os.environ:
    os.environ["HF_LEROBOT_HOME"] = os.environ["LEROBOT_DATA_ROOT"]
if "LEROBOT_HOME" in os.environ:
    os.environ.setdefault("HF_LEROBOT_HOME", os.environ["LEROBOT_HOME"])
    del os.environ["LEROBOT_HOME"]

REPO_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
for candidate in (REPO_ROOT, LEROBOT_SRC):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from lerobot.utils.constants import HF_LEROBOT_HOME

DEFAULT_DEPTH_REPO = "allenai/MolmoAct-7B-D-0812"
DEFAULT_DEPTH_CKPT = "depth_anything_v2_vitb.pth"
DEFAULT_VAE_CKPT = "vae-final.pt"
DEPTH_COLUMNS = ("depth_codes", "buffer_codes", "depth_updated_mask")


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")


# ---------------------------------------------------------------------------
# Minimal Depth-Anything-V2 implementation for the released vitb checkpoint.
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, height, width = x.shape
        patch_h, patch_w = self.patch_size
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(f"DepthAnything input shape {(height, width)} is not divisible by {self.patch_size}.")
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, proj_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.attn_drop = nn.Dropout(0.0)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (query @ key.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ value).transpose(1, 2).reshape(batch, tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values: float = 1.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: Any = partial(nn.LayerNorm, eps=1e-6),
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)
        self.ls1 = LayerScale(dim, init_values=init_values)
        self.drop_path1 = nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=0.0,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values)
        self.drop_path2 = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def _named_apply(fn: Any, module: nn.Module, name: str = "", include_root: bool = False) -> None:
    if include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        full_name = ".".join((name, child_name)) if name else child_name
        _named_apply(fn, child_module, full_name, include_root=True)


def _init_weights_vit_timm(module: nn.Module, name: str = "") -> None:
    del name
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        init_values: float = 1.0,
        interpolate_offset: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.interpolate_offset = interpolate_offset
        self.interpolate_antialias = False
        self.num_register_tokens = 0
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    init_values=init_values,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Identity()
        self.init_weights()

    def init_weights(self) -> None:
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        _named_apply(_init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x: Tensor, width: int, height: int) -> Tensor:
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        num_pos = self.pos_embed.shape[1] - 1
        if npatch == num_pos and width == height:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        width0 = width // self.patch_size + self.interpolate_offset
        height0 = height // self.patch_size + self.interpolate_offset
        sqrt_num_pos = math.sqrt(num_pos)
        patch_pos_embed = nn_func.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_num_pos), int(sqrt_num_pos), dim).permute(0, 3, 1, 2),
            scale_factor=(float(width0) / sqrt_num_pos, float(height0) / sqrt_num_pos),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )
        if int(width0) != patch_pos_embed.shape[-2] or int(height0) != patch_pos_embed.shape[-1]:
            raise RuntimeError("DepthAnything positional embedding interpolation produced an unexpected shape.")
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x: Tensor) -> Tensor:
        batch, _, width, height = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(batch, -1, -1), x), dim=1)
        return x + self.interpolate_pos_encoding(x, width, height)

    def get_intermediate_layers(
        self,
        x: Tensor,
        n: Iterable[int],
        return_class_token: bool = True,
        norm: bool = True,
    ) -> tuple[Any, ...]:
        x = self.prepare_tokens_with_masks(x)
        outputs: list[Tensor] = []
        blocks_to_take = set(n)
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in blocks_to_take:
                outputs.append(x)
        if len(outputs) != len(blocks_to_take):
            raise RuntimeError(f"Only collected {len(outputs)} of {len(blocks_to_take)} requested DINO layers.")
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        patch_tokens = [out[:, 1:] for out in outputs]
        if return_class_token:
            return tuple(zip(patch_tokens, class_tokens, strict=True))
        return tuple(patch_tokens)


def _make_fusion_block(features: int, size: tuple[int, int] | None = None) -> FeatureFusionBlock:
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
    )


def _make_scratch(in_shape: list[int], out_shape: int) -> nn.Module:
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    return scratch


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, activation: nn.Module, bn: bool) -> None:
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        features: int,
        activation: nn.Module,
        deconv: bool = False,
        bn: bool = False,
        expand: bool = False,
        align_corners: bool = True,
        size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        del deconv
        self.align_corners = align_corners
        self.size = size
        out_features = features // 2 if expand else features
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True)
        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs: Tensor, size: tuple[int, int] | None = None) -> Tensor:
        output = xs[0]
        if len(xs) == 2:
            output = self.skip_add.add(output, self.resConfUnit1(xs[1]))
        output = self.resConfUnit2(output)
        if size is None and self.size is None:
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}
        output = nn_func.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        return self.out_conv(output)


class DPTHead(nn.Module):
    def __init__(self, in_channels: int, features: int = 128, out_channels: list[int] | None = None) -> None:
        super().__init__()
        out_channels = out_channels or [96, 192, 384, 768]
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(in_channels=in_channels, out_channels=out_channel, kernel_size=1, stride=1, padding=0)
                for out_channel in out_channels
            ]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )
        self.scratch = _make_scratch(out_channels, features)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features)
        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

    def forward(self, out_features: tuple[Any, ...], patch_h: int, patch_w: int) -> Tensor:
        out = []
        for idx, x in enumerate(out_features):
            x = x[0]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[idx](x)
            x = self.resize_layers[idx](x)
            out.append(x)
        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        out = self.scratch.output_conv1(path_1)
        out = nn_func.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        return self.scratch.output_conv2(out)


class DepthAnythingV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.intermediate_layer_idx = [2, 5, 8, 11]
        self.pretrained = DinoVisionTransformer()
        self.depth_head = DPTHead(self.pretrained.embed_dim, features=128, out_channels=[96, 192, 384, 768])

    def forward(self, x: Tensor) -> Tensor:
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(
            x, self.intermediate_layer_idx, return_class_token=True
        )
        depth = self.depth_head(features, patch_h, patch_w)
        depth = nn_func.relu(depth)
        return depth.squeeze(1)


class Resize:
    def __init__(
        self,
        width: int,
        height: int,
        keep_aspect_ratio: bool = True,
        ensure_multiple_of: int = 14,
        resize_method: str = "lower_bound",
        image_interpolation_method: int = cv2.INTER_CUBIC,
    ) -> None:
        self.width = width
        self.height = height
        self.keep_aspect_ratio = keep_aspect_ratio
        self.multiple_of = ensure_multiple_of
        self.resize_method = resize_method
        self.image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x: float, min_val: int = 0, max_val: int | None = None) -> int:
        y = int(np.round(x / self.multiple_of) * self.multiple_of)
        if max_val is not None and y > max_val:
            y = int(np.floor(x / self.multiple_of) * self.multiple_of)
        if y < min_val:
            y = int(np.ceil(x / self.multiple_of) * self.multiple_of)
        return y

    def get_size(self, width: int, height: int) -> tuple[int, int]:
        scale_height = self.height / height
        scale_width = self.width / width
        if self.keep_aspect_ratio:
            if self.resize_method == "lower_bound":
                if scale_width > scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            else:
                raise ValueError(f"Unsupported resize_method={self.resize_method!r}.")
        new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.height)
        new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.width)
        return new_width, new_height

    def __call__(self, image: np.ndarray) -> np.ndarray:
        width, height = self.get_size(image.shape[1], image.shape[0])
        return cv2.resize(image, (width, height), interpolation=self.image_interpolation_method)


def depth_image_to_tensor(raw_bgr: np.ndarray, input_size: int, device: torch.device) -> tuple[Tensor, tuple[int, int]]:
    height, width = raw_bgr.shape[:2]
    image = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB) / 255.0
    image = Resize(width=input_size, height=input_size)(image)
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1))).astype(np.float32)
    return torch.from_numpy(image).to(device), (height, width)


# ---------------------------------------------------------------------------
# Minimal VQ-VAE implementation for the released depth tokenizer checkpoint.
# ---------------------------------------------------------------------------


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float, decay: float) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self.register_buffer("_embedding", torch.empty(num_embeddings, embedding_dim))
        self._embedding.data.normal_()
        self._commitment_cost = commitment_cost
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("_ema_w", torch.empty(num_embeddings, embedding_dim))
        self._ema_w.data.normal_()
        self._decay = decay
        self._epsilon = 1e-5

    def forward_indice(self, encoding_indices: Tensor) -> Tensor:
        input_shape = encoding_indices.shape
        encoding_indices = encoding_indices.reshape((-1, 1))
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=encoding_indices.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding).view(
            input_shape[0], input_shape[1], input_shape[2], self._embedding_dim
        )
        return quantized.permute(0, 3, 1, 2).contiguous()

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding).view(input_shape)
        e_latent_loss = nn_func.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return (
            loss,
            quantized.permute(0, 3, 1, 2).contiguous(),
            perplexity,
            encodings,
            encoding_indices.view(input_shape[0:3]),
        )


class ResBlockV1(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) + x


class VQVAE(nn.Module):
    def __init__(
        self,
        image_size: int = 320,
        num_tokens: int = 128,
        codebook_dim: int = 512,
        num_resnet_blocks: int = 2,
        hidden_dim: int = 16,
        channels: int = 1,
        loss_type: str = "mse",
        max_value: float = 10.0,
        downsample_ratio: int = 32,
        use_norm: bool = False,
        train_objective: str = "regression",
        residul_type: str = "v1",
        ema_decay: float = 0.99,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        if residul_type != "v1":
            raise ValueError(f"Only residul_type='v1' is supported, got {residul_type!r}.")
        if loss_type != "mse":
            raise ValueError(f"Only loss_type='mse' is supported, got {loss_type!r}.")
        self.hidden_dim = hidden_dim
        self.num_resnet_blocks = num_resnet_blocks
        self.image_size = image_size
        self.num_tokens = num_tokens
        self.downsample_ratio = downsample_ratio
        self.use_norm = use_norm
        self.loss_type = loss_type
        self.train_objective = train_objective
        self.max_value = max_value
        self.residul_type = residul_type
        layer_num = int(math.log2(downsample_ratio) - 1)
        self.layer_num = layer_num
        dim = hidden_dim
        enc_layers: list[nn.Module] = [nn.Conv2d(1, dim, 4, stride=2, padding=1), nn.ReLU()]
        for _ in range(layer_num):
            enc_layers.extend([nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1), nn.ReLU()])
            dim *= 2
        for _ in range(num_resnet_blocks):
            enc_layers.append(ResBlockV1(dim))
        enc_layers.append(nn.Conv2d(dim, codebook_dim, 1))
        self.encoder = nn.Sequential(*enc_layers)
        dim = hidden_dim * downsample_ratio // 2
        dec_layers: list[nn.Module] = [nn.Conv2d(codebook_dim, dim, 1), nn.ReLU()]
        for _ in range(num_resnet_blocks):
            dec_layers.append(ResBlockV1(dim))
        dec_layers.append(nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1))
        dec_layers.append(nn.ReLU())
        for _ in range(layer_num):
            dec_layers.append(nn.ConvTranspose2d(dim, dim // 2, 4, stride=2, padding=1))
            dec_layers.append(nn.ReLU())
            dim = dim // 2
        dec_layers.append(nn.Conv2d(dim, channels, 1))
        self.decoder = nn.Sequential(*dec_layers)
        self.loss_fn = nn_func.mse_loss
        self._vq_vae = VectorQuantizerEMA(num_tokens, codebook_dim, commitment_cost, ema_decay)

    def norm(self, images: Tensor) -> Tensor:
        return 2.0 * (images / self.max_value - 0.5)

    def denorm(self, images: Tensor) -> Tensor:
        return (images * 0.5 + 0.5) * self.max_value

    def decode(self, img_seq: Tensor) -> Tensor:
        quantized = self._vq_vae.forward_indice(img_seq)
        out = self.decoder(quantized)
        return self.denorm(out) if self.use_norm else out

    def forward(
        self,
        img: Tensor,
        label_img: Tensor | None = None,
        use_norm: bool | None = None,
        return_indices: bool = False,
    ) -> Any:
        label_img = img.clone().float() if label_img is None else label_img.float()
        img = img.float()
        use_norm = self.use_norm if use_norm is None else use_norm
        if use_norm:
            img = self.norm(img)
            label_img = self.norm(label_img)
        logits = self.encoder(img)
        vq_loss, quantized, _, _, code_indices = self._vq_vae(logits)
        if return_indices:
            return code_indices
        out = self.decoder(quantized)
        recon_loss = self.loss_fn(out, label_img)
        total_loss = recon_loss + vq_loss
        if use_norm:
            out = self.denorm(out)
        return total_loss, recon_loss, out


# ---------------------------------------------------------------------------
# Dataset and annotation helpers.
# ---------------------------------------------------------------------------


@dataclass
class DatasetLocation:
    repo_id: str
    root: Path
    output_root: Path


@dataclass
class EpisodeImages:
    episode_index: int
    row_indices: list[int]
    bgr_images: list[np.ndarray]


def default_output_root(dataset_root: Path, repo_id: str, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    depth_root = os.environ.get("LEROBOT_DEPTH_DATA_ROOT")
    if depth_root:
        return Path(depth_root).expanduser() / str(repo_id)
    return dataset_root.with_name(f"{dataset_root.name}_depth")


def resolve_checkpoint(path_or_filename: str | None, filename: str) -> Path:
    if path_or_filename:
        path = Path(path_or_filename).expanduser()
        if path.exists():
            return path
        return Path(hf_hub_download(repo_id=DEFAULT_DEPTH_REPO, filename=path_or_filename, token=_hf_token()))
    return Path(hf_hub_download(repo_id=DEFAULT_DEPTH_REPO, filename=filename, token=_hf_token()))


def resolve_dataset(repo_id_or_path: str, output_dir: str | None, revision: str | None) -> DatasetLocation:
    maybe_path = Path(repo_id_or_path).expanduser()
    if maybe_path.exists():
        root = maybe_path.resolve()
        repo_id = root.name
    else:
        repo_id = repo_id_or_path
        root = HF_LEROBOT_HOME / repo_id
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=root,
            token=_hf_token(),
        )
    out_root = default_output_root(root, repo_id, output_dir)
    return DatasetLocation(repo_id=repo_id, root=root, output_root=out_root)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=4) + "\n")


def validate_camera(info: dict[str, Any], camera_key: str) -> str:
    features = info.get("features")
    if not isinstance(features, dict) or camera_key not in features:
        available = sorted(k for k, v in (features or {}).items() if isinstance(v, dict) and v.get("dtype") in {"image", "video"})
        raise ValueError(f"camera_key={camera_key!r} not found. Available image/video keys: {available}")
    dtype = features[camera_key].get("dtype")
    if dtype not in {"image", "video"}:
        raise ValueError(f"camera_key={camera_key!r} must be an image or video feature, got dtype={dtype!r}.")
    return str(dtype)


def setup_output_dataset(
    dataset_root: Path,
    output_root: Path,
    info: dict[str, Any],
    camera_key: str,
    *,
    grid_size: int,
    vae_image_size: int,
    threshold: float,
    depth_input_size: int,
    overwrite: bool,
) -> None:
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    meta_src = dataset_root / "meta"
    meta_dst = output_root / "meta"
    shutil.copytree(meta_src, meta_dst, dirs_exist_ok=True)
    readme_src = dataset_root / "README.md"
    if readme_src.exists() and not (output_root / "README.md").exists():
        shutil.copy2(readme_src, output_root / "README.md")
    marker_src = dataset_root / ".lerobot_download_complete"
    if marker_src.exists():
        shutil.copy2(marker_src, output_root / ".lerobot_download_complete")

    videos_src = dataset_root / "videos"
    videos_dst = output_root / "videos"
    if videos_src.exists() and not videos_dst.exists():
        with suppress(FileExistsError):
            videos_dst.symlink_to(videos_src)

    updated = dict(info)
    updated["features"] = dict(updated["features"])
    updated["features"]["depth_codes"] = {
        "dtype": "int16",
        "shape": [grid_size],
        "names": None,
        "description": "Per-frame VQVAE depth codes.",
    }
    updated["features"]["buffer_codes"] = {
        "dtype": "int16",
        "shape": [grid_size],
        "names": None,
        "description": "Depth token buffer state after patch updates.",
    }
    updated["features"]["depth_updated_mask"] = {
        "dtype": "bool",
        "shape": [grid_size],
        "names": None,
        "description": "Mask of depth patches updated in the buffer.",
    }
    updated["depth_vae"] = {
        "vae_image_size": vae_image_size,
        "patch_size": 32,
        "grid_h": vae_image_size // 32,
        "grid_w": vae_image_size // 32,
        "num_tokens": 128,
        "cosine_threshold": threshold,
        "depth_encoder": "vitb",
        "depth_input_size": depth_input_size,
        "image_key": camera_key,
    }
    write_json(meta_dst / "info.json", updated)


def decode_image_cell(cell: Any, dataset_root: Path) -> np.ndarray:
    if isinstance(cell, dict):
        image_bytes = cell.get("bytes")
        if image_bytes:
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("Failed to decode image bytes from parquet.")
            return image
        image_path = cell.get("path")
        if image_path:
            path = Path(image_path)
            if not path.is_absolute():
                path = dataset_root / path
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read image path {path}.")
            return image
    raise ValueError(f"Unsupported image cell type: {type(cell).__name__}.")


def cosine_similarity_patches(img_a: np.ndarray, img_b: np.ndarray, patch_size: int) -> np.ndarray:
    height, width, _ = img_a.shape
    patches_h = height // patch_size
    patches_w = width // patch_size
    a = img_a[: patches_h * patch_size, : patches_w * patch_size].astype(np.float32)
    b = img_b[: patches_h * patch_size, : patches_w * patch_size].astype(np.float32)
    a = a.reshape(patches_h, patch_size, patches_w, patch_size, 3).transpose(0, 2, 1, 3, 4)
    b = b.reshape(patches_h, patch_size, patches_w, patch_size, 3).transpose(0, 2, 1, 3, 4)
    a = a.reshape(patches_h, patches_w, -1)
    b = b.reshape(patches_h, patches_w, -1)
    dot = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.where(denom < 1e-8, 1.0, dot / (denom + 1e-12)).astype(np.float32)


@torch.no_grad()
def encode_depth_codes(
    bgr_images: list[np.ndarray],
    depth_model: DepthAnythingV2,
    vae: VQVAE,
    *,
    depth_input_size: int,
    vae_image_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    exact = batch_size == 1
    rgb_resized: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    for start in range(0, len(bgr_images), batch_size):
        batch_images = bgr_images[start : start + batch_size]
        tensors: list[Tensor] = []
        original_sizes: list[tuple[int, int]] = []
        for bgr in batch_images:
            tensor, size = depth_image_to_tensor(bgr, depth_input_size, device)
            tensors.append(tensor)
            original_sizes.append(size)
            rgb_resized.append(cv2.resize(bgr, (vae_image_size, vae_image_size), interpolation=cv2.INTER_LINEAR))

        if exact:
            for idx, tensor in enumerate(tensors):
                depth = depth_model(tensor[None])[0]
                height, width = original_sizes[idx]
                resized_depth = nn_func.interpolate(
                    depth[None, None],
                    (height, width),
                    mode="bilinear",
                    align_corners=True,
                )[0, 0]
                raw = resized_depth.detach().cpu().numpy()
                dmin = raw.min()
                dmax = raw.max()
                depth_vis = ((raw - dmin) / (dmax - dmin + 1e-8) * 255.0).astype(np.uint8)
                depth_vis = cv2.resize(
                    depth_vis,
                    (vae_image_size, vae_image_size),
                    interpolation=cv2.INTER_NEAREST,
                )
                depth_tensor = torch.from_numpy(depth_vis).float() / 255.0
                depth_tensor = (depth_tensor - 0.5) / 0.5
                depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0).to(device)
                code_grid = vae(img=depth_tensor, return_indices=True)[0]
                codes.append(code_grid.detach().cpu().numpy().astype(np.int16))
        else:
            codes_by_index: list[np.ndarray | None] = [None] * len(tensors)
            shape_groups: dict[tuple[int, int, int, int], list[int]] = {}
            for idx, tensor in enumerate(tensors):
                input_height, input_width = tensor.shape[-2:]
                original_height, original_width = original_sizes[idx]
                shape_groups.setdefault((input_height, input_width, original_height, original_width), []).append(idx)
            for indices in shape_groups.values():
                input_height, input_width = tensors[indices[0]].shape[-2:]
                original_height, original_width = original_sizes[indices[0]]
                max_depth_batch = max(1, (2**31 - 1) // (64 * input_height * input_width))
                for chunk_start in range(0, len(indices), max_depth_batch):
                    chunk_indices = indices[chunk_start : chunk_start + max_depth_batch]
                    pixel_values = torch.stack([tensors[idx] for idx in chunk_indices], dim=0)
                    depth_batch = depth_model(pixel_values)
                    resized_depth = nn_func.interpolate(
                        depth_batch[:, None],
                        (original_height, original_width),
                        mode="bilinear",
                        align_corners=True,
                    )[:, 0]
                    flat_depth = resized_depth.flatten(1)
                    dmin = flat_depth.min(dim=1).values[:, None, None]
                    dmax = flat_depth.max(dim=1).values[:, None, None]
                    depth_vis = ((resized_depth - dmin) / (dmax - dmin + 1e-8) * 255.0).clamp(0, 255)
                    depth_vis = depth_vis.to(torch.uint8).to(torch.float32)
                    depth_tensor = nn_func.interpolate(
                        depth_vis[:, None],
                        (vae_image_size, vae_image_size),
                        mode="nearest",
                    )
                    depth_tensor = (depth_tensor / 255.0 - 0.5) / 0.5
                    code_grids = vae(img=depth_tensor, return_indices=True)
                    code_grids_np = code_grids.detach().cpu().numpy().astype(np.int16)
                    for local_idx, original_idx in enumerate(chunk_indices):
                        codes_by_index[original_idx] = code_grids_np[local_idx]
            if any(code_grid is None for code_grid in codes_by_index):
                raise RuntimeError("Internal error: missing batched depth code output.")
            codes.extend(code_grid for code_grid in codes_by_index if code_grid is not None)
    return rgb_resized, codes


def process_episode_frames(
    bgr_images: list[np.ndarray],
    depth_model: DepthAnythingV2,
    vae: VQVAE,
    *,
    vae_image_size: int,
    depth_input_size: int,
    threshold: float,
    batch_size: int,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    grid_h = vae_image_size // 32
    grid_w = vae_image_size // 32
    rgb_resized, code_grids = encode_depth_codes(
        bgr_images,
        depth_model,
        vae,
        depth_input_size=depth_input_size,
        vae_image_size=vae_image_size,
        batch_size=batch_size,
        device=device,
    )
    token_buffer = code_grids[0].copy()
    all_depth_codes: list[np.ndarray] = []
    all_buffer_codes: list[np.ndarray] = []
    all_updated_masks: list[np.ndarray] = []
    for idx, code_grid in enumerate(code_grids):
        if idx == 0:
            updated_mask = np.ones((grid_h, grid_w), dtype=bool)
        else:
            sim = cosine_similarity_patches(rgb_resized[idx - 1], rgb_resized[idx], 32)
            updated_mask = sim < threshold
            token_buffer[updated_mask] = code_grid[updated_mask]
        all_depth_codes.append(code_grid.reshape(-1).astype(np.int16))
        all_buffer_codes.append(token_buffer.reshape(-1).astype(np.int16))
        all_updated_masks.append(updated_mask.reshape(-1))
    return all_depth_codes, all_buffer_codes, all_updated_masks


def load_episode_metadata(dataset_root: Path, camera_key: str) -> dict[int, dict[str, Any]]:
    episodes_dir = dataset_root / "meta" / "episodes"
    metadata: dict[int, dict[str, Any]] = {}
    for chunk_dir in sorted(episodes_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue
        for parquet_path in sorted(chunk_dir.glob("*.parquet")):
            table = pq.read_table(parquet_path)
            data = table.to_pydict()
            chunk_col = f"videos/{camera_key}/chunk_index"
            file_col = f"videos/{camera_key}/file_index"
            from_col = f"videos/{camera_key}/from_timestamp"
            to_col = f"videos/{camera_key}/to_timestamp"
            missing = [col for col in (chunk_col, file_col, from_col, to_col) if col not in data]
            if missing:
                raise ValueError(f"Video camera_key={camera_key!r} is missing metadata columns: {missing}")
            for row_idx in range(table.num_rows):
                episode_index = int(data["episode_index"][row_idx])
                metadata[episode_index] = {
                    "video_chunk": int(data[chunk_col][row_idx]),
                    "video_file": int(data[file_col][row_idx]),
                    "from_timestamp": float(data[from_col][row_idx]),
                    "to_timestamp": float(data[to_col][row_idx]),
                }
    return metadata


def video_path_for_episode(dataset_root: Path, info: dict[str, Any], camera_key: str, metadata: dict[str, Any]) -> Path:
    pattern = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    return dataset_root / pattern.format(
        video_key=camera_key,
        chunk_index=int(metadata["video_chunk"]),
        file_index=int(metadata["video_file"]),
    )


def decode_video_frames(video_path: Path, from_timestamp: float, n_frames: int) -> list[np.ndarray]:
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        target_pts = int(from_timestamp / stream.time_base)
        if from_timestamp > 0:
            container.seek(target_pts, stream=stream)
        frames: list[np.ndarray] = []
        for frame in container.decode(video=0):
            if frame.pts is not None and frame.pts < target_pts:
                continue
            frames.append(frame.to_ndarray(format="bgr24"))
            if len(frames) >= n_frames:
                break
    finally:
        container.close()
    if len(frames) != n_frames:
        raise RuntimeError(f"Expected {n_frames} frames from {video_path}, got {len(frames)}.")
    return frames


def decode_episode_images(
    *,
    episode_index: int,
    rows: list[tuple[int, int]],
    dataset_root: Path,
    info: dict[str, Any],
    camera_key: str,
    camera_dtype: str,
    image_column: list[Any] | None,
    video_metadata: dict[int, dict[str, Any]] | None,
) -> EpisodeImages:
    rows.sort(key=lambda x: x[1])
    row_indices = [row_idx for row_idx, _ in rows]
    if camera_dtype == "image":
        if image_column is None:
            raise RuntimeError(f"Missing image column for camera_key={camera_key!r}.")
        bgr_images = [decode_image_cell(image_column[row_idx], dataset_root) for row_idx in row_indices]
    else:
        if video_metadata is None or episode_index not in video_metadata:
            raise RuntimeError(f"Missing video metadata for episode {episode_index}.")
        video_path = video_path_for_episode(dataset_root, info, camera_key, video_metadata[episode_index])
        bgr_images = decode_video_frames(video_path, video_metadata[episode_index]["from_timestamp"], len(row_indices))
    return EpisodeImages(episode_index=episode_index, row_indices=row_indices, bgr_images=bgr_images)


def iter_decoded_episodes(
    *,
    episodes: dict[int, list[tuple[int, int]]],
    dataset_root: Path,
    info: dict[str, Any],
    camera_key: str,
    camera_dtype: str,
    image_column: list[Any] | None,
    video_metadata: dict[int, dict[str, Any]] | None,
    num_workers: int,
) -> Iterable[EpisodeImages]:
    sorted_episodes = sorted(episodes.items())
    if num_workers <= 1 or len(sorted_episodes) <= 1:
        for episode_index, rows in sorted_episodes:
            yield decode_episode_images(
                episode_index=episode_index,
                rows=list(rows),
                dataset_root=dataset_root,
                info=info,
                camera_key=camera_key,
                camera_dtype=camera_dtype,
                image_column=image_column,
                video_metadata=video_metadata,
            )
        return

    episode_iter = iter(sorted_episodes)
    pending = deque()
    max_pending = min(num_workers, len(sorted_episodes))

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            episode_index, rows = next(episode_iter)
        except StopIteration:
            return False
        pending.append(
            executor.submit(
                decode_episode_images,
                episode_index=episode_index,
                rows=list(rows),
                dataset_root=dataset_root,
                info=info,
                camera_key=camera_key,
                camera_dtype=camera_dtype,
                image_column=image_column,
                video_metadata=video_metadata,
            )
        )
        return True

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for _ in range(max_pending):
            submit_next(executor)
        while pending:
            future = pending.popleft()
            yield future.result()
            submit_next(executor)


def table_has_depth_columns(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        names = set(pq.read_schema(path).names)
    except Exception:
        return False
    return all(col in names for col in DEPTH_COLUMNS)


def without_depth_columns(table: pa.Table) -> pa.Table:
    existing = [col for col in DEPTH_COLUMNS if col in table.column_names]
    return table.drop(existing) if existing else table


def process_parquet_file(
    *,
    parquet_path: Path,
    output_path: Path,
    dataset_root: Path,
    info: dict[str, Any],
    camera_key: str,
    camera_dtype: str,
    depth_model: DepthAnythingV2,
    vae: VQVAE,
    depth_input_size: int,
    vae_image_size: int,
    threshold: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    video_metadata: dict[int, dict[str, Any]] | None,
    overwrite: bool,
) -> None:
    if not overwrite and table_has_depth_columns(output_path):
        print(f"skip complete: {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(parquet_path)
    data = table.to_pydict()
    n_rows = table.num_rows
    episodes: dict[int, list[tuple[int, int]]] = {}
    for row_idx, episode_index in enumerate(data["episode_index"]):
        episodes.setdefault(int(episode_index), []).append((row_idx, int(data["frame_index"][row_idx])))

    all_depth_codes: list[list[int] | None] = [None] * n_rows
    all_buffer_codes: list[list[int] | None] = [None] * n_rows
    all_updated_masks: list[list[bool] | None] = [None] * n_rows

    image_column = data.get(camera_key) if camera_dtype == "image" else None
    for episode_images in iter_decoded_episodes(
        episodes=episodes,
        dataset_root=dataset_root,
        info=info,
        camera_key=camera_key,
        camera_dtype=camera_dtype,
        image_column=image_column,
        video_metadata=video_metadata,
        num_workers=num_workers,
    ):
        ep_depth_codes, ep_buffer_codes, ep_updated_masks = process_episode_frames(
            episode_images.bgr_images,
            depth_model,
            vae,
            vae_image_size=vae_image_size,
            depth_input_size=depth_input_size,
            threshold=threshold,
            batch_size=batch_size,
            device=device,
        )
        for idx, row_idx in enumerate(episode_images.row_indices):
            all_depth_codes[row_idx] = ep_depth_codes[idx].tolist()
            all_buffer_codes[row_idx] = ep_buffer_codes[idx].tolist()
            all_updated_masks[row_idx] = ep_updated_masks[idx].tolist()

    if any(value is None for value in all_depth_codes + all_buffer_codes + all_updated_masks):
        raise RuntimeError(f"Failed to annotate all rows in {parquet_path}.")

    table = without_depth_columns(table)
    table = table.append_column("depth_codes", pa.array(all_depth_codes, type=pa.list_(pa.int16())))
    table = table.append_column("buffer_codes", pa.array(all_buffer_codes, type=pa.list_(pa.int16())))
    table = table.append_column("depth_updated_mask", pa.array(all_updated_masks, type=pa.list_(pa.bool_())))
    pq.write_table(table, output_path)
    print(f"wrote: {output_path} rows={n_rows}")


def iter_data_files(dataset_root: Path) -> list[Path]:
    data_root = dataset_root / "data"
    return [path for chunk in sorted(data_root.iterdir()) if chunk.is_dir() for path in sorted(chunk.glob("*.parquet"))]


def load_models(depth_ckpt: Path, vae_ckpt: Path, vae_image_size: int, device: torch.device) -> tuple[DepthAnythingV2, VQVAE]:
    depth_model = DepthAnythingV2().to(device).eval()
    depth_state = torch.load(depth_ckpt, map_location="cpu")
    depth_model.load_state_dict(depth_state)

    vae = VQVAE(image_size=vae_image_size).to(device).eval()
    vae_state = torch.load(vae_ckpt, map_location="cpu")["weights"]
    if next(iter(vae_state)).startswith("module."):
        vae_state = {key.removeprefix("module."): value for key, value in vae_state.items()}
    vae.load_state_dict(vae_state)
    return depth_model, vae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MolmoAct2 depth annotations.")
    parser.add_argument("repo_id", help="LeRobot dataset repo id or local dataset directory.")
    parser.add_argument("--camera-key", required=True, help="Required image/video feature key to annotate.")
    parser.add_argument("--num-workers", type=int, default=4, help="CPU workers for per-episode image/video decoding.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="GPU inference batch size. Use 1 for exact original preprocessing behavior.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output dataset directory. Defaults to $LEROBOT_DEPTH_DATA_ROOT/<repo_id> "
            "when set, otherwise <dataset>_depth next to the source dataset."
        ),
    )
    parser.add_argument("--revision", default=None, help="Hub dataset revision.")
    parser.add_argument("--depth-anything-ckpt", default=None, help="Local path or HF filename for DepthAnythingV2 vitb.")
    parser.add_argument("--vae-ckpt", default=None, help="Local path or HF filename for depth VQ-VAE.")
    parser.add_argument("--threshold", type=float, default=0.996, help="RGB patch cosine threshold for updates.")
    parser.add_argument("--depth-input-size", type=int, default=518)
    parser.add_argument("--vae-image-size", type=int, default=320)
    parser.add_argument("--max-files", type=int, default=None, help="Optional smoke-test limit on data parquet files.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split data parquet files into this many shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index processed by this worker.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output dataset/files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.vae_image_size % 32 != 0:
        raise ValueError("--vae-image-size must be divisible by 32.")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards.")

    location = resolve_dataset(args.repo_id, args.output_dir, args.revision)
    info = load_json(location.root / "meta" / "info.json")
    camera_dtype = validate_camera(info, args.camera_key)
    grid_size = (args.vae_image_size // 32) ** 2
    setup_output_dataset(
        location.root,
        location.output_root,
        info,
        args.camera_key,
        grid_size=grid_size,
        vae_image_size=args.vae_image_size,
        threshold=args.threshold,
        depth_input_size=args.depth_input_size,
        overwrite=bool(args.overwrite),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"dataset: {location.root}")
    print(f"output: {location.output_root}")
    print(f"camera_key: {args.camera_key} ({camera_dtype})")
    print(f"device: {device}")
    print(f"batch_size: {args.batch_size} ({'exact' if args.batch_size == 1 else 'batched'})")

    all_files = iter_data_files(location.root)
    if args.max_files is not None:
        all_files = all_files[: args.max_files]
    files = [path for idx, path in enumerate(all_files) if idx % args.num_shards == args.shard_index]
    pending_files = (
        files
        if args.overwrite
        else [
            parquet_path
            for parquet_path in files
            if not table_has_depth_columns(location.output_root / parquet_path.relative_to(location.root))
        ]
    )
    print(f"parquet files: {len(all_files)} total, {len(files)} for shard {args.shard_index}/{args.num_shards}")
    print(f"pending files: {len(pending_files)}")
    if not pending_files:
        print(f"done: {location.output_root}")
        return

    depth_ckpt = resolve_checkpoint(args.depth_anything_ckpt, DEFAULT_DEPTH_CKPT)
    vae_ckpt = resolve_checkpoint(args.vae_ckpt, DEFAULT_VAE_CKPT)
    print(f"depth_anything_ckpt: {depth_ckpt}")
    print(f"vae_ckpt: {vae_ckpt}")
    depth_model, vae = load_models(depth_ckpt, vae_ckpt, args.vae_image_size, device)

    video_metadata = load_episode_metadata(location.root, args.camera_key) if camera_dtype == "video" else None
    for idx, parquet_path in enumerate(pending_files, start=1):
        rel_path = parquet_path.relative_to(location.root)
        output_path = location.output_root / rel_path
        print(f"[{idx}/{len(pending_files)}] {rel_path}")
        process_parquet_file(
            parquet_path=parquet_path,
            output_path=output_path,
            dataset_root=location.root,
            info=info,
            camera_key=args.camera_key,
            camera_dtype=camera_dtype,
            depth_model=depth_model,
            vae=vae,
            depth_input_size=args.depth_input_size,
            vae_image_size=args.vae_image_size,
            threshold=args.threshold,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            video_metadata=video_metadata,
            overwrite=bool(args.overwrite),
        )
    print(f"done: {location.output_root}")


if __name__ == "__main__":
    main()
