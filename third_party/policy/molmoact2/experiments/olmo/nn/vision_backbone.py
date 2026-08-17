import logging
import math
import os
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Tuple, Optional, List, Union, Literal

import einops
import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from olmo.config import BaseConfig, D, StrEnum
from olmo.nn.image_vit import VitConfig, VisionTransformer
from olmo.nn.llm import Activation
from olmo.nn.cp_load_balancer import CPLoadBalancerType, CPLoadBalancer
from olmo.preprocessing.image_preprocessor import ImagePreprocessor, normalize_image_augmentation_mode
from olmo.torch_util import freeze_module
from olmo.util import resource_path
from torch.nn import functional as F
from torch.distributed.fsdp import fully_shard
from torch.distributed.nn.functional import all_gather as differentiable_all_gather

log = logging.getLogger(__name__)

class ImagePaddingEmbed(StrEnum):
    """How to embed image padding information"""
    pad_and_partial_pad = "pad_and_partial_pad"
    pad_embed = "pad_embed"
    regress = "regress"


class ImagePooling2DType(StrEnum):
    """How to pool patch features"""
    attention = "attention"
    attention_meanq = "attention_meanq"
    attention_meanq_2x = "attention_meanq_2x"
    attention_meanq_4x = "attention_meanq_4x"
    attention_2wide = "attention_2wide"
    mean = "mean"
    none = "none"
    stack = "stack"


class ImageProjectType(StrEnum):
    """How to project the pooled features into the LLM embedding space"""
    random_linear = "random_linear"
    mlp = "mlp"
    mlpx2 = "2mlp"
    linear = "linear"


@dataclass
class MolmoVisionBackboneConfig(BaseConfig):
    """Vision ViT and the Image/Language Connector"""

    vit: VitConfig = field(default_factory=VitConfig)
    """The vision ViT"""

    image_pooling_2d: ImagePooling2DType = ImagePooling2DType.attention_meanq
    """Layer to pool image features"""

    pooling_attention_mask: bool = False
    """Use an attention mask when pooling instead setting masked embeddings to 0"""

    image_projector: ImageProjectType = ImageProjectType.mlp
    """Layer to project pooled image features to the LLM embedding space"""

    image_padding_embed: Optional[ImagePaddingEmbed] = None
    """
    Image padding mode to use to tell the model what parts of the image are padding
    """

    vit_layers: Tuple = (-1,)
    """What layers to use from the VIT"""

    skip_unused_layers: bool = True
    """Don't load layers we don't need from the ViT"""

    use_deepstack: bool = False
    """Use deepstack"""

    share_connector: bool = False
    """Share the connector across layers"""

    image_feature_dropout: float = 0.0
    """Dropout for image patch features"""

    connector_activation_checkpointing: bool = True
    """Allow activation checkpoint on the connector components"""

    compile_vit: Optional[str] = "blocks"
    """How to compile the ViT"""

    pool_size_embeds: Optional[List[int]] = None

    compile_connector: Optional[str] = "dynamic"

    normalize_on_gpu: bool = False
    """Run image normalization on the GPU
    
    Does this will allow image loading to keep the images in uint8 which will reduce 
    RAM/shared memory usage significantly
    """
    use_image_augmentation: Union[bool, str] = False
    """Image augmentation mode during preprocessing: none, photometric, or full"""

    def __post_init__(self):
        self.vit_layers = tuple(self.vit_layers)  # type: ignore[assignment]
        self.use_image_augmentation = normalize_image_augmentation_mode(self.use_image_augmentation)

    def build_preprocessor(self):
        return ImagePreprocessor(
            normalize=self.vit.normalize,
            resize=self.vit.resize_mode,
            pad_value=self.vit.pad_value,
            image_patch_size=self.vit.image_patch_size,
            base_image_input_size=self.vit.image_default_input_size,
            normalize_on_gpu=self.normalize_on_gpu,
            use_image_mask=self.image_padding_embed is not None,
            use_image_augmentation=self.use_image_augmentation,
        )

    def build(self, llm_config, device):
        return MolmoVisionBackbone(self, llm_config, device)

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        if "fix_image_padding" in config:
            assert config.image_padding_embed is None or config.fix_image_padding
            del config["fix_image_padding"]
        for k in ["residual_patch_features", "patch_residual", "patch_residual_thresh", "debug", "frame_embedding"]:
            if k in config:
                assert not config[k]
                del config[k]
        for k in ["image_pooling_h", "image_pooling_w"]:
            if k in config:
                assert config.pop(k) == 2
        if "use_image_augmentation" in config:
            config.use_image_augmentation = normalize_image_augmentation_mode(config.use_image_augmentation)
        config.vit = VitConfig.update_legacy_settings(config.vit)
        return config


class ImageProjectorMLP(nn.Module):
    """MLP used for the image projector"""

    def __init__(self, config, input_dim: int, dropout: float = 0.0, device=None):
        super().__init__()
        self.hidden_size = config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        self.initializer_range = config.initializer_range

        self.w1 = nn.Linear(
            input_dim,
            self.hidden_size // 2,
            bias=False,
            device=device,
        )
        self.w2 = nn.Linear(
            self.hidden_size // 2,
            config.d_model,
            bias=False,
            device=device,
            )
        self.w3 = nn.Linear(
            input_dim,
            self.hidden_size // 2,
            bias=False,
            device=device,
        )
        # Activation function.
        self.act = Activation.build(config.activation_type, split_inputs=True)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.normal_(self.w1.weight, std=self.initializer_range)
        nn.init.normal_(self.w2.weight, std=self.initializer_range)
        nn.init.normal_(self.w3.weight, std=self.initializer_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w2(self.act(self.w1(x), self.w3(x)))
        x = self.dropout(x)
        return x


class Residual(nn.Module):
    def __init__(self, submodule: nn.Module):
        super().__init__()
        self.submodule = submodule

    def reset_parameters(self):
        self.submodule.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.submodule(x)


class MolmoVisionBackbone(nn.Module):
    def __init__(self, config: MolmoVisionBackboneConfig, llm_config, device=None):
        super().__init__()
        self.config = config
        self.image_preprocessor = config.build_preprocessor()

        if config.use_deepstack:
            assert len(config.vit_layers) > 1, "Deepstack type requires at least 2 layers"

        if not config.use_deepstack or config.share_connector:
            self.image_pooling_2d, self.image_projector = self.build_connector(llm_config, device)
        else:
            image_pooling_2d_list, image_project_list = map(
                list,
                zip(
                    *[self.build_connector(llm_config, device) for _ in range(len(config.vit_layers))]
                )
            )
            self.image_pooling_2d = nn.ModuleList(image_pooling_2d_list)
            self.image_projector = nn.ModuleList(image_project_list)

        self.image_feature_dropout = nn.Dropout(config.image_feature_dropout)

        self.vit_layers = []
        for layer in config.vit_layers:
            if layer >= 0:
                self.vit_layers.append(layer)
            else:
                self.vit_layers.append(config.vit.image_num_layers + layer)
        last_layer_needed = (max(self.vit_layers)+1)

        vit_cfg = self.config.vit
        if last_layer_needed < config.vit.image_num_layers:
            if self.config.skip_unused_layers:
                vit_cfg = replace(vit_cfg, image_num_layers=last_layer_needed)
                self.image_vit: VisionTransformer = vit_cfg.build(device)
            else:
                # We might need to keep the layers for checkpoint compatibility, but we
                # freeze them since unfrozen layers with no gradient confuses torch's distributed
                # optimizer checkpointer
                self.image_vit: VisionTransformer = vit_cfg.build(device)
                for block in self.image_vit.transformer.resblocks[last_layer_needed-1:]:
                    freeze_module(block)
        else:
            self.image_vit: VisionTransformer = vit_cfg.build(device)

        self.num_prefix_tokens = self.image_vit.num_prefix_tokens
        assert self.num_prefix_tokens in {0, 1}, "Only 0 or 1 prefix tokens are supported"

        if config.use_deepstack:
            image_dim = vit_cfg.image_emb_dim
        else:
            image_dim = vit_cfg.image_emb_dim*len(self.config.vit_layers)
        self.pad_embed = None
        if config.image_padding_embed:
            if config.image_padding_embed in ["pad_embed", "regress"]:
                self.pad_embed = nn.Parameter(
                    torch.zeros((image_dim,), device=device))
            elif config.image_padding_embed == "pad_and_partial_pad":
                self.pad_embed = nn.Parameter(
                    torch.zeros((2, image_dim), device=device))
            else:
                raise ValueError(config.image_padding_embed)
        if config.pool_size_embeds:
            assert self.pad_embed is None
            self.pad_embed = nn.Parameter(
                torch.zeros((len(config.pool_size_embeds), image_dim), device=device))
            self._pad_embed_idx = {k*k: i for i, k in enumerate(config.pool_size_embeds)}

        self._cp_load_balancer: Optional[CPLoadBalancer] = None

    def build_connector(self, llm_config, device):
        config = self.config
        input_dim: int = None
        vit_cfg = config.vit
        if config.use_deepstack:
            pool_dim = vit_cfg.image_emb_dim
        else:
            pool_dim = vit_cfg.image_emb_dim * len(config.vit_layers)

        from olmo.nn.image_vit import ViTMultiHeadDotProductAttention

        if config.image_pooling_2d in {ImagePooling2DType.attention, ImagePooling2DType.attention_meanq}:
            image_pooling_2d = ViTMultiHeadDotProductAttention(config.vit, input_dim=pool_dim)
            input_dim = vit_cfg.image_emb_dim
        elif config.image_pooling_2d in [ImagePooling2DType.attention_2wide, ImagePooling2DType.attention_meanq_2x, ImagePooling2DType.attention_meanq_4x]:
            mha_cfg = deepcopy(config.vit)
            factor = 4 if config.image_pooling_2d ==ImagePooling2DType.attention_meanq_4x else 2
            mha_cfg.image_emb_dim *= factor
            mha_cfg.image_head_dim *= factor
            image_pooling_2d = ViTMultiHeadDotProductAttention(mha_cfg, input_dim=pool_dim)
            input_dim = mha_cfg.image_emb_dim
        elif config.image_pooling_2d in [ImagePooling2DType.none, ImagePooling2DType.stack, ImagePooling2DType.mean]:
            image_pooling_2d = None
            nlayers = 1 if config.vit_layers is None else len(config.vit_layers)
            input_dim = nlayers * vit_cfg.image_emb_dim
            if config.image_pooling_2d == ImagePooling2DType.stack:
                input_dim *= 4
        else:
            raise NotImplementedError(f"Unknown image pooling 2D method: {config.image_pooling_2d}")

        if config.image_projector == ImageProjectType.mlp:
            image_projector = ImageProjectorMLP(llm_config, input_dim, device=device)
        elif config.image_projector in [ImageProjectType.linear, ImageProjectType.random_linear]:
            image_projector = nn.Linear(input_dim, llm_config.d_model, bias=False, device=device)
            if config.image_projector == ImageProjectType.random_linear:
                image_projector.weight.requires_grad = False
        else:
            raise NotImplementedError(f"Unknown image projector: {config.image_projector}")
        
        return image_pooling_2d, image_projector

    @classmethod
    def build(cls, config: MolmoVisionBackboneConfig, outut_dim, device=None) -> 'MolmoVisionBackbone':
        return MolmoVisionBackbone(config, outut_dim, device)

    def reset_connector_parameters(self):
        if self.image_pooling_2d is not None:
            if not self.config.use_deepstack or self.config.share_connector:
                self.image_pooling_2d.reset_parameters()
            else:
                for module in self.image_pooling_2d:
                    module.reset_parameters()
        if not self.config.use_deepstack or self.config.share_connector:
            if self.config.image_projector == "2mlp":
                for module in self.image_projector:
                    module.reset_parameters()
            elif self.config.image_projector == "linear":
                nn.init.xavier_uniform_(self.image_projector.weight)
            elif self.config.image_projector in [ImageProjectType.random_linear]:
                nn.init.uniform_(self.image_projector.weight, -0.02, 0.02)
            else:
                self.image_projector.reset_parameters()
        else:
            for module in self.image_projector:
                if self.config.image_projector == "2mlp":
                    for submodule in module:
                        submodule.reset_parameters()
                elif self.config.image_projector == "linear":
                    nn.init.xavier_uniform_(module.weight)
                elif self.config.image_projector in [ImageProjectType.random_linear]:
                    nn.init.uniform_(module.weight, -0.02, 0.02)
                else:
                    module.reset_parameters()
        if self.pad_embed is not None:
            nn.init.zeros_(self.pad_embed)

    def reset_parameters(self):
        self.reset_connector_parameters()
        self.image_vit.reset_parameters()

    def reset_with_pretrained_weights(self):
        self.reset_connector_parameters()  # resets the connector
        self.image_vit.reset_with_pretrained_weights()

    def apply_fsdp2(self, **kwargs):
        self.image_vit.apply_fsdp2(**kwargs)
        if self.image_pooling_2d is not None:
            if not self.config.use_deepstack or self.config.share_connector:
                fully_shard(self.image_pooling_2d, **kwargs)
            else:
                for module in self.image_pooling_2d:
                    fully_shard(module, **kwargs)
        if not self.config.use_deepstack or self.config.share_connector:
            fully_shard(self.image_projector, **kwargs)
        else:
            for module in self.image_projector:
                fully_shard(module, **kwargs)
        # For any remaining parameters in `self`, like the pad embed
        fully_shard(self, **kwargs)

    def apply_activation_checkpointing(self):
        self.image_vit.apply_activation_checkpointing()
        if self.config.connector_activation_checkpointing:
            if not self.config.use_deepstack or self.config.share_connector:
                self.image_projector = checkpoint_wrapper(self.image_projector)
            else:
                self.image_projector = nn.ModuleList([checkpoint_wrapper(module) for module in self.image_projector])
            if self.image_pooling_2d is not None:
                if not self.config.use_deepstack or self.config.share_connector:
                    self.image_pooling_2d = checkpoint_wrapper(self.image_pooling_2d)
                else:
                    self.image_pooling_2d = nn.ModuleList([checkpoint_wrapper(module) for module in self.image_pooling_2d])

    def apply_compile(self, **kwargs):
        if self.config.compile_connector:
            if self.config.compile_connector == "dynamic":
                connect_kwargs = dict(kwargs, dynamic=True)
            elif self.config.compile_connector == "default":
                connect_kwargs = kwargs
            else:
                raise NotImplementedError(self.config.compile_connector)
            if self.image_pooling_2d is not None:
                if not self.config.use_deepstack or self.config.share_connector:
                    self.image_pooling_2d.compile(**connect_kwargs)
                else:
                    for module in self.image_pooling_2d:
                        module.compile(**connect_kwargs)
            if not self.config.use_deepstack or self.config.share_connector:
                self.image_projector.compile(**connect_kwargs)
            else:
                for module in self.image_projector:
                    module.compile(**connect_kwargs)
        if self.config.compile_vit == "blocks":
            for block in self.image_vit.transformer.resblocks:
                block.compile(**kwargs)
        elif self.config.compile_vit is not None:
            raise NotImplementedError(self.config.compile_vit)

    def get_connector_parameters(self):
        vit_params = set(self.image_vit.parameters())
        return (p for p in self.parameters() if p not in vit_params)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        : param images: (batch_size, num_crops, num_patch, n_pixels)
        """
        cfg = self.config
        B, T, N, D = images.shape
        images = images.view(B * T, N, D)
        if self.config.normalize_on_gpu:
            images = self.image_preprocessor.normalize_image_tensor(images)
        image_features = self.image_vit(images)

        if cfg.use_deepstack:
            image_features = [
                image_features[layer][:, self.num_prefix_tokens:].view(B, T, N, -1) for layer in self.vit_layers
            ]
        else:
            features = []
            for layer in self.vit_layers:
                features.append(image_features[layer])
            image_features = torch.cat(features, dim=-1)

            if self.num_prefix_tokens > 0:
                image_features = image_features[:, 1:]
            image_features = image_features.view(B, T, N, -1)
        return image_features
    
    def apply_connector(
        self,
        pooling_fn: nn.Module,
        projector_fn: nn.Module,
        image_features: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        batch_size = image_features.shape[0]
        dim = image_features.shape[-1]
        if self.config.pool_size_embeds:
            embed = self.pad_embed[self._pad_embed_idx[pooled_patches_idx.shape[-1]]]
            image_features = image_features + embed.view(1, 1, 1, -1)

        valid = pooled_patches_idx >= 0

        # Use `pooled_patches_idx` to arange the features for image pooling
        batch_idx = torch.arange(pooled_patches_idx.shape[0], dtype=torch.long, device=pooled_patches_idx.device)
        batch_idx = torch.tile(batch_idx.view(batch_size, 1, 1), [1, pooled_patches_idx.shape[1], pooled_patches_idx.shape[2]])

        # Now [batch, num_high_res_features, pool_dim, dim]
        to_pool = image_features.reshape(batch_size, -1, dim)[batch_idx, torch.clip(pooled_patches_idx, 0)]
        to_pool = to_pool * valid.float()[:, :, :, None]
        to_pool = to_pool.reshape([-1, pooled_patches_idx.shape[-1], dim])
        if self.config.pooling_attention_mask:
            attn_mask = valid.reshape([-1, 1, 1, valid.shape[-1]])
        else:
            attn_mask = None

        if cfg.image_pooling_2d in [ImagePooling2DType.attention_meanq, ImagePooling2DType.attention_meanq_2x, ImagePooling2DType.attention_meanq_4x]:
            if self.config.pooling_attention_mask:
                denom = valid.view(-1, to_pool.shape[-2]).float().sum(-1)
                denom = torch.where(denom == 0, 1, denom)
                query = to_pool.sum(-2, keepdim=True) / denom[:, None, None]
            else:
                query = to_pool.mean(-2, keepdim=True)
            pooled_features = pooling_fn(query, to_pool, attn_mask=attn_mask)
        elif cfg.image_pooling_2d == ImagePooling2DType.mean:
            denom = valid.reshape(-1, to_pool.shape[1]).float().sum(-1, keepdim=True)
            pooled_features = to_pool.sum(-2) / torch.clamp(denom, min=1)
        elif cfg.image_pooling_2d not in {ImagePooling2DType.none, ImagePooling2DType.stack}:
            pooled_features = pooling_fn(to_pool[:, :1, :], to_pool, attn_mask=attn_mask)
        else:
            pooled_features = to_pool

        pooled_features = pooled_features.reshape([batch_size, -1, pooled_features.shape[-1]])

        # MLP layer to map the feature.
        if cfg.image_projector == ImageProjectType.mlpx2:
            for module in projector_fn:
                pooled_features = module(pooled_features)
        else:
            pooled_features = projector_fn(pooled_features)
        
        return pooled_features
    
    def add_image_padding_embed(self, image_features: torch.Tensor, image_masks: torch.Tensor):
        cfg = self.config

        if cfg.image_padding_embed == "pad_embed":
            all_pad = (image_masks == 0).to(dtype=torch.float32)
            pad_embed = self.pad_embed[None, None, None, :]
            image_features = image_features + pad_embed * torch.unsqueeze(all_pad, -1)
        elif cfg.image_padding_embed == "regress":
            pad_embed = self.pad_embed[None, None, None, :]
            image_features = image_features + pad_embed * torch.unsqueeze(torch.maximum(image_masks, torch.zeros_like(image_masks)), -1)
        elif cfg.image_padding_embed == "pad_and_partial_pad":
            pad_embed = self.pad_embed[:, None, None, None, :]
            all_pad = image_masks == 0
            partial_pad = torch.logical_and(image_masks < 1, torch.logical_not(all_pad)).to(dtype=torch.float32)
            all_pad = all_pad.to(dtype=torch.float32)
            image_features = image_features + pad_embed[0] * torch.unsqueeze(all_pad, -1)
            image_features = image_features + pad_embed[1] * torch.unsqueeze(partial_pad, -1)
        else:
            raise ValueError(cfg.image_padding_embed)
        
        return image_features

    def forward(self, 
                images: torch.Tensor, 
                image_masks: torch.Tensor,
                pooled_patches_idx: torch.Tensor,
                enable_cp: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cfg = self.config

        # image_features: (batch_size, num_crops(=num_image), num_patch, nximage_emb_dim)
        batch_size, num_image = images.shape[:2]
        if (cp_load_balancer := self._cp_load_balancer) and enable_cp:
            inputs = [images]
            seq_dims = [1]
            pad_values: List[Union[int, float]] = [0.0]

            pooled_patches_idx = pooled_patches_idx.reshape(batch_size, num_image, -1, pooled_patches_idx.shape[-1])
            inputs.append(pooled_patches_idx)
            seq_dims.append(1)
            pad_values.append(-1)

            images, pooled_patches_idx = cp_load_balancer.batch_shard(
                inputs=inputs,
                seq_dims=seq_dims,
                pad_values=pad_values,
            )

            offset = pooled_patches_idx[0].numel() * cp_load_balancer.cp_rank
            pooled_patches_idx = torch.where(pooled_patches_idx > 0, pooled_patches_idx - offset, pooled_patches_idx)
            pooled_patches_idx = pooled_patches_idx.reshape(batch_size, -1, pooled_patches_idx.shape[-1])

        image_features = self.encode_image(images)

        if cfg.image_padding_embed:
            assert image_masks is not None
            if isinstance(image_features, (list, tuple)):
                image_features = [
                    self.add_image_padding_embed(image_feature, image_masks)
                    for image_feature in image_features
                ]
            else:
                image_features = self.add_image_padding_embed(image_features, image_masks)

        if isinstance(image_features, (list, tuple)):
            image_features = [self.image_feature_dropout(image_feature) for image_feature in image_features]
            if cfg.share_connector:
                pooling_fns = [self.image_pooling_2d] * len(self.vit_layers)
                projector_fns = [self.image_projector] * len(self.vit_layers)
            else:
                pooling_fns = self.image_pooling_2d
                projector_fns = self.image_projector
        else:
            image_features = [self.image_feature_dropout(image_features)]
            pooling_fns = [self.image_pooling_2d]
            projector_fns = [self.image_projector]

        multiple_pooling = isinstance(pooled_patches_idx, (tuple, list))
        if not multiple_pooling:
            pooled_patches_idxs = [pooled_patches_idx]
        else:
            pooled_patches_idxs = pooled_patches_idx

        all_pooled_features = []
        for pooled_patches_idx in pooled_patches_idxs:
            valid_token = torch.any(pooled_patches_idx >= 0, -1)
            pooled_features_list = []
            for image_features_i, pooling_fn, projector_fn in zip(image_features, pooling_fns, projector_fns):
                pooled_features_list.append(
                    self.apply_connector(
                        pooling_fn,
                        projector_fn,
                        image_features_i,
                        pooled_patches_idx,
                    )
                )
            if len(pooled_features_list) > 1:
                all_pooled_features.append((pooled_features_list, valid_token))
            else:
                all_pooled_features.append((pooled_features_list[0], valid_token))

        if multiple_pooling:
            return all_pooled_features
        else:
            image_features_list, valid_token = all_pooled_features[0]
            if isinstance(image_features_list, list):
                return [
                    image_features.view(-1, image_features.shape[-1])[valid_token.flatten()]
                    for image_features in image_features_list
                ]
            else:
                # all gather image features and valid tokens if using context parallelism
                if cp_load_balancer and enable_cp:
                    # Get the process group for all-gather
                    pg = self.pg if hasattr(self, 'pg') else None 
                    # Use differentiable all-gather to preserve gradient flow 
                    # All-gather image features across the context parallel group (differentiable)
                    gathered_features = differentiable_all_gather(image_features_list, group=pg)
                    image_features_list = torch.cat(gathered_features, dim=1)

                    # All-gather valid tokens (this doesn't need gradients, but we use the same function for consistency)
                    gathered_valids = differentiable_all_gather(valid_token, group=pg)
                    valid_token = torch.cat(gathered_valids, dim=1)
                return image_features_list.view(-1, image_features_list.shape[-1])[valid_token.flatten()]

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        load_balancer: CPLoadBalancerType,
        head_stride: int = 1,
        attention_type: Literal["ulysses", "ring"] = "ulysses",
    ):
        """
        Prepare the model for context-parallelism (CP).

        : param cp_mesh: The device mesh to use for context parallelism
        : param load_balancer: The load balancer type to use for CP
        : param head_stride: The head stride to use for ring attention load balancing
        : param attention_type: The CP attention mechanism to use ("ulysses" or "ring")
        """
        self._cp_load_balancer = load_balancer.build(cp_mesh)
        self.pg = cp_mesh.get_group()
