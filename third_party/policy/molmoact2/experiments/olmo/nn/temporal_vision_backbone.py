from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Union

import torch

from olmo.nn.temporal_image_vit import TemporalVitConfig
from olmo.nn.vision_backbone import MolmoVisionBackboneConfig, MolmoVisionBackbone, ImagePooling2DType, ImageProjectType
from torch.nn import functional as F
from torch.distributed.nn.functional import all_gather as differentiable_all_gather


@dataclass
class TemporalVisionBackboneConfig(MolmoVisionBackboneConfig):
    vit: TemporalVitConfig = field(default_factory=TemporalVitConfig)
    """The vision ViT"""

    def build(self, llm_config, device):
        return TemporalVisionBackbone(self, llm_config, device)


class TemporalVisionBackbone(MolmoVisionBackbone):
    @classmethod
    def build(cls, config: TemporalVisionBackboneConfig, output_dim, device=None) -> 'TemporalVisionBackbone':
        return TemporalVisionBackbone(config, output_dim, device)

    def encode_image(self,
                     images: torch.Tensor,
                     pooled_patches_idx: torch.Tensor,
                     num_images: torch.Tensor,
                     multimodal_type: torch.Tensor,
        ) -> torch.Tensor:
        """
        : param images: (batch_size, num_crops, num_patch, n_pixels)
        """
        cfg = self.config
        B, T, N, D = images.shape
        images = images.view(B * T, N, D)
        if self.config.normalize_on_gpu:
            images = self.image_preprocessor.normalize_image_tensor(images)
        vit_out = self.image_vit(images, None, pooled_patches_idx, num_images, multimodal_type)
        image_features = vit_out[0]

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
        return (image_features,) + vit_out[1:]

    def forward(self,
                images: torch.Tensor,
                image_masks: torch.Tensor,
                pooled_patches_idx: torch.Tensor,
                num_images: torch.Tensor,
                multimodal_type: torch.Tensor,
                enable_cp: bool = False
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cfg = self.config
        
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

        multiple_pooling = isinstance(pooled_patches_idx, (tuple, list))
        if not multiple_pooling:
            pooled_patches_idxs = [pooled_patches_idx]
        else:
            pooled_patches_idxs = pooled_patches_idx
        

        # image_features: (batch_size, num_crops(=num_image), num_patch, nximage_emb_dim)
        batch_size, num_image = images.shape[:2]
        pooled_patches_idx = pooled_patches_idxs[0]
        image_features, valid_mask, scores, sims, refill_over_kept, new_batch_size, prunable_frame_mask, prunable_pooled_patches_idx = self.encode_image(images, pooled_patches_idx, num_images, multimodal_type)

        aux_loss = torch.tensor([0], device=images.device, dtype=images.dtype)
        prunable_valid_mask = None
        
        # Use `pooled_patches_idx` to arange the features for image pooling
        batch_idx = torch.arange(pooled_patches_idx.shape[0], dtype=torch.long, device=pooled_patches_idx.device)
        batch_idx = torch.tile(batch_idx.view(batch_size, 1, 1), [1, pooled_patches_idx.shape[1], pooled_patches_idx.shape[2]])
        if valid_mask is not None and prunable_frame_mask.sum() > 0:
            prunable_valid_mask = valid_mask[prunable_frame_mask]
            valid_mask = valid_mask.reshape(batch_size, -1)[batch_idx, torch.clip(pooled_patches_idx, 0)]
            prunable_valid_mask = prunable_valid_mask.reshape(-1)[torch.clip(prunable_pooled_patches_idx, 0)].any(dim=-1)

        if scores is not None and self.training:

            sims = sims.reshape(num_image, scores.shape[1], num_image, scores.shape[1])
            
            # Vectorized extraction of neighboring frame pairs at the same patch location
            frame_idx = torch.arange(num_image - 1, device=sims.device)  # [0, 1, ..., num_image-2]
            patch_idx = torch.arange(scores.shape[1], device=sims.device)  # [0, 1, ..., 80]

            # Create meshgrid for all pairs
            frame_idx_grid, patch_idx_grid = torch.meshgrid(frame_idx, patch_idx, indexing='ij')
            # shape: [num_image-1, 81]

            # Gather the similarities
            neighbor_sims = sims[frame_idx_grid, patch_idx_grid, frame_idx_grid + 1, patch_idx_grid] # shape: [(num_image-1), 81]
            # neighbor_sims = neighbor_sims.view(-1)  # shape: [(num_image-1) * 81]
            scores_for_neighbors = scores[-neighbor_sims.shape[0]:]  # align shapes

            sim_reg_loss = F.mse_loss(scores_for_neighbors, 1 - neighbor_sims, reduction='none')
            sim_reg_loss = (sim_reg_loss * prunable_frame_mask[1:, None].float())
            sim_reg_loss = sim_reg_loss.sum() / (prunable_frame_mask[1:, None] * scores.shape[1]).sum().clip(1)
            aux_loss = sim_reg_loss * self.config.vit.sim_reg_loss_coefficient

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

        all_pooled_features = []
        for pooled_patches_idx in pooled_patches_idxs:
            valid_token = torch.any(pooled_patches_idx >= 0, -1)
            if valid_mask is not None and prunable_frame_mask.sum() > 0:
                valid_token = valid_mask.any(dim=-1) & valid_token
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
            image_to_return = all_pooled_features
        else:
            image_features_list, valid_token = all_pooled_features[0]
            if isinstance(image_features_list, list):
                image_to_return = [
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
                image_to_return = image_features_list.view(-1, image_features_list.shape[-1])[valid_token.flatten()]
        return image_to_return, aux_loss, prunable_valid_mask, scores, refill_over_kept, new_batch_size
