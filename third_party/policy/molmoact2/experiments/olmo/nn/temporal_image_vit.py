import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from olmo.nn.image_vit import VitConfig, VisionBackboneType, VisionTransformer, DinoVisionTransformer, vit_activation_checkpoint_function, ViTMultiHeadDotProductAttention
from olmo.nn.image_vit import SiglipVisionTransformer as BaseSiglipViT
from olmo.nn.image_vit import ResidualAttentionBlock as BaseResidualAttentionBlock

from olmo.torch_util import get_global_rank

from torch.distributed.fsdp import fully_shard

from olmo.preprocessing.multimodal_preprocessor import MultimodalTypes

log = logging.getLogger(__name__)


@dataclass
class TemporalVitConfig(VitConfig):
    topk: float = 0.5
    """The topk most similar tokens to be removed or merged."""
    
    new_ref_threshold: float = 0.85
    """Used ONLY if `prune_method` is NOT `scorer`. If the average per-patch cosine similarity between the current frame and the reference patches are above this threshold, prune from the current frame. Otherwise, the entire current frame is retained and used as reference."""

    merge: bool = False
    """[Unimplemented] If True, tokens will be merged according to their pooled patches, i.e. if doing 3x3 pooling, pruned tokens get compressed 1:9."""
    
    prune_at: int = 3
    """Which layer to start pruning at."""
    
    prune: bool = True
    
    prune_method: str = "scorer"
    """Choose between `random`, `heuristic`, `easy`, and `scorer`."""
    
    sim_reg_loss_coefficient: float = 0.5
    
    rescale_bias_min: float = math.exp(-2)
    
    prune_from_frame: int = 0
    """We only prune if a video contains more than `prune_from_frame` frames."""
    
    pruning_in_scorer: bool = True

    def build(self, device):
        if self.image_model_type == VisionBackboneType.openai:
            return VisionTransformer(self)
        elif self.image_model_type == VisionBackboneType.siglip:
            return SiglipVisionTransformer(self)
        elif self.image_model_type == VisionBackboneType.dino:
            return DinoVisionTransformer(self)
        else:
            raise NotImplementedError(f"Unknown image model type: {self.image_model_type}")

    
class ScorerMLP(nn.Module):

    def __init__(self, input_dim: int, config: TemporalVitConfig, device=None):
        super().__init__()
        self.input_dim = input_dim
        self.config = config

        self.w1 = nn.Linear(
            input_dim,
            config.image_mlp_dim,
            bias=True,
            device=device,
        )
        self.w2 = nn.Linear(
            config.image_mlp_dim,
            config.image_mlp_dim,
            bias=True,
            device=device,
        )
        self.w3 = nn.Linear(
            config.image_mlp_dim,
            1,
            bias=True,
            device=device,
        )
        self.act = nn.SiLU()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.w1.weight, std=math.sqrt(1 / self.input_dim), a=-2.0, b=2.0)
        nn.init.trunc_normal_(self.w2.weight, std=math.sqrt(1 / self.config.image_mlp_dim), a=-2.0, b=2.0)
        nn.init.trunc_normal_(self.w3.weight, std=math.sqrt(1 / self.config.image_mlp_dim), a=-2.0, b=2.0)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)
        nn.init.zeros_(self.w3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        x = self.act(x)
        x = self.w3(x)
        return x.squeeze(-1)
    
 
class TemporalTokenScorer(nn.Module):
    def __init__(self, config: TemporalVitConfig, device=None):
        super().__init__()
        self.config = config
        
        self.patch_token_pooler = ViTMultiHeadDotProductAttention(config, device=device)
        self.mlp = ScorerMLP(config.image_emb_dim*2, config, device=device)
        self.sigmoid = nn.Sigmoid()

    def reset_parameters(self):
        self.patch_token_pooler.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x: torch.Tensor, pooled_patches_idx: torch.Tensor) -> torch.Tensor:
        BT, _, D = x.shape
        valid = pooled_patches_idx >= 0
        to_pool = x.reshape(-1, D)[torch.clip(pooled_patches_idx, 0)]
        to_pool = to_pool * valid.float()[:, :, None] #[:, :, :, None]
        to_pool = to_pool.reshape([-1, pooled_patches_idx.shape[-1], D])
        query = to_pool.mean(-2, keepdim=True)
        x = self.patch_token_pooler(query, to_pool)
        x = x.reshape(BT, -1, D)
        
        BT, N, D = x.shape
        
        # then score by concating previous and current frame features
        pad_frame = torch.zeros((1, N, D), device=x.device, dtype=x.dtype)  # padding frame
        x = torch.cat([pad_frame, x], dim=0)
        prev = x[:-1]
        curr = x[1:]
        x = torch.cat([prev, curr], dim=-1)
        x = self.mlp(x)
        return self.sigmoid(x)


class BlockCollection(nn.Module):

    def __init__(self, config: TemporalVitConfig, device=None):
        super().__init__()
        self.config = config
        self._activation_checkpoint_fn: Optional[Callable] = None
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(config, device) for _ in range(config.image_num_layers)
        ])
        if self.config.prune and self.config.prune_method == "scorer":
            self.temporal_token_scorer = TemporalTokenScorer(config, device=device)
        self.prune_methods = dict(
            heuristic=self.forward_heuristic,
            easy=self.forward_heuristic_easy,
            scorer=self.forward_with_scorer,
            random=self.forward_random,
        )
        
    def reset_parameters(self):
        for r in self.resblocks:
            r.reset_parameters()
        if self.config.prune and self.config.prune_method == "scorer":
            self.temporal_token_scorer.reset_parameters()

    def forward(self, x: torch.Tensor, pooled_patches_idx: torch.Tensor = None, num_images=None, multimodal_type=None) -> List[torch.Tensor]:
        if self.config.prune:
            return self.prune_methods[self.config.prune_method](x, pooled_patches_idx, num_images, multimodal_type)
        else:
            return self.forward_no_pruning(x)


    def forward_no_pruning(self, x):
        hidden_states = []
        for r in self.resblocks:
            if self._activation_checkpoint_fn:
                x = self._activation_checkpoint_fn(r, x)
            else:
                x = r(x)
            hidden_states.append(x)
        return hidden_states, None, None, None, torch.tensor([0], device=x.device), x.shape[0], None, None


    def forward_random(self, x: torch.Tensor, pooled_patches_idx: torch.Tensor, num_images=None, multimodal_type=None) -> List[torch.Tensor]:
        return self.forward_heuristic(x, pooled_patches_idx, num_images, multimodal_type, prune_method="random")

    def forward_heuristic_easy(self, x: torch.Tensor, pooled_patches_idx: torch.Tensor, num_images=None, multimodal_type=None) -> List[torch.Tensor]:
        return self.forward_heuristic(x, pooled_patches_idx, num_images, multimodal_type, prune_method="easy")

    def forward_heuristic(self, x_orig: torch.Tensor, pooled_patches_idx: torch.Tensor, num_images=None, multimodal_type=None, prune_method: str = None) -> List[torch.Tensor]:
        """
        Remove all pruned tokens found with the specified heuristic and rearrange them to reduce total batch size.
        
        Default heuristic iterates from first to last frame and prune based on found reference frames. Better as it looks at temporal evolution across multiple frames.
        
        `Easy` heuristic only compares each frame to its previous frame as reference.
        
        `Random` heuristic randomly prunes tokens.

        Args:
            x_orig (torch.Tensor): The original input tensor of shape (BT, N, D).
            pooled_patches_idx (torch.Tensor): The indices of the pooled patches of shape (B, NP, G).

        Returns:
            Tensors:
            - **hidden_states**: List of hidden states at each layer.
            - **valid_mask**: The final valid mask after pruning.
            - DO NOT USE THE THIRD RETURN VALUE. FOR API CONSISTENCY ONLY.
            - **sims**: The cosine similarities between every frame and every other frame.
            - **first_padding_frame_idx**: Indices of the first padding frame for each batch.
            - **refill_mask**: The mask indicating which tokens were filled back in after pruning.
            - **new_batch_size**: The new batch size after pruning.
        """
        hidden_states = []
        BT, N, D = x_orig.shape
        B, NP, G = pooled_patches_idx.shape
        T = BT // B

        # Mask to track valid tokens (not pruned)
        valid_mask = torch.ones(B, T * N, dtype=torch.bool, device=x_orig.device)
        thres = self.config.new_ref_threshold

        first_padding_frame_idx = find_first_padding_frame(pooled_patches_idx, T)
        idx = first_padding_frame_idx * N  # shape: (B,)
        arange = torch.arange(valid_mask.shape[1], device=valid_mask.device)  # shape: (T*N,)
        mask = arange.unsqueeze(0) >= idx.unsqueeze(1)  # shape: (B, T*N)
        valid_mask[mask] = False

        num_tokens_to_prune = 9*9 * self.config.topk * (first_padding_frame_idx - self.config.prune_from_frame)
        num_tokens_to_prune = num_tokens_to_prune.int().clip(0)
        # worst_case_num_frames = [min(BT, math.ceil(BT * 2 * (1 - prune_k_per_time * (i + 1)))) for i in range(len(self.config.prune_layers))]
        worst_case_num_frames = None

        pooled_cosine_sims = compute_cosine_sims(x_orig, pooled_patches_idx) if prune_method != "random" else None
        max_retries = 5
        for tries in range(max_retries):
            if prune_method == "easy":
                valid_mask, flag = easy_prune(valid_mask, pooled_patches_idx, pooled_cosine_sims, first_padding_frame_idx, thres, num_tokens_to_prune, T)
            elif prune_method == "random":
                valid_mask, flag = random_prune(valid_mask, pooled_patches_idx, first_padding_frame_idx, num_tokens_to_prune, T)
            else:
                valid_mask, flag = new_prune(valid_mask, pooled_patches_idx, pooled_cosine_sims, first_padding_frame_idx, thres, num_tokens_to_prune, T)

            thres -= 0.05
            if flag or thres < 0: break

        x = x_orig
        attn_mask = None

        for i, r in enumerate(self.resblocks):
            if self._activation_checkpoint_fn:
                x = self._activation_checkpoint_fn(r, x, attn_mask=attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)

            if self.config.merge or i > self.config.prune_at:
                x_new = torch.zeros_like(x_orig).to(x.dtype)
                valid_mask_reshaped = valid_mask.reshape(B * T, N)  # (B*T, N)
                row_idx, col_idx = torch.where(valid_mask_reshaped)
                rows, cols = new_locs[row_idx, col_idx].unbind(-1)  # [15498,  2]
                x_new[row_idx, col_idx] = x[rows, cols]
                x_orig = x_new
                hidden_states.append(x_new)

            else:
                x_orig = x
                hidden_states.append(x)

            if i == self.config.prune_at:
                x, attn_mask, new_locs, refill_mask = rearrange_tokens(x_orig, valid_mask, worst_case_num_frames)
                cur_prune_layer += 1

        return hidden_states, valid_mask, pooled_cosine_sims, pooled_cosine_sims, refill_mask, x.shape[0], None, None


    def forward_with_scorer(self, x: torch.Tensor, pooled_patches_idx: torch.Tensor, num_images=None, multimodal_type=None) -> List[torch.Tensor]:
        """After layer `self.config.prune_at`, use attention biasing to scale down pruned tokens found with `prune_with_scorer`.
        After layer `self.config.prune_at+1`, if `self.config.pruning_in_scorer` is `True`, prune the tokens and rearrange them to reduce total batch size.

        Args:
            x (torch.Tensor): The input tensor of shape (BT, N, D).
            pooled_patches_idx (torch.Tensor): The indices of the pooled patches of shape (B, NP, G).
        
        Returns:
            Tensors:
            - **hidden_states**: List of hidden states at each layer.
            - **valid_mask**: The final valid mask after pruning.
            - **scores**: The scores from the scorer.
            - **sims**: The cosine similarities between pooled patches of every frame and every other frame.
            - **first_padding_frame_idx**: Indices of the first padding frame for each batch.
            - **refill_mask**: The mask indicating which tokens were filled back in after pruning.
            - **new_batch_size**: The new batch size after pruning.
        """
        hidden_states = []
        BT, N, D = x.shape
        B, NP, G = pooled_patches_idx.shape
        T = BT // B

        num_tokens_to_prune = 9*9 * self.config.topk * (num_images - self.config.prune_from_frame)
        num_tokens_to_prune *= multimodal_type == MultimodalTypes.VIDEO
        num_tokens_to_prune = num_tokens_to_prune.int().clip(0)
        num_images = num_images.clip(0)
        first_padding_frame_idx = num_images.sum(dim=-1)
        # force pad in the end
        num_images = torch.cat([num_images, T - num_images.sum(dim=-1, keepdim=True)], dim=-1)
        num_tokens_to_prune = torch.cat([num_tokens_to_prune, torch.zeros((B, 1), dtype=num_tokens_to_prune.dtype, device=num_tokens_to_prune.device)], dim=-1)
        num_images = num_images.view(-1)
        num_tokens_to_prune = num_tokens_to_prune.view(-1)
        prunable_frame_mask = num_tokens_to_prune.repeat_interleave(num_images) > 0
        num_prunable_frames = prunable_frame_mask.sum().item()

        valid_mask = torch.ones(num_prunable_frames, N, dtype=torch.bool, device=x.device)

        attn_mask_to_use, scores_to_return, refill_over_kept, sims = None, None, torch.tensor([0], device=x.device), None

        x_orig = x
        prunable_pooled_patches_idx = None

        for i, r in enumerate(self.resblocks):
            if self._activation_checkpoint_fn:
                x = self._activation_checkpoint_fn(r, x, attn_mask=attn_mask_to_use)
            else:
                x = r(x, attn_mask=attn_mask_to_use)
            
            if i > self.config.prune_at + 1 and self.config.pruning_in_scorer and num_prunable_frames > 0:
                x_new = torch.zeros_like(x_orig).to(x.dtype)
                row_idx, col_idx = torch.where(valid_mask)
                rows, cols = new_locs[row_idx, col_idx].unbind(-1)  # [15498,  2]
                x_new[row_idx, col_idx] = x[rows, cols]
                x_orig = x_new
                hidden_states.append(x_new)
            else:
                x_orig = x
                hidden_states.append(x)

            if i == self.config.prune_at:
                # pool_width = int(math.sqrt(pooled_patches_idx.shape[2]))  # sometimes it is just shape 1 due to padding and such, not gonna work.
                pool_width = 3
                
                # recompute pooled_patches_idx here for scoring
                # resize_idx = np.arange(N).reshape([int(np.sqrt(N)), int(np.sqrt(N))])
                # single_prunable_pooled_patches_idx = arange_for_pooling(resize_idx, pool_width, pool_width).reshape(-1, pool_width * pool_width)
                resize_idx = torch.arange(N, dtype=torch.long, device=x.device).reshape(int(math.sqrt(N)), int(math.sqrt(N)))
                H, W = resize_idx.shape

                # 2. Calculate padding needed to make dimensions divisible by pool_width
                # h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
                # w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
                h_pad = pool_width * ((H + pool_width - 1) // pool_width) - H
                w_pad = pool_width * ((W + pool_width - 1) // pool_width) - W

                # 3. Apply padding
                # idx_arr = np.pad(idx_arr, [[h_pad//2, (h_pad+1)//2], [w_pad//2, (w_pad+1)//2]], ...)
                # Note: F.pad's padding tuple is (left, right, top, bottom)
                pad_left = w_pad // 2
                pad_right = (w_pad + 1) // 2
                pad_top = h_pad // 2
                pad_bottom = (h_pad + 1) // 2

                padded_idx = F.pad(resize_idx, (pad_left, pad_right, pad_top, pad_bottom), 
                                    mode='constant', value=-1)

                # 4. Get padded dimensions and number of blocks
                H_padded, W_padded = padded_idx.shape
                num_blocks_h = H_padded // pool_width
                num_blocks_w = W_padded // pool_width

                # 5. Replicate rearrange and final reshape
                # This replaces:
                # einops.rearrange(idx_arr, "(h dh) (w dw) -> h w (dh dw)", dh=pool_h, dw=pool_w)
                # .reshape(-1, pool_width * pool_width)

                # First, reshape to (h, dh, w, dw)
                reshaped = padded_idx.reshape(
                    num_blocks_h, pool_width, 
                    num_blocks_w, pool_width
                )

                # Permute to (h, w, dh, dw)
                permuted = reshaped.permute(0, 2, 1, 3)

                # Finally, reshape to (h*w, dh*dw)
                single_prunable_pooled_patches_idx = permuted.reshape(-1, pool_width * pool_width)
                # single_prunable_pooled_patches_idx = torch.tensor(single_prunable_pooled_patches_idx, device=prunable_x.device)
                
                # prunable_pooled_patches_idx = []
                # offset = 0
                # for i in range(num_prunable_frames):
                #     cur_pooled_idx = single_prunable_pooled_patches_idx.clone()
                #     pooled_idx_with_offset = torch.where(
                #         cur_pooled_idx >= 0,
                #         cur_pooled_idx + offset,
                #         cur_pooled_idx,
                #     )
                #     prunable_pooled_patches_idx.append(pooled_idx_with_offset)
                #     offset += N
                # prunable_pooled_patches_idx = torch.cat(prunable_pooled_patches_idx)
                
                # The number of pooled patches per frame (e.g., 9, from 3x3)
                P = single_prunable_pooled_patches_idx.shape[1]

                # 1. Unsqueeze the base indices to shape [1, X, P]
                #    (Represents a "batch" of 1)
                base_idx = single_prunable_pooled_patches_idx.unsqueeze(0)

                # 2. Create the offsets: [0, N, 2*N, ...]
                #    Unsqueeze to [num_prunable_frames, 1, 1] for broadcasting
                offsets = (
                    torch.arange(
                        num_prunable_frames,
                        device=base_idx.device,
                        dtype=base_idx.dtype,
                    )
                    * N
                ).view(-1, 1, 1)  # .view(-1, 1, 1) is equivalent to .unsqueeze(1).unsqueeze(2)
                offsets_for_scorer = (
                    torch.arange(
                        BT,
                        device=base_idx.device,
                        dtype=base_idx.dtype,
                    )
                    * N
                ).view(-1, 1, 1)  # only used for scorer due to dummy process needed for fsdp/compile/whatever

                # 3. Add them together. Broadcasting handles the rest:
                #    [1, X, P] + [F, 1, 1] => [F, X, P]
                #    (F = num_prunable_frames)
                offset_idx_matrix = base_idx + offsets
                offset_idx_matrix_for_scorer = base_idx + offsets_for_scorer

                # 4. Apply the conditional 'where' logic.
                #    The condition (base_idx >= 0) is [1, X, P]
                #    base_idx (the "else" value) is [1, X, P]
                #    Both broadcast correctly to the [F, X, P] shape.
                prunable_pooled_patches_idx_3d = torch.where(
                    base_idx >= 0, offset_idx_matrix, base_idx
                )
                pooled_patches_idx_for_scorer_3d = torch.where(
                    base_idx >= 0, offset_idx_matrix_for_scorer, base_idx
                )

                # 5. Reshape to merge the frame (F) and patch (X) dimensions
                #    This is the correct equivalent of your torch.cat()
                #    Shape [F, X, P] => [F * X, P]
                prunable_pooled_patches_idx = prunable_pooled_patches_idx_3d.reshape(-1, P)
                pooled_patches_idx_for_scorer = pooled_patches_idx_for_scorer_3d.reshape(-1, P)

                sims = compute_cosine_sims(x, pooled_patches_idx_for_scorer) if self.training else None
                valid_mask, scores_to_return = self.prune_with_scorer(x, valid_mask, pooled_patches_idx_for_scorer, prunable_pooled_patches_idx, num_tokens_to_prune[num_tokens_to_prune > 0], num_images[num_tokens_to_prune > 0], prunable_frame_mask)
                
                if num_prunable_frames == 0:
                    continue
                
                scores_last_dim = scores_to_return.shape[-1]
                scores = scores_to_return.reshape(-1, int(math.sqrt(scores_last_dim)), int(math.sqrt(scores_last_dim)))
                scores = scores.repeat_interleave(pool_width, dim=1).repeat_interleave(pool_width, dim=2)
                scores = scores[prunable_frame_mask]

                # Rescale to [e^-2, 1] proportionately
                old_min, old_max = scores.min(dim=-1, keepdim=True).values, scores.max(dim=-1, keepdim=True).values
                new_min, new_max = self.config.rescale_bias_min, 1.0
                scores = new_min + (scores - old_min) * (new_max - new_min) / (old_max - old_min + 1e-6)

                scores = scores.reshape(num_prunable_frames, -1)

                prunable_num_images = num_images[num_tokens_to_prune > 0]
                first_frame_indices = torch.cat([torch.tensor([0], device=scores.device), prunable_num_images.cumsum(dim=0)[:-1]])
                first_frame_mask = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
                first_frame_mask[first_frame_indices] = True
                scores = torch.where(first_frame_mask.unsqueeze(-1), 1.0, scores)

                scores_for_attention = torch.ones(x.shape[:2], dtype=scores.dtype, device=scores.device)
                scores_for_attention[prunable_frame_mask] = scores

                attn_mask = torch.log(scores_for_attention).unsqueeze(1).unsqueeze(2)  # score 1, we want 0 bias; score 0, we want -inf bias
                attn_mask_to_use = attn_mask

            elif i == self.config.prune_at + 1 and self.config.pruning_in_scorer and num_prunable_frames > 0:
                valid_mask_for_rearranging = torch.ones(BT, N, dtype=torch.bool, device=x.device)
                valid_mask_for_rearranging = valid_mask_for_rearranging.reshape(B, T, -1)
                arange = torch.arange(T, device=valid_mask_for_rearranging.device)  # shape: (T,)
                mask = arange.unsqueeze(0) >= first_padding_frame_idx.unsqueeze(1)  # shape: (B, T)
                valid_mask_for_rearranging[mask] = False
                valid_mask_for_rearranging = valid_mask_for_rearranging.reshape(-1, N)
                
                valid_mask_for_rearranging[prunable_frame_mask] = valid_mask
                valid_mask = valid_mask_for_rearranging
                x, attn_mask, new_locs, refill_mask = rearrange_tokens(x_orig, valid_mask)
                attn_mask_to_use = attn_mask
                
                refill_over_kept = refill_mask.sum() / valid_mask[prunable_frame_mask].sum()

        return hidden_states, valid_mask, scores_to_return, sims, refill_over_kept, x.shape[0], prunable_frame_mask, prunable_pooled_patches_idx


    def prune_with_scorer(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        pooled_patches_idx_for_scorer: torch.Tensor,
        prunable_pooled_patches_idx: torch.Tensor,
        num_to_prune: torch.Tensor,
        num_images: torch.Tensor,
        prunable_frame_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score tokens based on results from `TemporalTokenScorer`, then sort them by topk to prune them.

        Args:
            x (torch.Tensor): The input tensor of shape (B*T, N, D).
            valid_mask (torch.Tensor): A mask indicating valid tokens.
            pooled_patches_idx (torch.Tensor): Tensor of shape (B, NP, G) indicating the pooled patch indices.
            first_padding_frame_idx (torch.Tensor): Indices of the first padding frame for each batch.
            num_to_prune (torch.Tensor): The number of tokens to prune for each batch.
            T (int): The maximum number of frames possible.

        Returns:
            Tensors:
            - **valid_mask**: Updated valid mask after pruning.
            - **scores**: The scores from the scorer.
        """
        new_scores = self.temporal_token_scorer(x, pooled_patches_idx_for_scorer)
        scores = new_scores.clone()
        if prunable_frame_mask.sum() == 0:
            return valid_mask, new_scores
        scores = scores[prunable_frame_mask]
        
        valid_mask = valid_mask.clone()
        new_valid_mask = reshape_for_pooling(valid_mask, prunable_pooled_patches_idx)  # (B, NP)
        new_valid_mask = new_valid_mask.reshape(num_images.sum(), -1)

        # different from the original code, which sorts descending because we directly measure similarity,
        # we assume the scorer produces importance scores, so we want to prune the least important tokens.
        end_idx = num_images.cumsum(dim=0)
        start_idx = torch.cat((torch.tensor([0], device=end_idx.device), end_idx[:-1]))
        for idx, cur_num_to_prune in enumerate(num_to_prune):
            cur_scores = scores[start_idx[idx]:end_idx[idx]]
            cur_scores[0] = 100  # always keep first frame of video
            _, indices = cur_scores.view(-1).sort()
            cur_valid_mask = new_valid_mask[start_idx[idx]:end_idx[idx]]
            cur_valid_mask.view(-1)[indices[:cur_num_to_prune]] = False
            new_valid_mask[start_idx[idx]:end_idx[idx]] = cur_valid_mask

        # new_valid_mask = new_valid_mask.view(B, -1)
        new_valid_mask = new_valid_mask.view(-1)
        keep_patches = torch.nonzero(new_valid_mask, as_tuple=False).squeeze(-1)  # [num_keep, 1]
        keep_indices = prunable_pooled_patches_idx[keep_patches].flatten()
        mask_full = torch.zeros_like(valid_mask)
        mask_full.view(-1)[keep_indices] = True

        return mask_full, new_scores


def reshape_for_pooling(mask: torch.Tensor, pooled_patches_idx: torch.Tensor) -> torch.Tensor:
    """
    Reshape the mask to match the pooled patches index.
    
    Args:
        mask (torch.Tensor): Input tensor of shape (B*T, N).
        pooled_patches_idx (torch.Tensor): Pooled patches index of shape (B, NP, G).
        
    Returns:
        torch.Tensor: Reshaped tensor of shape (B, NP).
    """
    NP, G = pooled_patches_idx.shape
    mask = mask.clone()
    # pool_batch_idx = torch.arange(B, dtype=torch.long, device=pooled_patches_idx.device)
    # pool_batch_idx = torch.tile(pool_batch_idx.view(B, 1, 1), [1, NP, G])
    # mask = mask.reshape(B, -1)[pool_batch_idx, torch.clip(pooled_patches_idx, 0)]  # (B, NP, G)
    mask = mask.reshape(-1)[torch.clip(pooled_patches_idx, 0)]
    mask = mask.any(dim=-1)  # [B, NP]
    return mask


def rearrange_tokens(
    x: torch.Tensor, 
    valid_mask: torch.Tensor,
    worst_case_num_frames: Optional[int] = None,
) -> torch.Tensor:
    """
    Reassign token locations using first-fit descending greedy algorithm.
    
    The kept tokens will be rearranged into N slots per frame as packed as possible to reduce
    the total batch size/number of frames being processed at a time.
    
    Args:
        x (torch.Tensor):                       Input of shape (B*T, N, D) in its original shape, with tokens at any position being pruned.
        valid_mask (torch.Tensor):              Mask of shape (B, T*N) where True indicates tokens to keep.
        worst_case_num_frames (Optional[int]):  The number of frames to process at once (replaces BT' below).
                                                    This is an upper bound for compilation purposes.
                                                    If None, each input tensor will have dynamic batch sizes due to token pruning,
                                                    and torch.compile will fail.
    Returns:
        Tensors:
        - **rearranged**:       Tensor of shape (BT', N, D) with tokens rearranged according to assignment.
        - **attn_mask**:        Tensor of shape (BT', 1, N, N) indicating attention mask for rearranged tokens. 
                                    Only tokens that were originally from the same frame can attend to each other.
        - **new_locs**:         Tensor of shape (B*T, N, 2) indicating the new locations of tokens in the rearranged tensor.
    """

    BT, N, D = x.shape
    new_N = N
    
    # Reshape valid_mask for easier processing
    valid_mask = valid_mask.clone()
    valid_mask = valid_mask.reshape(BT, N)
    
    # Count valid tokens per frame
    valid_counts = valid_mask.sum(dim=-1)  # Shape: (B*T)
    sorted_counts, sorted_indices = valid_counts.sort(descending=True)
    
    # # Fully vectorized determination of frame assignments using cumulative sums (suboptimal assignment)
    # cum_counts = valid_counts.cumsum(0) # starting offset for each frame
    # frame_assignments = (cum_counts - 1) // new_N          # each original frame's new frame index
    # frame_end_indices = cum_counts - frame_assignments * new_N
    # frame_start_indices = frame_end_indices - valid_counts
    
    # First-fit descending algorithm to determine assignments
    # Mask out zero counts
    nonzero_mask = sorted_counts > 0
    counts = sorted_counts[nonzero_mask]
    indices = sorted_indices[nonzero_mask]

    frame_assignments = torch.full((BT,), -1, dtype=torch.long, device=x.device)
    frame_start_indices = torch.zeros(BT, dtype=torch.long, device=x.device)
    frame_end_indices = torch.zeros(BT, dtype=torch.long, device=x.device)
    target_frame_usage = torch.zeros(BT, dtype=torch.long, device=x.device)

    for count, frame_idx in zip(counts, indices):
        # Find all frames that can fit this count
        fits = (target_frame_usage + count <= new_N)
        # Get the first available frame
        tt = torch.where(fits)[0][0]
        frame_assignments[frame_idx] = tt
        frame_start_indices[frame_idx] = target_frame_usage[tt]
        frame_end_indices[frame_idx] = target_frame_usage[tt] + count
        target_frame_usage[tt] += count

    # Count how many original frames are assigned to each target frame
    unique_assignments, counts = torch.unique(frame_assignments, return_counts=True)
    # Find target frames that have only one original frame assigned
    single_target_frames = unique_assignments[counts == 1]
    
    # Create a mask for original frames that are alone in their target
    is_alone = torch.isin(frame_assignments, single_target_frames)
    target_frame_usage[single_target_frames] = new_N
    frame_end_indices[is_alone] = new_N
    old_valid_mask = valid_mask.clone()
    valid_mask[is_alone] = True
    refill_mask = valid_mask & ~old_valid_mask

    # Find the max target frame index to determine BT'
    BT_prime = target_frame_usage.nonzero().shape[0] if worst_case_num_frames is None else worst_case_num_frames
    # BT_prime = frame_assignments.max() + 1 if worst_case_num_frames is None else worst_case_num_frames
    rearranged = torch.zeros(BT_prime, new_N, D, dtype=x.dtype, device=x.device)
    new_locs = torch.zeros(BT, N, 2, dtype=torch.long, device=x.device)
    
    # Compute valid positions per original frame.
    orig_frames, orig_cols = torch.nonzero(valid_mask, as_tuple=True)  # shape (K, 2); each row: [orig_frame, col]
    local_order = (valid_mask.cumsum(dim=1) - 1)[valid_mask]  # shape (K,)

    tgt_frames = frame_assignments[orig_frames]
    tgt_cols = frame_start_indices[orig_frames] + local_order
    
    rearranged[tgt_frames, tgt_cols] = x[orig_frames, orig_cols]
    new_locs[orig_frames, orig_cols] = torch.stack([tgt_frames, tgt_cols], dim=-1)

    attn_mask = torch.zeros(BT_prime, new_N, new_N, dtype=torch.bool, device=x.device)
    
    # # Unvectorized version
    # for b in range(BT):
    #     attn_mask[frame_assignments[b], frame_start_indices[b]:frame_end_indices[b], frame_start_indices[b]:frame_end_indices[b]] = True
        
    # L is the length of the valid window per original frame b
    L = frame_end_indices - frame_start_indices  # shape: (BT,)
    max_window = L.max().item()
    device = frame_assignments.device

    # Create a grid for each original frame b of size (max_window, max_window)
    I = torch.arange(max_window, device=device).view(1, max_window, 1).expand(BT, max_window, max_window)
    J = torch.arange(max_window, device=device).view(1, 1, max_window).expand(BT, max_window, max_window)

    # Valid positions for each frame b where both indices are within the window
    valid = (I < L.view(BT, 1, 1)) & (J < L.view(BT, 1, 1))

    # Compute the actual token indices by adding the per-frame starting offset.
    rows = frame_start_indices.view(BT, 1, 1) + I
    cols = frame_start_indices.view(BT, 1, 1) + J

    # Map each original b to its new frame: frame_assignments[b] gives the new frame index.
    # Then update attn_mask accordingly.
    attn_mask[
        frame_assignments.unsqueeze(1).unsqueeze(2).expand_as(rows)[valid],
        rows[valid],
        cols[valid]
    ] = True

    return rearranged, attn_mask.unsqueeze(1), new_locs, refill_mask


def compute_cosine_sims(x: torch.Tensor, pooled_patches_idx: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarities between pooled patches.

    Args:
        x (torch.Tensor): Input tensor of shape (B*T, N, D).
        pooled_patches_idx (torch.Tensor): Tensor of shape (B, NP, G) indicating the pooled patch indices.

    Returns:
        Tensor: Cosine similarities between pooled patches of shape (B, NP, NP).
    """
    BT, N, D = x.shape
    NP, G = pooled_patches_idx.shape

    # batch_idx = torch.arange(B, dtype=torch.long, device=pooled_patches_idx.device)
    # batch_idx = torch.tile(batch_idx.view(B, 1, 1), [1, NP, G])

    # Now [batch, num_high_res_features, pool_dim, dim]
    # x_pool = x.reshape(B, -1, D)[batch_idx, torch.clip(pooled_patches_idx, 0)]  # (B, T * N // G, G, D)
    # x_pool = x_pool.reshape(B, -1, G * D)
    x_pool = x.reshape(-1, D)[torch.clip(pooled_patches_idx, 0)]  # (B*NP, G, D)
    x_pool = x_pool.reshape(-1, G*D)
    x_pool = F.normalize(x_pool, dim=-1)              # [B, M, D]
    pooled_cosine_sims = torch.mm(x_pool, x_pool.t())  # [B, M, M]

    return pooled_cosine_sims


def find_first_padding_frame(pooled_patches_idx: torch.Tensor, T: int) -> torch.Tensor:
    """Find the first padding frame for each batch.

    Args:
        pooled_patches_idx (torch.Tensor): Tensor of shape (B, NP, G) indicating the pooled patch indices.
        T (int): The maximum number of frames possible.

    Returns:
        Tensor: Indices of shape (B,) indicating the index of the first padding frame for each batch.
    """
    B, NP, G = pooled_patches_idx.shape
    valid = pooled_patches_idx >= 0
    valid_token = torch.any(valid, -1)
    first_padding_frame_idx = valid_token.float().argmin(dim=1) // (NP // T)
    first_padding_frame_idx[first_padding_frame_idx == 0] = T

    return first_padding_frame_idx


def new_prune(
    valid_mask: torch.Tensor,
    pooled_patches_idx: torch.Tensor,
    pooled_cosine_sims: torch.Tensor,
    first_padding_frame_idx: torch.Tensor,
    new_ref_threshold: float,
    num_to_prune: torch.Tensor,
    T: int,
    prune_times: int = 1,
) -> Tuple[torch.Tensor, bool]:
    """Prune tokens based on their validity and similarity scores.

    Args:
        valid_mask (torch.Tensor): A mask indicating valid tokens.
        pooled_patches_idx (torch.Tensor): Tensor of shape (B, NP, G) indicating the pooled patch indices.
        pooled_cosine_sims (torch.Tensor): Cosine similarities between pooled patches.
        first_padding_frame_idx (torch.Tensor): Indices of the first padding frame for each batch.
        new_ref_threshold (float): The new reference threshold for pruning.
        num_to_prune (torch.Tensor): The number of tokens to prune for each batch.
        T (int): The maximum number of frames possible.
        prune_times (int): The number of times to prune tokens in this call.

    Returns:
        Tensors:
        - **new_valid_mask**: Updated valid mask after pruning.
        - **flag**: False if we had to prune patches labelled to ignore (with score -1). The caller decides whether to retry pruning with a lower threshold.
    """
    B, TN = valid_mask.shape
    B, NP, G = pooled_patches_idx.shape
    
    valid_mask = valid_mask.clone()
    haha_valid_mask = valid_mask.clone()
    new_valid_mask = reshape_for_pooling(valid_mask, pooled_patches_idx)  # (B, NP)

    # Initialize reference tensor: [B, N//G]
    reference = torch.arange(0, NP // T, device=pooled_patches_idx.device).unsqueeze(0).repeat(B, 1)  # [B, N//G]

    # Initialize scores tensor: [B, T * N//G]
    scores = torch.full((B, NP), fill_value=-1, dtype=pooled_cosine_sims.dtype, device=pooled_cosine_sims.device)
    max_t = int(first_padding_frame_idx.max().item())

    for t in range(1, max_t):
        cur_indices = torch.arange(t * NP // T, (t + 1) * NP // T, device=pooled_cosine_sims.device)  # shape: (L,)
        orig_sims = torch.gather(pooled_cosine_sims[:, cur_indices], dim=2, index=reference.unsqueeze(-1)).squeeze(-1)
        
        # No valid mask filtering: simply compute average similarity per batch.
        avg_sim = orig_sims.mean(dim=1)  # shape: (B,)
        
        # For batches where average similarity is below threshold, do a full update.
        update_whole = avg_sim < new_ref_threshold  # shape: (B,)
        
        # For updated batches, mark scores as -1 for these frame indices.
        scores[:, cur_indices] = torch.where(
            update_whole.unsqueeze(1), 
            torch.full_like(orig_sims, -1.0),
            orig_sims
        ).squeeze(-1)
        
        # When full update: update the entire reference vector to the current frame's indices.
        if update_whole.any():
            reference[update_whole] = cur_indices  # full update for those batches
        
        # Otherwise, for each batch individually, update only those positions where similarity is low.
        not_whole = ~update_whole
        if not_whole.any():
            # Loop over batches that did not meet the full update criteria.
            for b in torch.nonzero(not_whole).flatten():
                upd_mask = orig_sims[b] < new_ref_threshold  # boolean mask of shape (L,)
                reference[b, upd_mask] = cur_indices[upd_mask]
        
    for b in range(B):        
        # ignore padding tokens if any
        if first_padding_frame_idx[b] < T:
            scores[b, first_padding_frame_idx[b] * NP // T :] = -100  # ignore padding tokens
    scores[~new_valid_mask] = -100  # ignore tokens that are already pruned
    
    # ---- Incremental pruning ----
    # Sort scores descendingly (higher score means more likely to be pruned)
    _, sorted_indices = scores.sort(descending=True)
    # Prepare to collect the incremental valid masks (in pooled shape)
    valid_mask_list = []
    # working_mask will be updated incrementally (in pooled shape, i.e. same shape as new_valid_mask)
    working_mask = new_valid_mask.clone()
    # Track per-batch how many tokens have been pruned so far (in pooled indices)
    pruned_so_far = torch.zeros(B, dtype=torch.long, device=working_mask.device)
    for it in range(prune_times):
        # For each batch, compute number of tokens to prune in this iteration.
        for b in range(B):
            total = int(num_to_prune[b].item())
            base = total // prune_times
            extra = 1 if it < (total % prune_times) else 0
            prune_count = base + extra
            start = pruned_so_far[b]
            end = start + prune_count
            working_mask[b, sorted_indices[b, start:end]] = False
            pruned_so_far[b] = end

        # Compute pruned_mask in the original (flattened) valid_mask space.
        # Determine which pooled patches are kept.
        keep_patches = torch.nonzero(working_mask, as_tuple=False)  # [num_keep, 2]
        keep_indices = pooled_patches_idx[keep_patches[:, 0], keep_patches[:, 1]].flatten()
        mask_full = torch.zeros_like(valid_mask)
        mask_full[keep_patches[:, 0].repeat_interleave(G), keep_indices] = True
        # Tokens pruned in this call are those that were originally valid but are no longer kept.
        pruned_mask = haha_valid_mask & ~mask_full
        valid_mask[pruned_mask] = False
        valid_mask_list.append(valid_mask.clone())
    
    new_valid_mask = reshape_for_pooling(valid_mask, pooled_patches_idx)
    flag = not torch.any(scores[~new_valid_mask] == -1).item()
    return valid_mask_list[0], flag


def easy_prune(
    valid_mask: torch.Tensor,
    pooled_patches_idx: torch.Tensor,
    pooled_cosine_sims: torch.Tensor,
    first_padding_frame_idx: torch.Tensor,
    new_ref_threshold: float,
    num_to_prune: torch.Tensor,
    T: int,
    prune_times: int = 1,
) -> Tuple[torch.Tensor, bool]:
    """Prune tokens based on their validity and similarity scores.
    Easy version that only looks at the immediate previous frame.

    Args:
        valid_mask (torch.Tensor): A mask indicating valid tokens.
        pooled_patches_idx (torch.Tensor): Tensor of shape (B, NP, G) indicating the pooled patch indices.
        pooled_cosine_sims (torch.Tensor): Cosine similarities between pooled patches.
        first_padding_frame_idx (torch.Tensor): Indices of the first padding frame for each batch.
        new_ref_threshold (float): The new reference threshold for pruning.
        num_to_prune (torch.Tensor): The number of tokens to prune for each batch.
        T (int): The maximum number of frames possible.
        prune_times (int): The number of times to prune tokens in this call.

    Returns:
        Tensors:
        - **new_valid_mask**: Updated valid mask after pruning.
        - **flag**: False if we had to prune patches labelled to ignore (with score -1). The caller decides whether to retry pruning with a lower threshold.
    """
    B, TN = valid_mask.shape
    B, NP, G = pooled_patches_idx.shape
    
    valid_mask = valid_mask.clone()
    haha_valid_mask = valid_mask.clone()
    new_valid_mask = reshape_for_pooling(valid_mask, pooled_patches_idx)  # (B, NP)

    # Initialize scores tensor: [B, T * N//G]
    scores = torch.full((B, NP), fill_value=-1, dtype=pooled_cosine_sims.dtype, device=pooled_cosine_sims.device)
    
    max_t = int(first_padding_frame_idx.max().item())

    for t in range(1, max_t):
        reference = torch.arange((t-1) * NP // T, t * NP // T, device=pooled_patches_idx.device)
        cur_indices = torch.arange(t * NP // T, (t + 1) * NP // T, device=pooled_cosine_sims.device)  # shape: (L,)
        # orig_sims = torch.gather(pooled_cosine_sims[:, cur_indices], dim=2, index=reference.unsqueeze(-1)).squeeze(-1)
        orig_sims = pooled_cosine_sims[:, cur_indices, reference]  # shape: (B, L)
        
        avg_sim = orig_sims.mean(dim=1)  # shape: (B,)
        update_whole = avg_sim < new_ref_threshold  # shape: (B,)
        
        # For updated batches, mark scores as -1 for these frame indices.
        scores[:, cur_indices] = torch.where(
            update_whole.unsqueeze(1), 
            torch.full_like(orig_sims, -1.0),
            orig_sims
        ).squeeze(-1)

    for b in range(B):
        # ignore padding tokens if any
        if first_padding_frame_idx[b] < T:
            scores[b, first_padding_frame_idx[b] * NP // T :] = -100  # ignore padding tokens
    scores[~new_valid_mask] = -100  # ignore tokens that are already pruned
    
    # ---- Incremental pruning ----
    # Sort scores descendingly (higher score means more likely to be pruned)
    _, sorted_indices = scores.sort(descending=True)
    # Prepare to collect the incremental valid masks (in pooled shape)
    valid_mask_list = []
    # working_mask will be updated incrementally (in pooled shape, i.e. same shape as new_valid_mask)
    working_mask = new_valid_mask.clone()
    # Track per-batch how many tokens have been pruned so far (in pooled indices)
    pruned_so_far = torch.zeros(B, dtype=torch.long, device=working_mask.device)
    for it in range(prune_times):
        # For each batch, compute number of tokens to prune in this iteration.
        for b in range(B):
            total = int(num_to_prune[b].item())
            base = total // prune_times
            extra = 1 if it < (total % prune_times) else 0
            prune_count = base + extra
            start = pruned_so_far[b]
            end = start + prune_count
            working_mask[b, sorted_indices[b, start:end]] = False
            pruned_so_far[b] = end

        # Compute pruned_mask in the original (flattened) valid_mask space.
        # Determine which pooled patches are kept.
        keep_patches = torch.nonzero(working_mask, as_tuple=False)  # [num_keep, 2]
        keep_indices = pooled_patches_idx[keep_patches[:, 0], keep_patches[:, 1]].flatten()
        mask_full = torch.zeros_like(valid_mask)
        mask_full[keep_patches[:, 0].repeat_interleave(G), keep_indices] = True
        # Tokens pruned in this call are those that were originally valid but are no longer kept.
        pruned_mask = haha_valid_mask & ~mask_full
        valid_mask[pruned_mask] = False
        valid_mask_list.append(valid_mask.clone())
    
    new_valid_mask = reshape_for_pooling(valid_mask, pooled_patches_idx)
    flag = not torch.any(scores[~new_valid_mask] == -1).item()
    return valid_mask_list[0], flag


def random_prune(
    valid_mask: torch.Tensor,
    pooled_patches_idx: torch.Tensor,
    first_padding_frame_idx: torch.Tensor,
    num_to_prune: torch.Tensor,
    T: int,
) -> Tuple[torch.Tensor, bool]:
    """
    Randomly prune tokens for sanity check.

    Args:
        valid_mask (torch.Tensor): A mask indicating valid tokens.
        num_to_prune (torch.Tensor): The number of tokens to prune for each batch.

    Returns:
        Tensors:
        - **new_valid_mask**: Updated valid mask after pruning.
        - **flag**: Always True (no retry needed).
    """
    B, TN = valid_mask.shape
    device = valid_mask.device
    
    B, NP, G = pooled_patches_idx.shape
    
    valid_mask = valid_mask.clone() # clone so we don't modify the original mask
    pool_batch_idx = torch.arange(B, dtype=torch.long, device=pooled_patches_idx.device)
    pool_batch_idx = torch.tile(pool_batch_idx.view(B, 1, 1), [1, NP, G])
    new_valid_mask = valid_mask.reshape(B, -1)[pool_batch_idx, torch.clip(pooled_patches_idx, 0)]  # (B, NP, G)
    new_valid_mask = new_valid_mask.any(dim=-1)  # [B, NP]

    # Initialize scores tensor: [B, T * N//G]
    scores = torch.full((B, NP), fill_value=100, dtype=pooled_patches_idx.dtype, device=pooled_patches_idx.device)

    for b in range(B):
        if first_padding_frame_idx[b] < T:
            scores[b, first_padding_frame_idx[b] * NP // T :] = -100  # ignore padding tokens
    scores[~new_valid_mask] = -100  # ignore tokens that are already pruned
    
    # Sort and select topk for pruning, but randomly select among valid (non-padding) tokens
    _, indices = scores.sort(descending=True)
    pruned_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    for b in range(B):
        # Only consider indices where scores are not -100 (not padding/pruned)
        valid_indices = indices[b][scores[b, indices[b]] != -100]
        num = min(num_to_prune[b].item(), valid_indices.shape[0])
        if num > 0:
            chosen = torch.randperm(valid_indices.shape[0], device=device)[:num]
            prune_idx = valid_indices[chosen]
            new_valid_mask[b, prune_idx] = False
            
    keep_patches = torch.nonzero(new_valid_mask, as_tuple=False) # [num_keep, 2]
    keep_indices = pooled_patches_idx[keep_patches[:, 0], keep_patches[:, 1]].flatten()  # [num_keep * G]
    
    mask = torch.zeros_like(valid_mask)
    mask[keep_patches[:, 0].repeat_interleave(G), keep_indices] = True
    
    # Set the mask for these kept indices to True
    pruned_mask = ~mask
    pruned_mask = valid_mask & pruned_mask  # Mask for tokens pruned in this call only
    valid_mask[pruned_mask] = False

    return valid_mask, True


class ResidualAttentionBlock(BaseResidualAttentionBlock):
    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), attn_mask=attn_mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class SiglipVisionTransformer(BaseSiglipViT):
    def __init__(self, config: TemporalVitConfig, device=None):
        super().__init__(config, device)
        self.transformer = BlockCollection(config, device)

    def apply_activation_checkpointing(self):
        super().apply_activation_checkpointing()
        fn = vit_activation_checkpoint_function(self.config)
        if self.config.prune and self.config.prune_method == "scorer":
            self.transformer.temporal_token_scorer = checkpoint_wrapper(self.transformer.temporal_token_scorer, checkpoint_fn=fn)

    def reset_parameters(self):
        super().reset_parameters()
        if self.config.prune and self.config.prune_method == "scorer":
            self.transformer.temporal_token_scorer.reset_parameters()

    def apply_fsdp2(self, *args, **kwargs):
        for block in self.transformer.resblocks:
            fully_shard(block, *args, **kwargs)
        if self.config.prune and self.config.prune_method == "scorer":
            fully_shard(self.transformer.temporal_token_scorer, *args, **kwargs)
        fully_shard(self, *args, **kwargs)

    def forward(self, x: torch.Tensor, patch_num: int = None, pooled_patches_idx: torch.Tensor = None, num_images=None, multimodal_type=None) -> List[torch.Tensor]:
        """
        : param x: (batch_size, num_patch, n_pixels)
        """
        if patch_num is None:
            patch_num = self.config.image_num_patch
        B, N, D = x.shape

        x = self.patch_embedding(x)
        x = self.add_pos_emb(x, patch_num)
        return self.transformer(x, pooled_patches_idx, num_images, multimodal_type)
