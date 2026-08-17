import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.fsdp import fully_shard

from olmo.config import BaseConfig, D
from olmo.nn.flash_attention_api import dispatch_flash_attn


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _round_up_multiple(value: int, multiple_of: int) -> int:
    if multiple_of <= 0:
        return value
    return int(math.ceil(value / multiple_of) * multiple_of)


def _init_linear(linear: nn.Linear, *, zero: bool = False, scale: float = 1.0) -> None:
    if zero:
        nn.init.zeros_(linear.weight)
    else:
        nn.init.xavier_uniform_(linear.weight)
        if scale != 1.0:
            with torch.no_grad():
                linear.weight.mul_(scale)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _tensor_all_finite(tensor: Optional[torch.Tensor]) -> Optional[bool]:
    if tensor is None:
        return None
    return bool(torch.isfinite(tensor).all().item())


def _tensor_max_abs(tensor: Optional[torch.Tensor]) -> Optional[float]:
    if tensor is None:
        return None
    safe = torch.nan_to_num(tensor.detach().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return float(safe.abs().max().item())


def _nonfinite_batch_rows(tensor: Optional[torch.Tensor]) -> Optional[list[int]]:
    if tensor is None or tensor.ndim == 0:
        return None
    bad = ~torch.isfinite(tensor)
    if bad.ndim == 1:
        rows = bad
    else:
        rows = bad.reshape(bad.shape[0], -1).any(dim=1)
    return rows.nonzero(as_tuple=False).flatten().tolist()


class ActionExpertRMSNorm(nn.Module):
    def __init__(
        self,
        size: int,
        *,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.size = size
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(size, device=device))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            dtype = x.dtype
            x_float = x.to(torch.float32)
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
            out = x_float * torch.rsqrt(variance + self.eps)
            out = out.to(dtype)
        if self.weight is not None:
            out = out * self.weight
        return out

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)


class ActionExpertRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim.")
        self.head_dim = head_dim
        self.base = base

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, half_dim, device=q.device, dtype=torch.float32) / max(half_dim, 1))
        )
        positions = torch.arange(seq_len, device=q.device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos().to(dtype=q.dtype).view(1, 1, seq_len, half_dim)
        sin = freqs.sin().to(dtype=q.dtype).view(1, 1, seq_len, half_dim)

        def _apply(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., :half_dim], x[..., half_dim:]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return _apply(q), _apply(k)


class ActionExpertSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_dropout = attn_dropout
        self.qk_norm = qk_norm
        self.q_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.k_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.rope = ActionExpertRotaryEmbedding(self.head_dim) if use_rope else None
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_drop = nn.Dropout(proj_dropout)

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.q_norm is None or self.k_norm is None:
            return q, k
        return self.q_norm(q), self.k_norm(k)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        dropout_p = self.attn_dropout if self.training else 0.0
        if attn_mask is None and q.is_cuda:
            try:
                return dispatch_flash_attn(
                    q,
                    k,
                    v,
                    dropout_p=dropout_p,
                    causal=is_causal,
                )
            except RuntimeError:
                pass

        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
        return out.transpose(1, 2).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        q, k = self._apply_qk_norm(q, k)
        if self.rope is not None:
            q, k = self.rope(q, k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.contiguous()
        out = self._attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        out = out.reshape(bsz, seq_len, self.hidden_size)
        out = self.out_proj(out)
        return self.out_drop(out)


class ActionExpertCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_dropout = attn_dropout
        self.q_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.k_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.kv_proj = nn.Linear(hidden_size, hidden_size * 2)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_drop = nn.Dropout(proj_dropout)

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.q_norm is None or self.k_norm is None:
            return q, k
        return self.q_norm(q), self.k_norm(k)

    def _as_heads(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            if x.shape[2] == self.num_heads:
                return x
            if x.shape[1] == self.num_heads:
                return x.transpose(1, 2).contiguous()
            raise ValueError(f"Unexpected cross-attention KV shape {tuple(x.shape)}")
        if x.dim() != 3:
            raise ValueError(f"Expected 3D/4D cross-attention KV, got {tuple(x.shape)}")
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dropout_p = self.attn_dropout if self.training else 0.0
        if attn_mask is None and q.is_cuda:
            try:
                return dispatch_flash_attn(
                    q,
                    k,
                    v,
                    dropout_p=dropout_p,
                    causal=False,
                )
            except RuntimeError:
                pass

        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        return out.transpose(1, 2).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv: Optional[torch.Tensor] = None,
        kv_k: Optional[torch.Tensor] = None,
        kv_v: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (kv_k is None) != (kv_v is None):
            raise ValueError("kv_k and kv_v must both be provided or both be None.")
        if kv is not None and kv_k is not None:
            raise ValueError("Provide either kv or kv_k/kv_v, not both.")
        bsz, tgt_len, _ = x.shape
        q = self.q_proj(x).view(bsz, tgt_len, self.num_heads, self.head_dim)
        if kv_k is not None and kv_v is not None:
            k = self._as_heads(kv_k)
            v = self._as_heads(kv_v)
        else:
            if kv is None:
                raise ValueError("cross-attention requires kv or kv_k/kv_v.")
            src_len = kv.shape[1]
            kv_proj = self.kv_proj(kv).view(bsz, src_len, 2, self.num_heads, self.head_dim)
            k = kv_proj[:, :, 0]
            v = kv_proj[:, :, 1]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        q, k = self._apply_qk_norm(q, k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        out = self._attention(q, k, v, attn_mask=attn_mask)
        out = out.reshape(bsz, tgt_len, self.hidden_size)
        out = self.out_proj(out)
        return self.out_drop(out)


class ActionExpertMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        mlp_ratio: float,
        multiple_of: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = _round_up_multiple(int(hidden_size * mlp_ratio), multiple_of)
        self.up_proj = nn.Linear(hidden_size, inner_dim)
        self.gate_proj = nn.Linear(hidden_size, inner_dim)
        self.down_proj = nn.Linear(inner_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return self.dropout(x)


class ActionExpertModulation(nn.Module):
    def __init__(self, hidden_size: int, num_chunks: int) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.linear = nn.Linear(hidden_size, num_chunks * hidden_size)

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        return self.linear(self.act(conditioning))


class ActionExpertBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float,
        ffn_multiple_of: int,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        rope: bool = True,
    ) -> None:
        super().__init__()
        self.self_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.cross_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.ff_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.self_attn = ActionExpertSelfAttention(
            hidden_size,
            num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
            use_rope=rope,
        )
        self.cross_attn = ActionExpertCrossAttention(
            hidden_size,
            num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
        )
        self.mlp = ActionExpertMLP(
            hidden_size,
            mlp_ratio=mlp_ratio,
            multiple_of=ffn_multiple_of,
            dropout=dropout,
        )
        self.modulation = ActionExpertModulation(hidden_size, 9)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        *,
        cross_context: Optional[torch.Tensor] = None,
        cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.modulation(conditioning).chunk(9, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.self_attn(
            _modulate(self.self_norm(x), shift_msa, scale_msa),
            attn_mask=self_attn_mask,
            is_causal=is_causal,
        )

        attn_kwargs = {}
        if cross_kv is not None:
            attn_kwargs["kv_k"] = cross_kv[0]
            attn_kwargs["kv_v"] = cross_kv[1]
        else:
            if cross_context is None:
                raise ValueError("cross-attention requires cross_context or cross_kv.")
            attn_kwargs["kv"] = cross_context
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(
            _modulate(self.cross_norm(x), shift_mca, scale_mca),
            attn_mask=attn_mask,
            **attn_kwargs,
        )

        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            _modulate(self.ff_norm(x), shift_mlp, scale_mlp)
        )
        return x


class ActionExpertFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_dim: int) -> None:
        super().__init__()
        self.norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.modulation = ActionExpertModulation(hidden_size, 2)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(conditioning).chunk(2, dim=1)
        return self.linear(_modulate(self.norm(x), shift, scale))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() > 1:
            timesteps = timesteps.view(timesteps.shape[0], -1)[:, 0]
        device = timesteps.device
        half_dim = self.dim // 2
        freq = torch.exp(
            torch.arange(half_dim, device=device, dtype=timesteps.dtype)
            * (-math.log(10000.0) / max(half_dim - 1, 1))
        )
        args = timesteps[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


@dataclass
class ActionExpertConfig(BaseConfig):
    max_horizon: int = 30
    max_action_dim: int = 32
    hidden_size: int = 1024
    num_layers: int = 32
    num_heads: int = 16
    mlp_ratio: float = 8.0 / 3.0
    ffn_multiple_of: int = 256
    timestep_embed_dim: int = 256
    dropout: float = 0.0
    attn_dropout: float = 0.0
    context_layer_norm: bool = True
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    rope: bool = True
    rope_on_cross_attention: bool = False
    causal_attn: bool = False
    compile: Optional[str] = "blocks"

    def build(
        self,
        llm_dim: int,
        llm_kv_dim: Optional[int] = None,
        llm_num_kv_heads: Optional[int] = None,
        llm_num_layers: Optional[int] = None,
        device=None,
    ) -> "ActionExpert":
        return ActionExpert(
            self,
            llm_dim=llm_dim,
            llm_kv_dim=llm_kv_dim,
            llm_num_kv_heads=llm_num_kv_heads,
            llm_num_layers=llm_num_layers,
            device=device,
        )

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        if "implementation" in config:
            implementation = str(config["implementation"])
            if implementation not in {"new", "modern"}:
                raise ValueError(
                    "Only the modern MolmoAct2 action expert is supported. "
                    f"Found legacy implementation={implementation!r}."
                )
            del config["implementation"]
        if "action_dim" in config:
            legacy_dim = int(config["action_dim"])
            if "max_action_dim" in config and int(config["max_action_dim"]) != legacy_dim:
                raise ValueError(
                    "Found conflicting action expert action dimensions in config: "
                    f"action_dim={legacy_dim} vs max_action_dim={int(config['max_action_dim'])}."
                )
            config["max_action_dim"] = legacy_dim
            del config["action_dim"]
        if "context_connection" in config:
            del config["context_connection"]
        return config


class ActionExpert(nn.Module):
    def __init__(
        self,
        config: ActionExpertConfig,
        llm_dim: int,
        llm_kv_dim: Optional[int] = None,
        llm_num_kv_heads: Optional[int] = None,
        llm_num_layers: Optional[int] = None,
        device=None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.llm_dim = llm_dim
        self.llm_kv_dim = llm_dim if llm_kv_dim is None else llm_kv_dim
        self.llm_num_kv_heads = llm_num_kv_heads
        self.llm_num_layers = llm_num_layers
        self.use_kv_flat_condition = True
        self.use_kv_condition = True
        self.action_head_dim = config.hidden_size // config.num_heads
        self.llm_kv_head_dim = None

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(config.timestep_embed_dim),
            nn.Linear(config.timestep_embed_dim, config.hidden_size, device=device),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size, device=device),
        )

        self.action_embed = nn.Linear(config.max_action_dim, config.hidden_size, device=device)
        self.state_encoder = nn.Linear(config.hidden_size, config.hidden_size, device=device)
        self.state_norm = ActionExpertRMSNorm(config.hidden_size, eps=1e-6, device=device)

        self.context_k_proj = nn.Linear(self.llm_kv_dim, config.hidden_size, bias=False, device=device)
        self.context_v_proj = nn.Linear(self.llm_kv_dim, config.hidden_size, bias=False, device=device)

        self.context_norm = (
            ActionExpertRMSNorm(config.hidden_size, eps=1e-6)
            if config.context_layer_norm
            else nn.Identity()
        )

        self.blocks = nn.ModuleList(
            [
                ActionExpertBlock(
                    config.hidden_size,
                    config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    ffn_multiple_of=config.ffn_multiple_of,
                    attn_dropout=config.attn_dropout,
                    dropout=config.dropout,
                    qk_norm=config.qk_norm,
                    qk_norm_eps=config.qk_norm_eps,
                    rope=config.rope,
                )
                for _ in range(config.num_layers)
            ]
        )
        if self.use_kv_condition:
            for block in self.blocks:
                block.cross_attn.kv_proj.weight.requires_grad = False
                if block.cross_attn.kv_proj.bias is not None:
                    block.cross_attn.kv_proj.bias.requires_grad = False

        self.final_layer = ActionExpertFinalLayer(config.hidden_size, config.max_action_dim)
        self.reset_parameters()

    def _reshape_hidden_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], self.config.num_heads, self.action_head_dim)

    def _flatten_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], x.shape[1], self.hidden_size)

    def reset_parameters(self):
        for module in self.time_embed.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module)
        _init_linear(self.action_embed)
        _init_linear(self.state_encoder)
        self.state_norm.reset_parameters()
        self.context_k_proj.reset_parameters()
        self.context_v_proj.reset_parameters()
        if isinstance(self.context_norm, ActionExpertRMSNorm):
            self.context_norm.reset_parameters()

        residual_scale = (2 * max(self.config.num_layers, 1)) ** -0.5
        for block in self.blocks:
            _init_linear(block.self_attn.qkv)
            _init_linear(block.self_attn.out_proj, scale=residual_scale)
            _init_linear(block.cross_attn.q_proj)
            _init_linear(block.cross_attn.kv_proj)
            _init_linear(block.cross_attn.out_proj, scale=residual_scale)
            _init_linear(block.mlp.up_proj)
            _init_linear(block.mlp.gate_proj)
            _init_linear(block.mlp.down_proj, scale=residual_scale)
            _init_linear(block.modulation.linear, zero=True)
            block.self_norm.reset_parameters()
            block.cross_norm.reset_parameters()
            block.ff_norm.reset_parameters()
            if block.self_attn.q_norm is not None:
                block.self_attn.q_norm.reset_parameters()
            if block.self_attn.k_norm is not None:
                block.self_attn.k_norm.reset_parameters()
            if block.cross_attn.q_norm is not None:
                block.cross_attn.q_norm.reset_parameters()
            if block.cross_attn.k_norm is not None:
                block.cross_attn.k_norm.reset_parameters()

        self.final_layer.norm.reset_parameters()
        _init_linear(self.final_layer.modulation.linear, zero=True)
        _init_linear(self.final_layer.linear, zero=True)

    def prepare_state_dict_for_loading(self, state_dict: dict[str, torch.Tensor], prefix: str = "") -> int:
        """Add cleaned-code compatibility keys that older MolmoAct2 checkpoints do not contain."""
        added = 0

        def sample_tensor() -> torch.Tensor | None:
            for key in (
                f"{prefix}action_embed.weight",
                f"{prefix}final_layer.linear.weight",
                f"{prefix}context_k_proj.weight",
            ):
                value = state_dict.get(key)
                if isinstance(value, torch.Tensor):
                    return value
            return None

        sample = sample_tensor()
        dtype = sample.dtype if sample is not None else torch.float32
        device = sample.device if sample is not None else torch.device("cpu")
        hidden_size = int(self.config.hidden_size)

        state_encoder_weight_key = f"{prefix}state_encoder.weight"
        if state_encoder_weight_key not in state_dict:
            state_dict[state_encoder_weight_key] = torch.eye(hidden_size, dtype=dtype, device=device)
            added += 1

        state_encoder_bias_key = f"{prefix}state_encoder.bias"
        if state_encoder_bias_key not in state_dict:
            state_dict[state_encoder_bias_key] = torch.zeros(hidden_size, dtype=dtype, device=device)
            added += 1

        # Under the single supported kv_cache conditioning path, projected K/V tensors
        # are passed directly into cross-attention. The kv_proj module is frozen and
        # inactive, but strict checkpoint loading still expects its parameters.
        for layer_idx in range(len(self.blocks)):
            q_proj_key = f"{prefix}blocks.{layer_idx}.cross_attn.q_proj.weight"
            q_proj = state_dict.get(q_proj_key)
            layer_hidden_size = int(q_proj.shape[0]) if isinstance(q_proj, torch.Tensor) else hidden_size
            layer_dtype = q_proj.dtype if isinstance(q_proj, torch.Tensor) else dtype
            layer_device = q_proj.device if isinstance(q_proj, torch.Tensor) else device

            kv_weight_key = f"{prefix}blocks.{layer_idx}.cross_attn.kv_proj.weight"
            if kv_weight_key not in state_dict:
                state_dict[kv_weight_key] = torch.zeros(
                    (layer_hidden_size * 2, layer_hidden_size),
                    dtype=layer_dtype,
                    device=layer_device,
                )
                added += 1

            kv_bias_key = f"{prefix}blocks.{layer_idx}.cross_attn.kv_proj.bias"
            if kv_bias_key not in state_dict:
                state_dict[kv_bias_key] = torch.zeros(
                    layer_hidden_size * 2,
                    dtype=layer_dtype,
                    device=layer_device,
                )
                added += 1

        return added

    def timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.time_embed(timesteps)

    def get_continuous_state_conditioning_modules(self) -> Sequence[nn.Module]:
        return [self.state_encoder, self.state_norm]

    def freeze_continuous_state_conditioning(self) -> None:
        for module in self.get_continuous_state_conditioning_modules():
            module.requires_grad_(False)

    def _encode_states(
        self,
        states: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if states is None:
            return None
        if states.dim() == 2:
            states = states.unsqueeze(1)
        if states.shape[-1] != self.hidden_size:
            feat_dim = states.shape[-1]
            if feat_dim < self.hidden_size:
                states = F.pad(states, (0, self.hidden_size - feat_dim))
            else:
                states = states[..., : self.hidden_size]
        encoded = self.state_encoder(states)
        return self.state_norm(encoded)

    def _project_kv_tensor(self, x: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        flat = self.context_norm(proj(x))
        return self._reshape_hidden_to_heads(flat)

    def _prepare_kv_context(
        self,
        encoder_kv_states: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        encoded_states: Optional[torch.Tensor],
    ) -> Sequence[Tuple[torch.Tensor, torch.Tensor]]:
        if self.context_k_proj is None or self.context_v_proj is None:
            raise RuntimeError("KV conditioning requested but KV projection modules are not initialized.")
        kv_contexts = []
        state_heads = self._reshape_hidden_to_heads(encoded_states) if encoded_states is not None else None

        if len(encoder_kv_states) != len(self.blocks):
            raise ValueError(
                "MolmoAct2 action expert expects one KV state per action expert block "
                f"(got {len(encoder_kv_states)}, expected {len(self.blocks)})."
            )

        for layer_idx, (k_in, v_in) in enumerate(encoder_kv_states):
            if not torch.isfinite(k_in).all() or not torch.isfinite(v_in).all():
                raise RuntimeError(
                    "Non-finite encoder KV states before action expert projection: "
                    f"layer_idx={layer_idx}, "
                    f"k_finite={_tensor_all_finite(k_in)}, "
                    f"v_finite={_tensor_all_finite(v_in)}, "
                    f"k_bad_batch_rows={_nonfinite_batch_rows(k_in)}, "
                    f"v_bad_batch_rows={_nonfinite_batch_rows(v_in)}, "
                    f"k_shape={tuple(k_in.shape)}, "
                    f"v_shape={tuple(v_in.shape)}."
                )
            k_ctx = self._project_kv_tensor(k_in, self.context_k_proj)
            v_ctx = self._project_kv_tensor(v_in, self.context_v_proj)
            if not torch.isfinite(k_ctx).all() or not torch.isfinite(v_ctx).all():
                raise RuntimeError(
                    "Non-finite projected KV context in action expert: "
                    f"layer_idx={layer_idx}, "
                    f"k_in_finite={_tensor_all_finite(k_in)}, "
                    f"v_in_finite={_tensor_all_finite(v_in)}, "
                    f"k_ctx_finite={_tensor_all_finite(k_ctx)}, "
                    f"v_ctx_finite={_tensor_all_finite(v_ctx)}, "
                    f"k_in_max_abs={_tensor_max_abs(k_in)}, "
                    f"v_in_max_abs={_tensor_max_abs(v_in)}, "
                    f"k_ctx_max_abs={_tensor_max_abs(k_ctx)}, "
                    f"v_ctx_max_abs={_tensor_max_abs(v_ctx)}."
                )
            if state_heads is not None:
                k_ctx = torch.cat([k_ctx, state_heads], dim=1)
                v_ctx = torch.cat([v_ctx, state_heads], dim=1)
            kv_contexts.append((k_ctx, v_ctx))
        return kv_contexts

    def _build_cross_attention_mask(
        self,
        encoder_attention_mask: Optional[torch.Tensor],
        encoded_states: Optional[torch.Tensor],
        batch_size: int,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        state_seq_len = 0 if encoded_states is None else encoded_states.shape[1]
        if encoder_attention_mask is None and state_seq_len == 0:
            return None
        if encoder_attention_mask is None:
            return None
        if encoder_attention_mask.dim() == 2:
            mask = encoder_attention_mask[:, None, None, :].to(dtype=dtype)
        else:
            mask = encoder_attention_mask.to(dtype=dtype)
        if state_seq_len > 0:
            ones = torch.ones(
                batch_size,
                1,
                1,
                state_seq_len,
                device=mask.device,
                dtype=mask.dtype,
            )
            mask = torch.cat([mask, ones], dim=-1)
        return (1.0 - mask) * torch.finfo(dtype).min

    def _build_self_attention_mask(
        self,
        action_attention_mask: Optional[torch.Tensor],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        mask = None
        if action_attention_mask is not None:
            valid = action_attention_mask.to(device=device, dtype=torch.bool)
            if valid.ndim != 2 or valid.shape[1] != seq_len:
                raise ValueError(
                    f"Expected action_attention_mask shape (batch, {seq_len}), got {tuple(valid.shape)}."
                )
            key_mask = (~valid)[:, None, None, :].to(dtype=dtype)
            mask = key_mask * torch.finfo(dtype).min
        if self.config.causal_attn:
            causal = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).triu(diagonal=1)
            causal = causal.unsqueeze(0).unsqueeze(0).to(dtype=dtype) * torch.finfo(dtype).min
            mask = causal if mask is None else (mask + causal)
        return mask

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_kv_states: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        action_attention_mask: Optional[torch.Tensor] = None,
        state_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if encoder_kv_states is None:
            raise ValueError("Provide encoder_kv_states for MolmoAct2 action expert conditioning.")
        if len(encoder_kv_states) == 0:
            raise ValueError("Expected at least one encoder kv state for action expert conditioning")
        bsz, seq_len, _ = actions.shape
        if seq_len > self.config.max_horizon:
            raise ValueError(
                f"Action sequence length {seq_len} exceeds configured max_horizon={self.config.max_horizon}"
            )

        timestep_embed = self.time_embed(timesteps)
        if not torch.isfinite(timestep_embed).all():
            raise RuntimeError(
                "Non-finite timestep embedding in action expert: "
                f"timesteps_finite={_tensor_all_finite(timesteps)}, "
                f"timestep_embed_finite={_tensor_all_finite(timestep_embed)}, "
                f"timesteps_max_abs={_tensor_max_abs(timesteps)}, "
                f"timestep_embed_max_abs={_tensor_max_abs(timestep_embed)}."
            )
        encoded_states = self._encode_states(state_embeddings)
        if encoded_states is not None and not torch.isfinite(encoded_states).all():
            raise RuntimeError(
                "Non-finite encoded state conditioning in action expert: "
                f"state_embeddings_finite={_tensor_all_finite(state_embeddings)}, "
                f"encoded_states_finite={_tensor_all_finite(encoded_states)}, "
                f"state_embeddings_max_abs={_tensor_max_abs(state_embeddings)}, "
                f"encoded_states_max_abs={_tensor_max_abs(encoded_states)}."
            )
        x = self.action_embed(actions)
        if not torch.isfinite(x).all():
            raise RuntimeError(
                "Non-finite action embedding in action expert: "
                f"actions_finite={_tensor_all_finite(actions)}, "
                f"action_embed_finite={_tensor_all_finite(x)}, "
                f"actions_max_abs={_tensor_max_abs(actions)}, "
                f"action_embed_max_abs={_tensor_max_abs(x)}."
            )
        conditioning = timestep_embed
        valid_action = None
        if action_attention_mask is not None:
            valid_action = action_attention_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
            x = x * valid_action
            if not torch.isfinite(x).all():
                raise RuntimeError(
                    "Non-finite masked action embedding in action expert: "
                    f"action_attention_mask_finite={_tensor_all_finite(action_attention_mask)}, "
                    f"masked_action_embed_finite={_tensor_all_finite(x)}, "
                    f"masked_action_embed_max_abs={_tensor_max_abs(x)}."
                )

        kv_contexts = self._prepare_kv_context(
            encoder_kv_states,
            encoded_states,
        )

        cross_mask = self._build_cross_attention_mask(
            encoder_attention_mask,
            encoded_states,
            bsz,
            x.dtype,
        )
        self_mask = self._build_self_attention_mask(
            action_attention_mask,
            seq_len,
            x.device,
            x.dtype,
        )

        for block_idx, (block, kv_context) in enumerate(zip(self.blocks, kv_contexts)):
            x = block(
                x,
                conditioning,
                cross_kv=kv_context,
                self_attn_mask=self_mask,
                attn_mask=cross_mask,
                is_causal=self.config.causal_attn,
            )
            if valid_action is not None:
                x = x * valid_action
            if not torch.isfinite(x).all():
                k_ctx, v_ctx = kv_context
                raise RuntimeError(
                    "Non-finite hidden state inside KV-conditioned action expert block: "
                    f"block_idx={block_idx}, "
                    f"x_finite={_tensor_all_finite(x)}, "
                    f"k_ctx_finite={_tensor_all_finite(k_ctx)}, "
                    f"v_ctx_finite={_tensor_all_finite(v_ctx)}, "
                    f"conditioning_finite={_tensor_all_finite(conditioning)}, "
                    f"k_ctx_max_abs={_tensor_max_abs(k_ctx)}, "
                    f"v_ctx_max_abs={_tensor_max_abs(v_ctx)}, "
                    f"conditioning_max_abs={_tensor_max_abs(conditioning)}, "
                    f"x_max_abs={_tensor_max_abs(x)}, "
                    f"self_mask_present={self_mask is not None}, "
                    f"cross_mask_present={cross_mask is not None}."
                )

        out = self.final_layer(x, conditioning)
        if valid_action is not None:
            out = out * valid_action
        if not torch.isfinite(out).all():
            raise RuntimeError(
                "Non-finite final action expert output before return: "
                f"out_finite={_tensor_all_finite(out)}, "
                f"x_finite={_tensor_all_finite(x)}, "
                f"conditioning_finite={_tensor_all_finite(conditioning)}, "
                f"out_max_abs={_tensor_max_abs(out)}, "
                f"x_max_abs={_tensor_max_abs(x)}, "
                f"conditioning_max_abs={_tensor_max_abs(conditioning)}."
            )
        return out

    def reset_with_pretrained_weights(self):
        return

    def apply_activation_checkpointing(self):
        self.blocks = nn.ModuleList([checkpoint_wrapper(block) for block in self.blocks])

    def apply_compile(self, **compile_kwargs):
        if self.config.compile == "blocks":
            for block in self.blocks:
                block.compile(**compile_kwargs)
        elif self.config.compile is not None:
            raise NotImplementedError(self.config.compile)

    def apply_fsdp2(self, **fully_shard_kwargs):
        fully_shard(self, **fully_shard_kwargs)
