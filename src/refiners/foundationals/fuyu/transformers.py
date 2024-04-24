from typing import Tuple

import torch
from torch import Tensor, device as Device, dtype as DType

import refiners.fluxion.layers as fl
from refiners.foundationals.dinov2.vit import FeedForward
from refiners.foundationals.fuyu.common import CustomReshape, ScaledDotProductAttentionWithAttnMask, SquaredReLU


class RotaryPositionalEmbedding(fl.Module):
    """
    This layer implements RoPE
    see [https://arxiv.org/pdf/2104.09864.pdf]
    """

    def __init__(
        self, dim: int = 32, base: int = 10_000, device: Device | str | None = None, dtype: DType | None = None
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.device = device
        self.dtype = dtype
        # Create positional encodings
        self.theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)).to(
            self.device
        )
        self.cos: Tensor = torch.empty(0, device=self.device, dtype=self.dtype)
        self.sin: Tensor = torch.empty(0, device=self.device, dtype=self.dtype)

    def _cache(self, seq_len: int) -> None:
        if seq_len < self.cos.shape[0]:
            return
        t = torch.arange(seq_len, device=self.device, dtype=torch.int64).float()
        freqs = torch.outer(t, self.theta)
        embs = torch.cat([freqs, freqs], dim=-1)
        self.cos = embs.cos().to(self.dtype)
        self.sin = embs.sin().to(self.dtype)

    def _neg_half(self, x: Tensor) -> Tensor:
        return torch.cat([-x[:, :, :, self.dim // 2 :], x[:, :, :, : self.dim // 2]], dim=-1)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        seq_len = q.shape[1]
        self._cache(seq_len)

        # [batch_size, seq_length, num_heads, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # Q rotation
        q_rope, q_pass = q[..., : self.dim], q[..., self.dim :]
        q_neg_half = self._neg_half(q_rope)
        q_rope = (q_rope * self.cos[:seq_len]) + (q_neg_half * self.sin[:seq_len])
        q_rot = torch.cat((q_rope, q_pass), dim=-1)

        # K rotation
        k_rope, k_pass = k[..., : self.dim], k[..., self.dim :]
        k_neg_half = self._neg_half(k_rope)
        k_rope = (k_rope * self.cos[:seq_len]) + (k_neg_half * self.sin[:seq_len])
        k_rot = torch.cat((k_rope, k_pass), dim=-1)

        # [batch_size, num_heads, seq_length, head_dim] -> [batch_size, seq_length, num_heads, head_dim]
        q_rot = q_rot.transpose(1, 2)
        k_rot = k_rot.transpose(1, 2)

        return q_rot, k_rot, v


class QKVProjection(fl.Chain):
    """
    Apply query, key, value projection

    Args:
        embedding_dim: The embedding dimension of the input and output Tensor.
        num_heads: The number of heads of the attention mechanism.
        use_bias: Whether to use bias in the linear layers.
        norm_eps: epsilon for Layer Norm
        device: The device to use.
        dtype: The dtype to use
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        use_bias: bool,
        norm_eps: float,
        device: Device | str | None,
        dtype: DType | None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.heads_dim = embedding_dim // num_heads
        self.use_bias = use_bias
        self.norm_eps = norm_eps

        super().__init__(
            fl.Linear(
                in_features=self.embedding_dim,
                out_features=3 * self.num_heads * self.heads_dim,
                bias=self.use_bias,
                device=device,
                dtype=dtype,
            ),
            CustomReshape(self.num_heads, 3, self.heads_dim),
            fl.Parallel(
                fl.Chain(  # Q projection
                    fl.Slicing(dim=-2, start=0, end=1),
                    fl.Squeeze(dim=-2),
                    fl.LayerNorm(normalized_shape=self.heads_dim, eps=self.norm_eps, device=device, dtype=dtype),
                ),
                fl.Chain(  # K projection
                    fl.Slicing(dim=-2, start=1, end=2),
                    fl.Squeeze(dim=-2),
                    fl.LayerNorm(normalized_shape=self.heads_dim, eps=self.norm_eps, device=device, dtype=dtype),
                ),
                fl.Chain(  # V projection
                    fl.Slicing(dim=-2, start=2, end=3),
                    fl.Squeeze(dim=-2),
                ),
            ),
        )


class FuyuSelfAttention(fl.Chain):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 1,
        base: int = 10_000,
        norm_eps: float = 1e-6,
        partial_rotary_factor: float = 0.5,
        use_bias: bool = True,
        is_optimized: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize the Attention layer.

        Args:
            embedding_dim: The embedding dimension of the input and output Tensor.
            num_heads: The number of heads to use.
            base: constant used to compute the rotations in the Rotary Positional Embedding
            norm_eps: epsilon for Layer Norm
            use_bias: Whether to use bias in the linear layers.
            is_optimized: Whether to use optimized attention.
            device: The device to use.
            dtype: The dtype to use.
        """
        assert (
            embedding_dim % num_heads == 0
        ), f"embedding_dim {embedding_dim} must be divisible by num_heads {num_heads}"
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.heads_dim = embedding_dim // num_heads
        self.use_bias = use_bias
        self.is_optimized = is_optimized
        self.base = base
        self.norm_eps = norm_eps
        self.partial_rotary_factor = partial_rotary_factor
        self.rotary_dim = int(self.heads_dim * self.partial_rotary_factor)

        super().__init__(
            QKVProjection(
                embedding_dim=self.embedding_dim,
                num_heads=self.num_heads,
                use_bias=self.use_bias,
                norm_eps=self.norm_eps,
                device=device,
                dtype=dtype,
            ),
            RotaryPositionalEmbedding(
                dim=self.rotary_dim,
                base=self.base,
                device=device,
                dtype=dtype,
            fl.Distribute(  # B seq_len num_heads heads_dim => B seq_len embdedding_dim
                CustomReshape(self.embedding_dim),  # Q
                CustomReshape(self.embedding_dim),  # K
                CustomReshape(self.embedding_dim),  # V
            ),
            ScaledDotProductAttentionWithAttnMask(num_heads=self.num_heads, is_optimized=self.is_optimized),
            fl.Linear(  # Output projection [B seqlen embedding dim]
                in_features=self.embedding_dim,
                out_features=self.embedding_dim,
                bias=self.use_bias,
                device=device,
                dtype=dtype,
            ),
        )


class FuyuTransformerLayer(fl.Chain):
    """Apply a multi-head self-attention mechanism to the input Tensor."""

    def __init__(
        self,
        embedding_dim: int = 4_096,
        feedforward_dim: int = 16_384,
        num_heads: int = 64,
        norm_eps: float = 1e-6,
        base: int = 10_000,
        partial_rotary_factor: float = 0.5,
        use_bias: bool = True,
        is_optimized: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.norm_eps = norm_eps
        self.feedforward_dim = feedforward_dim
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor

        super().__init__(
            fl.Residual(
                fl.LayerNorm(normalized_shape=self.embedding_dim, eps=self.norm_eps, device=device, dtype=dtype),
                FuyuSelfAttention(
                    embedding_dim=self.embedding_dim,
                    num_heads=self.num_heads,
                    base=self.base,
                    norm_eps=self.norm_eps,
                    partial_rotary_factor=self.partial_rotary_factor,
                    use_bias=use_bias,
                    is_optimized=is_optimized,
                    device=device,
                    dtype=dtype,
                ),
            ),
            fl.Residual(
                fl.LayerNorm(normalized_shape=embedding_dim, eps=norm_eps, device=device, dtype=dtype),
                FeedForward(
                    embedding_dim=embedding_dim,
                    feedforward_dim=feedforward_dim,
                    activation=SquaredReLU(),
                    device=device,
                    dtype=dtype,
                ),
            ),
        )


class FuyuTransformer(fl.Chain):
    """Alias for a Chain of FuyuTransformerLayer."""
