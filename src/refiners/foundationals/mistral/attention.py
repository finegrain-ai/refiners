import math
from typing import Tuple

import torch
from jaxtyping import Float
from torch import Tensor

import refiners.fluxion.layers as fl

from .rope import MistralRotaryEmbedding, apply_rotary_pos_emb


def repeat_kv(keys: Tensor, values: Tensor, repeats: int) -> Tuple[Tensor, Tensor]:
    # (Seq, N_Heads_KV, Head_Dim) --> (Seq, N_Heads, Head_Dim)
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=1)
    # (Seq, N_Heads_KV, Head_Dim) --> (Seq, N_Heads, Head_Dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=1)
    return keys, values


class MistralAttention(fl.Chain):
    """Mistral Style Attention"""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_kv_heads: int,
        sliding_window: int,
        rope_theta: float,
        max_position_embeddings: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.head_dim = embedding_dim // num_heads
        if (self.head_dim * num_heads) != embedding_dim:
            raise ValueError(
                f"embedding_dim must be divisible by num_heads (got `embedding_dim`: {embedding_dim}"
                f" and `num_heads`: {num_heads})."
            )
        super().__init__(
            fl.Parallel(
                fl.Linear(  # Query projection
                    in_features=embedding_dim,
                    out_features=num_heads * self.head_dim,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
                fl.Linear(  # Key projection
                    in_features=embedding_dim,
                    out_features=num_kv_heads * self.head_dim,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
                fl.Linear(  # Value projection
                    in_features=embedding_dim,
                    out_features=num_kv_heads * self.head_dim,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
            ),
            ScaledDotProductAttention(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=self.head_dim,
                rope_theta=rope_theta,
                max_position_embeddings=max_position_embeddings,
                sliding_window=sliding_window,
                device=device,
                dtype=dtype,
            ),
            fl.Linear(  # Output projection
                in_features=num_heads * self.head_dim,
                out_features=embedding_dim,
                bias=False,
                device=device,
                dtype=dtype,
            ),
        )


class ScaledDotProductAttention(fl.Module):
    """Apply scaled dot product attention to Qurey, Key and Value tensors.
    """
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float,
        max_position_embeddings: int,
        sliding_window: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.repeats = num_heads // num_kv_heads

        self.rotary_emb = MistralRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            device=device,
            dtype=dtype
        )

    def forward(
        self,
        query: Float[Tensor, "batch num_queries (num_heads * head_dim)"],
        key: Float[Tensor, "batch num_keys (num_kv_heads * head_dim)"],
        value: Float[Tensor, "batch num_values (num_kv_heads * head_dim)"],
    ) -> Float[Tensor, "batch num_queries (num_heads * head_dim)"]:
        bsz, seq_len, _ = query.shape

        def _build_sliding_window_mask(seq_len: int) -> Tensor:
            tensor = torch.full((seq_len, seq_len), dtype=query.dtype,
                                fill_value=1, device=query.device)
            mask = torch.tril(tensor, diagonal=0).to(query.dtype)
            mask = torch.triu(mask, diagonal=-self.sliding_window+1)
            return torch.log(mask)

        query = query.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        key, value = repeat_kv(key, value, self.repeats)

        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)

        mask = _build_sliding_window_mask(seq_len)
        scores += mask[None, None, ...]

        scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(scores, value)
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(bsz, seq_len, -1)
        return output
