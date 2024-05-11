from typing import Tuple

import torch
from torch import Tensor

import refiners.fluxion.layers as fl


class MistralRotaryEmbedding(fl.Module):
    """
    This layer implements RoPE
    see [https://arxiv.org/pdf/2104.09864.pdf]
    """
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
        )

        self.cos: Tensor = torch.empty(0, device=device, dtype=dtype)
        self.sin: Tensor = torch.empty(0, device=device, dtype=dtype)

        self._set_cos_sin(
            seq_len=max_position_embeddings,
            device=device,
            dtype=dtype,
        )

    def _set_cos_sin(
        self,
        seq_len: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64)
        freqs = torch.outer(t, self.theta)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos().to(dtype)
        self.sin = emb.sin().to(dtype)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._set_cos_sin(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos[:seq_len].to(dtype=x.dtype),  # type: ignore
            self.sin[:seq_len].to(dtype=x.dtype),  # type: ignore
        )


def apply_rotary_pos_emb(query: Tensor,key: Tensor,cos: Tensor,sin: Tensor) -> Tuple[Tensor, Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        query (torch.Tensor): The query tensor.
        key (torch.Tensor): The key tensor.
        cos (torch.Tensor): The cosine part of the rotary embedding.
        sin (torch.Tensor): The sine part of the rotary embedding.
    Returns:
        tuple(torch.Tensor) comprising of the query and key tensors rotated using the Rotary
        Position Embedding.
    """
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos[None, None, ...]
    sin = sin[None, None, ...]
    q_embed = (query * cos) + (_rotate_half(query) * sin)
    k_embed = (key * cos) + (_rotate_half(key) * sin)
    return q_embed, k_embed
