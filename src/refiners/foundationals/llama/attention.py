import math

import torch
from torch import nn

import refiners.fluxion.layers as fl

from .common import LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv


class LLamaAttention(fl.Chain):
    """Llama Attention layer ."""

    def __init__(
        self,
        embedding_dim: int,
        num_att_heads: int = 1,
        num_kv_heads: int = 1,
        max_position_embeddings: int = 2048,
        scaling_factor: float = 1.0,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        use_bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_heads = num_att_heads
        self.num_kv_heads = num_kv_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.scaling_factor = scaling_factor
        self.attention_dropout = attention_dropout
        self.head_dim = embedding_dim // num_att_heads
        self.use_bias = use_bias
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        super().__init__(
            fl.Distribute(
                fl.Linear(  # Query projection
                    in_features=self.embedding_dim,
                    out_features=self.num_heads * self.head_dim,
                    bias=self.use_bias,
                    device=device,
                    dtype=dtype,
                ),
                fl.Linear(  # Key projection
                    in_features=self.embedding_dim,
                    out_features=self.num_kv_heads * self.head_dim,
                    bias=self.use_bias,
                    device=device,
                    dtype=dtype,
                ),
                fl.Linear(  # Value projection
                    in_features=self.embedding_dim,
                    out_features=self.num_kv_heads * self.head_dim,
                    bias=self.use_bias,
                    device=device,
                    dtype=dtype,
                ),
            ),
            LlamaBaseOpAttention(
                embedding_dim=self.embedding_dim,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=self.scaling_factor,
                rope_theta=self.rope_theta,
                attention_dropout=self.attention_dropout,
                device=self.device,
                dtype=self.dtype,
            ),
            fl.Linear(  # Output projection
                in_features=self.embedding_dim,
                out_features=self.embedding_dim,
                bias=True,
                device=device,
                dtype=dtype,
            ),
        )


class LLamaSdpaAttention(fl.Chain):
    """Llama attention layer using torch.nn.functional.scaled_dot_product_attention ."""

    def __init__(
        self,
        embedding_dim: int,
        num_att_heads: int = 1,
        num_kv_heads: int = 1,
        max_position_embeddings: int = 2048,
        scaling_factor: float = 1.0,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        use_bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_heads = num_att_heads
        self.num_kv_heads = num_kv_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.scaling_factor = scaling_factor
        self.attention_dropout = attention_dropout
        self.head_dim = embedding_dim // num_att_heads
        self.use_bias = use_bias
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        super().__init__(
            fl.Distribute(
                fl.Linear(  # Query projection
                    in_features=self.embedding_dim,
                    out_features=self.num_heads * self.head_dim,
                    bias=self.use_bias,
                    device=device,
                    dtype=dtype,
                ),
                fl.Linear(  # Key projection
                    in_features=self.embedding_dim,
                    out_features=self.num_kv_heads * self.head_dim,
                    bias=self.use_bias,
                    device=device,
                    dtype=dtype,
                ),
                fl.Linear(  # Value projection
                    in_features=self.embedding_dim,
                    out_features=self.num_kv_heads * self.head_dim,
                    bias=self.use_bias,
                    device=device,
                    dtype=dtype,
                ),
            ),
            LlamaScaledDotProductAttentionOp(
                embedding_dim=self.embedding_dim,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=self.scaling_factor,
                rope_theta=self.rope_theta,
                device=self.device,
                dtype=self.dtype,
            ),
            fl.Linear(  # Output projection
                in_features=self.embedding_dim,
                out_features=self.embedding_dim,
                bias=True,
                device=device,
                dtype=dtype,
            ),
        )


class LlamaBaseOpAttention(fl.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        scaling_factor: float,
        rope_theta: float,
        attention_dropout: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.rotary_emb = LlamaRotaryEmbedding(head_dim, max_position_embeddings, scaling_factor, rope_theta, device)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.attention_dropout = attention_dropout
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.repeats = num_heads // num_key_value_heads
        self.scale = head_dim**-0.5
        self.dtype = dtype
        self.device = device

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        # §§Not sure about q_len§§
        bsz, _, _ = query.shape
        q_len = self.embedding_dim
        position_ids = torch.arange(0, q_len)

        query_states = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.embedding_dim)

        return attn_output


class LlamaScaledDotProductAttentionOp(fl.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        scaling_factor: float,
        rope_theta: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.rotary_emb = LlamaRotaryEmbedding(head_dim, max_position_embeddings, scaling_factor, rope_theta, device)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.repeats = num_heads // num_key_value_heads
        self.scale = head_dim**-0.5
        self.dtype = dtype
        self.device = device

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        # §§Not sure about q_len§§
        bsz, _, _ = query.shape
        q_len = self.embedding_dim
        position_ids = torch.arange(0, q_len)

        query_states = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.embedding_dim)

        return attn_output
