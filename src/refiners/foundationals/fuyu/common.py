import math

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn.functional import relu, scaled_dot_product_attention as _scaled_dot_product_attention, softmax

import refiners.fluxion.layers as fl


class SquaredReLU(fl.Activation):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.pow(relu(x), 2)
    
class Softmax(fl.Module):
    def forward(self, x: Tensor) -> Tensor:
        return softmax(x, dim=0)

class CustomReshape(fl.Module):
    """Reshape operation layer.

    This layer reshapes the input tensor to a specific shape (which must be compatible with the original shape).
    See also [torch.reshape][torch.reshape].

    Warning:
        The first dimension and seconde dimension (batch dimension and 
        sequence length) are forcefully preserved.

    Example:
        ```py
        reshape = fl.Reshape(5, 2)

        tensor = torch.randn(2, 6, 10, 1)
        output = reshape(tensor)

        assert output.shape == (2, 6, 5, 2)
        ```
    """

    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return torch.reshape(
            input=x,
            shape=(x.shape[0], x.shape[1], *self.shape),
        )

class Padding(fl.Module):
    def __init__(
        self,
        patch_size: int = 30,
        padding_value: float = 1.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.padding_value = padding_value

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape

        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size

        padded_x = torch.nn.functional.pad(
            input=x,
            pad=(0, pad_w, 0, pad_h),
            mode="constant",
            value=self.padding_value,
        )
        return padded_x
    
def scaled_dot_product_attention(
    query: Float[Tensor, "batch source_sequence_length dim"],
    key: Float[Tensor, "batch target_sequence_length dim"],
    value: Float[Tensor, "batch target_sequence_length dim"],
    attn_mask : Float[Tensor, "batch target_sequence_length source_sequence_length"] = None,
) -> Float[Tensor, "batch source_sequence_length dim"]:
    """Scaled Dot Product Attention.

    Note:
        Optimization depends on which PyTorch backend is used.

    See [[arXiv:1706.03762] Attention Is All You Need (Equation 1)](https://arxiv.org/abs/1706.03762) for more details.
    See also [torch.nn.functional.scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """
    #compute attention in float32
    return _scaled_dot_product_attention(
        query=query.float(),
        key=key.float(),
        value=value.float(),
        attn_mask=attn_mask
    )


def scaled_dot_product_attention_non_optimized(
    query: Float[Tensor, "batch source_sequence_length dim"],
    key: Float[Tensor, "batch target_sequence_length dim"],
    value: Float[Tensor, "batch target_sequence_length dim"],
    attn_mask : Float[Tensor, "batch target_sequence_length source_sequence_length"] = None
) -> Float[Tensor, "batch source_sequence_length dim"]:
    """Non-optimized Scaled Dot Product Attention.

    See [[arXiv:1706.03762] Attention Is All You Need (Equation 1)](https://arxiv.org/abs/1706.03762) for more details.
    """
    if attn_mask is not None:
        # TODO: implement masking => if mask composed of 0 and 1, large negative value instead of 0
        # and 0 instead of 1. Add before softmax
        raise NotImplementedError(
            "attention masking for `scaled_dot_product_attention_non_optimized` is not yet implemented"
        )
    dim = query.shape[-1]
    attention = query @ key.permute(0, 1, 3, 2)
    attention = attention / math.sqrt(dim)
    attention = torch.softmax(input=attention.float(), dim=-1)
    return attention @ value
    
class ScaledDotProductAttentionWithAttnMask(fl.ContextModule):
    """Scaled Dot Product Attention.

    ??? note "See [[arXiv:1706.03762] Attention Is All You Need (Figure 2)](https://arxiv.org/abs/1706.03762) for more details"

        ![](https://ar5iv.labs.arxiv.org/html/1706.03762/assets/Figures/ModalNet-19.png)

    Note:
        This layer simply wraps `scaled_dot_product_attention` inside an `fl.Module`.

    Receives:
        Query (Float[Tensor, "batch num_queries embedding_dim"]):
        Key (Float[Tensor, "batch num_keys embedding_dim"]):
        Value (Float[Tensor, "batch num_values embedding_dim"]):
        AttnMask (Float[Tensor, "batch target_sequence_length source_sequence_length"])

    Returns:
        (Float[Tensor, "batch num_queries embedding_dim"]):

    Example:
        ```py
        attention = fl.ScaledDotProductAttention(num_heads=8)

        query = torch.randn(2, 10, 128)
        key = torch.randn(2, 10, 128)
        value = torch.randn(2, 10, 128)
        output = attention(query, key, value)

        assert output.shape == (2, 10, 128)
        ```
    """

    def __init__(
        self,
        num_heads: int = 1,
        is_optimized: bool = True,
        slice_size: int | None = None,
    ) -> None:
        """Initialize the Scaled Dot Product Attention layer.

        Args:
            num_heads: The number of heads to use.
            is_optimized: Whether to use optimized attention.
            slice_size: The slice size to use for the optimized attention.
        """
        super().__init__()
        self.num_heads = num_heads
        self.is_optimized = is_optimized
        self.slice_size = slice_size
        self.dot_product = (
            scaled_dot_product_attention if self.is_optimized else scaled_dot_product_attention_non_optimized
        )

    def forward(
        self,
        query: Float[Tensor, "batch num_queries embedding_dim"],
        key: Float[Tensor, "batch num_keys embedding_dim"],
        value: Float[Tensor, "batch num_values embedding_dim"],
    ) -> Float[Tensor, "batch num_queries embedding_dim"]:
        
        attn_mask = self.use_context(context_name="attention")["mask"]
        
        if self.slice_size:
            return self._sliced_attention(
                query=query,
                key=key,
                value=value,
                slice_size=self.slice_size,
                attn_mask=attn_mask,
            )
        else:
            return self._process_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask
            )

    def _sliced_attention(
        self,
        query: Float[Tensor, "batch num_queries embedding_dim"],
        key: Float[Tensor, "batch num_keys embedding_dim"],
        value: Float[Tensor, "batch num_values embedding_dim"],
        slice_size: int,
        attn_mask : Float[Tensor, "batch target_sequence_length source_sequence_length"] = None,
    ) -> Float[Tensor, "batch num_queries embedding_dim"]:
        """Compute the scaled dot product attention in slices.

        This is useful when the input tensors are too large to be processed in one go.
        """
        _, num_queries, _ = query.shape
        output = torch.zeros_like(query)
        for start_idx in range(0, num_queries, slice_size):
            end_idx = min(start_idx + slice_size, num_queries)
            output[:, start_idx:end_idx, :] = self._process_attention(
                query=query[:, start_idx:end_idx, :],
                key=key,
                value=value,
                attn_mask=attn_mask[:, start_idx:end_idx, :]
            )
        return output

    def _process_attention(
        self,
        query: Float[Tensor, "batch num_queries embedding_dim"],
        key: Float[Tensor, "batch num_keys embedding_dim"],
        value: Float[Tensor, "batch num_values embedding_dim"],
        attn_mask : Float[Tensor, "batch target_sequence_length source_sequence_length"] = None,
    ) -> Float[Tensor, "batch num_queries embedding_dim"]:
        """Compute the scaled dot product attention.

        Split the input tensors (query, key, value) into multiple heads along the embedding dimension,
        then compute the scaled dot product attention for each head, and finally merge the heads back.
        """
        # the attention is cast in float32
        dtype = query.dtype
        return self._merge_multi_head(
            x=self.dot_product(
                query=self._split_to_multi_head(query),
                key=self._split_to_multi_head(key),
                value=self._split_to_multi_head(value),
                attn_mask=attn_mask,
            ).to(dtype)
        )

    def _split_to_multi_head(
        self,
        x: Float[Tensor, "batch_size sequence_length embedding_dim"],
    ) -> Float[Tensor, "batch_size num_heads sequence_length (embedding_dim//num_heads)"]:
        """Split the input tensor into multiple heads along the embedding dimension.

        See also `merge_multi_head`, which is the inverse operation.
        """
        assert (
            x.ndim == 3
        ), f"Expected input tensor with shape (batch_size sequence_length embedding_dim), got {x.shape}"
        assert (
            x.shape[-1] % self.num_heads == 0
        ), f"Expected embedding_dim (x.shape[-1]={x.shape[-1]}) to be divisible by num_heads ({self.num_heads})"

        return x.reshape(x.shape[0], x.shape[1], self.num_heads, x.shape[-1] // self.num_heads).transpose(1, 2)

    def _merge_multi_head(
        self,
        x: Float[Tensor, "batch_size num_heads sequence_length heads_dim"],
    ) -> Float[Tensor, "batch_size sequence_length heads_dim * num_heads"]:
        """Merge the input tensor from multiple heads along the embedding dimension.

        See also `split_to_multi_head`, which is the inverse operation.
        """
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], self.num_heads * x.shape[-1])
    
