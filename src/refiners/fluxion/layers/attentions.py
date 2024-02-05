import math

import torch
from jaxtyping import Float
from torch import Tensor, device as Device, dtype as DType
from torch.nn.functional import scaled_dot_product_attention as _scaled_dot_product_attention

from refiners.fluxion.context import Contexts
from refiners.fluxion.layers.basics import Identity
from refiners.fluxion.layers.chain import Chain, Distribute, Lambda, Parallel
from refiners.fluxion.layers.linear import Linear
from refiners.fluxion.layers.module import Module


def scaled_dot_product_attention(
    query: Float[Tensor, "batch source_sequence_length dim"],
    key: Float[Tensor, "batch target_sequence_length dim"],
    value: Float[Tensor, "batch target_sequence_length dim"],
    is_causal: bool = False,
) -> Float[Tensor, "batch source_sequence_length dim"]:
    """Scaled Dot Product Attention.

    Note:
        Optimization depends on which PyTorch backend is used.

    See [[arXiv:1706.03762] Attention Is All You Need (Equation 1)](https://arxiv.org/abs/1706.03762) for more details.
    See also [torch.nn.functional.scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
    """
    return _scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        is_causal=is_causal,
    )


def scaled_dot_product_attention_non_optimized(
    query: Float[Tensor, "batch source_sequence_length dim"],
    key: Float[Tensor, "batch target_sequence_length dim"],
    value: Float[Tensor, "batch target_sequence_length dim"],
    is_causal: bool = False,
) -> Float[Tensor, "batch source_sequence_length dim"]:
    """Non-optimized Scaled Dot Product Attention.

    See [[arXiv:1706.03762] Attention Is All You Need (Equation 1)](https://arxiv.org/abs/1706.03762) for more details.
    """
    if is_causal:
        # TODO: implement causal attention
        raise NotImplementedError(
            "Causal attention for `scaled_dot_product_attention_non_optimized` is not yet implemented"
        )

    dim = query.shape[-1]
    attention = query @ key.permute(0, 1, 3, 2)
    attention = attention / math.sqrt(dim)
    attention = torch.softmax(input=attention, dim=-1)
    return attention @ value


class ScaledDotProductAttention(Module):
    """Scaled Dot Product Attention.

    ??? note "See [[arXiv:1706.03762] Attention Is All You Need (Figure 2)](https://arxiv.org/abs/1706.03762) for more details"

        ![](https://ar5iv.labs.arxiv.org/html/1706.03762/assets/Figures/ModalNet-19.png)

    Note:
        This layer simply wraps `scaled_dot_product_attention` inside an `fl.Module`.

    Receives:
        Query (Float[Tensor, "batch num_queries embedding_dim"]):
        Key (Float[Tensor, "batch num_keys embedding_dim"]):
        Value (Float[Tensor, "batch num_values embedding_dim"]):

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
        is_causal: bool = False,
        is_optimized: bool = True,
        slice_size: int | None = None,
    ) -> None:
        """Initialize the Scaled Dot Product Attention layer.

        Args:
            num_heads: The number of heads to use.
            is_causal: Whether to use causal attention.
            is_optimized: Whether to use optimized attention.
            slice_size: The slice size to use for the optimized attention.
        """
        super().__init__()
        self.num_heads = num_heads
        self.is_causal = is_causal
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
        if self.slice_size:
            return self._sliced_attention(
                query=query,
                key=key,
                value=value,
                slice_size=self.slice_size,
            )
        else:
            return self._process_attention(
                query=query,
                key=key,
                value=value,
            )

    def _sliced_attention(
        self,
        query: Float[Tensor, "batch num_queries embedding_dim"],
        key: Float[Tensor, "batch num_keys embedding_dim"],
        value: Float[Tensor, "batch num_values embedding_dim"],
        slice_size: int,
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
            )
        return output

    def _process_attention(
        self,
        query: Float[Tensor, "batch num_queries embedding_dim"],
        key: Float[Tensor, "batch num_keys embedding_dim"],
        value: Float[Tensor, "batch num_values embedding_dim"],
    ) -> Float[Tensor, "batch num_queries embedding_dim"]:
        """Compute the scaled dot product attention.

        Split the input tensors (query, key, value) into multiple heads along the embedding dimension,
        then compute the scaled dot product attention for each head, and finally merge the heads back.
        """
        return self._merge_multi_head(
            x=self.dot_product(
                query=self._split_to_multi_head(query),
                key=self._split_to_multi_head(key),
                value=self._split_to_multi_head(value),
                is_causal=self.is_causal,
            )
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


class Attention(Chain):
    """Multi-Head Attention layer.

    ??? note "See [[arXiv:1706.03762] Attention Is All You Need (Figure 2)](https://arxiv.org/abs/1706.03762) for more details"

        ![](https://ar5iv.labs.arxiv.org/html/1706.03762/assets/Figures/ModalNet-20.png)

    Note: This layer simply chains
        - a [`Distribute`][refiners.fluxion.layers.chain.Distribute] layer,
            containing 3 [`Linear`][refiners.fluxion.layers.linear.Linear] layers,
            which transforms the 3 inputs into Query, Key and Value
        - a [`ScaledDotProductAttention`][refiners.fluxion.layers.attentions.ScaledDotProductAttention] layer
        - a [`Linear`][refiners.fluxion.layers.linear.Linear] layer,
            which projects the output of the
            [`ScaledDotProductAttention`][refiners.fluxion.layers.attentions.ScaledDotProductAttention] layer

    Receives:
        Query (Float[Tensor, "batch sequence_length embedding_dim"]):
        Key (Float[Tensor, "batch sequence_length embedding_dim"]):
        Value (Float[Tensor, "batch sequence_length embedding_dim"]):

    Returns:
        (Float[Tensor, "batch sequence_length embedding_dim"]):

    Example:
        ```py
        attention = fl.Attention(num_heads=8, embedding_dim=128)

        tensor = torch.randn(2, 10, 128)
        output = attention(tensor, tensor, tensor)

        assert output.shape == (2, 10, 128)
        ```
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 1,
        key_embedding_dim: int | None = None,
        value_embedding_dim: int | None = None,
        inner_dim: int | None = None,
        use_bias: bool = True,
        is_causal: bool = False,
        is_optimized: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize the Attention layer.

        Args:
            embedding_dim: The embedding dimension of the input and output tensors.
            num_heads: The number of heads to use.
            key_embedding_dim: The embedding dimension of the key tensor.
            value_embedding_dim: The embedding dimension of the value tensor.
            inner_dim: The inner dimension of the linear layers.
            use_bias: Whether to use bias in the linear layers.
            is_causal: Whether to use causal attention.
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
        self.key_embedding_dim = key_embedding_dim or embedding_dim
        self.value_embedding_dim = value_embedding_dim or embedding_dim
        self.inner_dim = inner_dim or embedding_dim
        self.use_bias = use_bias
        self.is_causal = is_causal
        self.is_optimized = is_optimized

        super().__init__(
            Distribute(
                Linear(  # Query projection
                    in_features=self.embedding_dim,
                    out_features=self.inner_dim,
                    bias=self.use_bias,
                    device=device,
                    dtype=dtype,
                ),
                Linear(  # Key projection
                    in_features=self.key_embedding_dim,
                    out_features=self.inner_dim,
                    bias=self.use_bias,
                    device=device,
                    dtype=dtype,
                ),
                Linear(  # Value projection
                    in_features=self.value_embedding_dim,
                    out_features=self.inner_dim,
                    bias=self.use_bias,
                    device=device,
                    dtype=dtype,
                ),
            ),
            ScaledDotProductAttention(
                num_heads=num_heads,
                is_causal=is_causal,
                is_optimized=is_optimized,
            ),
            Linear(  # Output projection
                in_features=self.inner_dim,
                out_features=self.embedding_dim,
                bias=True,
                device=device,
                dtype=dtype,
            ),
        )


class SelfAttention(Attention):
    """Multi-Head Self-Attention layer.

    Note: This layer simply chains
        - a [`Parallel`][refiners.fluxion.layers.chain.Parallel] layer,
        which duplicates the input Tensor
        (for each Linear layer in the `Attention` layer)
        - an [`Attention`][refiners.fluxion.layers.attentions.Attention] layer

    Receives:
        (Float[Tensor, "batch sequence_length embedding_dim"]):

    Returns:
        (Float[Tensor, "batch sequence_length embedding_dim"]):

    Example:
        ```py
        self_attention = fl.SelfAttention(num_heads=8, embedding_dim=128)

        tensor = torch.randn(2, 10, 128)
        output = self_attention(tensor)

        assert output.shape == (2, 10, 128)
        ```
    """

    def __init__(
        self,
        embedding_dim: int,
        inner_dim: int | None = None,
        num_heads: int = 1,
        use_bias: bool = True,
        is_causal: bool = False,
        is_optimized: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize the Self-Attention layer.

        Args:
            embedding_dim: The embedding dimension of the input and output tensors.
            inner_dim: The inner dimension of the linear layers.
            num_heads: The number of heads to use.
            use_bias: Whether to use bias in the linear layers.
            is_causal: Whether to use causal attention.
            is_optimized: Whether to use optimized attention.
            device: The device to use.
            dtype: The dtype to use.
        """
        super().__init__(
            embedding_dim=embedding_dim,
            inner_dim=inner_dim,
            num_heads=num_heads,
            use_bias=use_bias,
            is_causal=is_causal,
            is_optimized=is_optimized,
            device=device,
            dtype=dtype,
        )
        self.insert(
            index=0,
            module=Parallel(
                Identity(),  # Query projection's input
                Identity(),  # Key projection's input
                Identity(),  # Value projection's input
            ),
        )


class SelfAttention2d(SelfAttention):
    """Multi-Head 2D Self-Attention layer.

    Note: This Module simply chains
        - a [`Lambda`][refiners.fluxion.layers.chain.Lambda] layer,
            which transforms the input Tensor into a sequence
        - a [`SelfAttention`][refiners.fluxion.layers.attentions.SelfAttention] layer
        - a [`Lambda`][refiners.fluxion.layers.chain.Lambda] layer,
            which transforms the output sequence into a 2D Tensor

    Receives:
        (Float[Tensor, "batch channels height width"]):

    Returns:
        (Float[Tensor, "batch channels height width"]):

    Example:
        ```py
        self_attention = fl.SelfAttention2d(channels=128, num_heads=8)

        tensor = torch.randn(2, 128, 64, 64)
        output = self_attention(tensor)

        assert output.shape == (2, 128, 64, 64)
        ```
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        use_bias: bool = True,
        is_causal: bool = False,
        is_optimized: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize the 2D Self-Attention layer.

        Args:
            channels: The number of channels of the input and output tensors.
            num_heads: The number of heads to use.
            use_bias: Whether to use bias in the linear layers.
            is_causal: Whether to use causal attention.
            is_optimized: Whether to use optimized attention.
            device: The device to use.
            dtype: The dtype to use.
        """
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"
        self.channels = channels

        super().__init__(
            embedding_dim=channels,
            num_heads=num_heads,
            use_bias=use_bias,
            is_causal=is_causal,
            is_optimized=is_optimized,
            device=device,
            dtype=dtype,
        )

        self.insert(0, Lambda(self._tensor_2d_to_sequence))
        self.append(Lambda(self._sequence_to_tensor_2d))

    def init_context(self) -> Contexts:
        return {
            "reshape": {
                "height": None,
                "width": None,
            }
        }

    def _tensor_2d_to_sequence(
        self,
        x: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, "batch height*width channels"]:
        """Transform a 2D Tensor into a sequence.

        The height and width of the input Tensor are stored in a `"reshape"` context,
        so that the output Tensor can be transformed back into a 2D Tensor in the `sequence_to_tensor_2d` method.
        """
        height, width = x.shape[-2:]
        self.set_context(
            context="reshape",
            value={
                "height": height,
                "width": width,
            },
        )
        return x.reshape(x.shape[0], x.shape[1], height * width).transpose(1, 2)

    def _sequence_to_tensor_2d(
        self,
        x: Float[Tensor, "batch sequence_length channels"],
    ) -> Float[Tensor, "batch channels height width"]:
        """Transform a sequence into a 2D Tensor.

        The height and width of the output Tensor are retrieved from the `"reshape"` context,
        which was set in the `tensor_2d_to_sequence` method.
        """
        height, width = self.use_context("reshape").values()
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], height, width)
