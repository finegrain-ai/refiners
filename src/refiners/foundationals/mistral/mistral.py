import torch

import refiners.fluxion.layers as fl
from refiners.fluxion.layers.activations import Activation

from .attention import MistralAttention


class MistralRMSNorm(fl.WeightedModule):
    """
    Implements RMS Normalization layer.

    Args:
        embedding_dim: The dimension of the input data.
        eps: A small value to prevent division by zero.
        device: The PyTorch device to use.
        dtype: The PyTorch data type to use.
    """

    def __init__(
        self,
        embedding_dim: int,
        eps: float,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(embedding_dim, dtype=dtype, device=device))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(fl.Chain):
    """
    Implements Mistral feedforward layer.

    Args:
        embedding_dim: The dimension of the input data.
        feedforward_dim: Internal feedfoward dimension.
        activation: Activation function.
        device: The PyTorch device to use.
        dtype: The PyTorch data type to use.
    """

    def __init__(
        self,
        embedding_dim: int,
        feedforward_dim: int,
        activation: Activation = fl.SiLU,  # type: ignore
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            fl.Parallel(
                fl.Chain(
                    fl.Linear(
                        in_features=embedding_dim,
                        out_features=feedforward_dim,
                        bias=False,
                        device=device,
                        dtype=dtype,
                    ),
                    activation(),
                ),
                fl.Linear(
                    in_features=embedding_dim,
                    out_features=feedforward_dim,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
            ),
            fl.Lambda(lambda x, y: x * y),  # type: ignore
            fl.Linear(
                in_features=feedforward_dim,
                out_features=embedding_dim,
                bias=False,
                device=device,
                dtype=dtype,
            ),
        )


class MistralTranformerLayer(fl.Chain):
    """Apply a multi-head self-attention mechanism to the input tensor."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_kv_heads: int,
        sliding_window: int,
        rope_theta: float,
        max_position_embeddings: int,
        feedforward_dim: int,
        norm_eps: float,
        activation: Activation = fl.SiLU,  # type: ignore
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            fl.Residual(
                MistralRMSNorm(embedding_dim=embedding_dim, eps=norm_eps, device=device, dtype=dtype),
                MistralAttention(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    sliding_window=sliding_window,
                    rope_theta=rope_theta,
                    max_position_embeddings=max_position_embeddings,
                    device=device,
                    dtype=dtype,
                ),
            ),
            fl.Residual(
                MistralRMSNorm(embedding_dim=embedding_dim, eps=norm_eps, device=device, dtype=dtype),
                FeedForward(
                    embedding_dim=embedding_dim,
                    feedforward_dim=feedforward_dim,
                    activation=activation,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )


class MistralTransformer(fl.Chain):
    """Alias for a Chain of TransformerBlock."""


class Mistral(fl.Chain):
    """
    Mistral Model

    See [[arXiv:2310.06825] Mistral 7B](https://arxiv.org/pdf/2310.06825.pdf)
    for more details.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        vocab_size: int,
        sliding_window: int,
        feedforward_dim: int,
        norm_eps: float,
        rope_theta: float,
        max_position_embeddings: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            fl.Embedding(vocab_size, embedding_dim, dtype=dtype, device=device),
            MistralTransformer(
                MistralTranformerLayer(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    sliding_window=sliding_window,
                    rope_theta=rope_theta,
                    max_position_embeddings=max_position_embeddings,
                    feedforward_dim=feedforward_dim,
                    norm_eps=norm_eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ),
            MistralRMSNorm(embedding_dim=embedding_dim, eps=norm_eps, device=device, dtype=dtype),
            fl.Linear(
                in_features=embedding_dim,
                out_features=vocab_size,
                bias=False,
                device=device,
                dtype=dtype,
            ),
        )

class Mistral7b(Mistral):
    """
    Mistral model with 7b parameters

    Attributes:
        embedding_dim (int): 4_096
        num_layers (int): 32
        num_heads (int): 32
        num_kv_heads (int): 8
        vocab_size (int): 32_768
        sliding_window: int = 4096
        feedforward_dim (int): 14_336
        norm_eps (float): 1e-5
        rope_theta: float = 10000.0
        max_position_embeddings: int = 32768
    """

    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            embedding_dim=4_096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            vocab_size=32_768,
            sliding_window=4_096,
            feedforward_dim=14_336,
            norm_eps=1e-5,
            rope_theta=10_000.0,
            max_position_embeddings=32_768,
            device=device,
            dtype=dtype,
        )
