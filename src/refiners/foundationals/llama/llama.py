import torch
from torch.nn import Parameter as TorchParameter

import refiners.fluxion.layers as fl
from refiners.fluxion.layers.activations import Activation, SiLU
from refiners.fluxion.layers.module import WeightedModule

from .attention import LLamaSdpaAttention


class LlamaRMSNorm(WeightedModule):
    """LlamaRMSNorm is equivalent to T5LayerNorm"""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = TorchParameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class LLamaMLP(fl.Chain):
    """
    Implements Llama MLP block.
    Args:
        dim: The dimension of the input data.
        feedforward_dim: Internal feedfoward dimension.
        activation: Activation function.
        device: The PyTorch device to use.
        dtype: The PyTorch data type to use.
    """

    def __init__(
        self,
        dim: int,
        feedforward_dim: int,
        activation: Activation = SiLU,  # type: ignore
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.embedding_dim = dim
        self.hidden_dim = feedforward_dim

        super().__init__(
            fl.Parallel(
                fl.Chain(
                    fl.Linear(
                        in_features=dim,
                        out_features=feedforward_dim,
                        bias=False,
                        device=device,
                        dtype=dtype,
                    ),
                    activation(),
                ),
                fl.Linear(
                    in_features=dim,
                    out_features=feedforward_dim,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
            ),
            fl.Lambda(lambda x, y: x * y),  # type: ignore
            fl.Linear(
                in_features=feedforward_dim,
                out_features=dim,
                bias=False,
                device=device,
                dtype=dtype,
            ),
        )


class LLamaDecoderLayer(fl.Chain):
    """LLama Transformer main layer block"""

    def __init__(
        self,
        dim: int,
        n_att_heads: int,
        n_kv_heads: int,
        max_position_embeddings: int,
        feedforward_dim: int,
        norm_eps: float = 1e-6,
        activation: Activation = SiLU,  # type: ignore
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            fl.Residual(
                LlamaRMSNorm(hidden_size=dim, eps=norm_eps, device=device, dtype=dtype),
                LLamaSdpaAttention(
                    embedding_dim=dim,
                    num_att_heads=n_att_heads,
                    num_kv_heads=n_kv_heads,
                    max_position_embeddings=max_position_embeddings,
                    device=device,
                    dtype=dtype,
                ),
            ),
            fl.Residual(
                LlamaRMSNorm(hidden_size=dim, eps=norm_eps, device=device, dtype=dtype),
                LLamaMLP(dim=dim, feedforward_dim=feedforward_dim, activation=activation, device=device, dtype=dtype),
            ),
        )
