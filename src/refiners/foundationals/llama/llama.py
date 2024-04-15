import torch
from torch.nn import Parameter as TorchParameter

import refiners.fluxion.layers as fl
from refiners.fluxion.layers.activations import Activation, SiLU
from refiners.fluxion.layers.module import WeightedModule


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
