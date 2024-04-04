import torch
from torch.nn import Parameter as TorchParameter

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
