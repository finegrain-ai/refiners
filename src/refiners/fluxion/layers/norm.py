from torch import ones, zeros, Tensor, sqrt, device as Device, dtype as DType
from torch.nn import GroupNorm as _GroupNorm, Parameter, LayerNorm as _LayerNorm
from jaxtyping import Float
from refiners.fluxion.layers.module import WeightedModule


class LayerNorm(_LayerNorm, WeightedModule):
    def __init__(
        self,
        normalized_shape: int | list[int],
        eps: float = 0.00001,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(  # type: ignore
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=True,  # otherwise not a WeightedModule
            device=device,
            dtype=dtype,
        )


class GroupNorm(_GroupNorm, WeightedModule):
    def __init__(
        self,
        channels: int,
        num_groups: int,
        eps: float = 1e-5,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(  # type: ignore
            num_groups=num_groups,
            num_channels=channels,
            eps=eps,
            affine=True,  # otherwise not a WeightedModule
            device=device,
            dtype=dtype,
        )
        self.channels = channels
        self.num_groups = num_groups
        self.eps = eps


class LayerNorm2d(WeightedModule):
    """
    2D Layer Normalization module.

    Parameters:
        channels (int): Number of channels in the input tensor.
        eps (float, optional): A small constant for numerical stability. Default: 1e-6.
    """

    def __init__(
        self,
        channels: int,
        eps: float = 1e-6,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        self.weight = Parameter(ones(channels, device=device, dtype=dtype))
        self.bias = Parameter(zeros(channels, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Float[Tensor, "batch channels height width"]) -> Float[Tensor, "batch channels height width"]:
        x_mean = x.mean(1, keepdim=True)
        x_var = (x - x_mean).pow(2).mean(1, keepdim=True)
        x_norm = (x - x_mean) / sqrt(x_var + self.eps)
        x_out = self.weight.unsqueeze(-1).unsqueeze(-1) * x_norm + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x_out
