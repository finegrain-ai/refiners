from jaxtyping import Float
from torch import Tensor, device as Device, dtype as DType, ones, sqrt, zeros
from torch.nn import (
    GroupNorm as _GroupNorm,
    InstanceNorm2d as _InstanceNorm2d,
    LayerNorm as _LayerNorm,
    Parameter as TorchParameter,
)

from refiners.fluxion.layers.module import Module, WeightedModule


class LayerNorm(_LayerNorm, WeightedModule):
    """Layer Normalization layer.

    This layer wraps [`torch.nn.LayerNorm`][torch.nn.LayerNorm].

    Receives:
        (Float[Tensor, "batch *normalized_shape"]):

    Returns:
        (Float[Tensor, "batch *normalized_shape"]):

    Example:
        ```py
        layernorm = fl.LayerNorm(normalized_shape=128)

        tensor = torch.randn(2, 128)
        output = layernorm(tensor)

        assert output.shape == (2, 128)
        ```
    """

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
    """Group Normalization layer.

    This layer wraps [`torch.nn.GroupNorm`][torch.nn.GroupNorm].

    Receives:
        (Float[Tensor, "batch channels *normalized_shape"]):

    Returns:
        (Float[Tensor, "batch channels *normalized_shape"]):

    Example:
        ```py
        groupnorm = fl.GroupNorm(channels=128, num_groups=8)

        tensor = torch.randn(2, 128, 8)
        output = groupnorm(tensor)

        assert output.shape == (2, 128, 8)
        ```
    """

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
    """2D Layer Normalization layer.

    This layer applies Layer Normalization along the 2nd dimension of a 4D tensor.

    Receives:
        (Float[Tensor, "batch channels height width"]):

    Returns:
        (Float[Tensor, "batch channels height width"]):
    """

    def __init__(
        self,
        channels: int,
        eps: float = 1e-6,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        self.weight = TorchParameter(ones(channels, device=device, dtype=dtype))
        self.bias = TorchParameter(zeros(channels, device=device, dtype=dtype))
        self.eps = eps

    def forward(
        self,
        x: Float[Tensor, "batch channels height width"],
    ) -> Float[Tensor, "batch channels height width"]:
        x_mean = x.mean(1, keepdim=True)
        x_var = (x - x_mean).pow(2).mean(1, keepdim=True)
        x_norm = (x - x_mean) / sqrt(x_var + self.eps)
        x_out = self.weight.unsqueeze(-1).unsqueeze(-1) * x_norm + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x_out


class InstanceNorm2d(_InstanceNorm2d, Module):
    """Instance Normalization layer.

    This layer wraps [`torch.nn.InstanceNorm2d`][torch.nn.InstanceNorm2d].

    Receives:
        (Float[Tensor, "batch channels height width"]):

    Returns:
        (Float[Tensor, "batch channels height width"]):
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(  # type: ignore
            num_features=num_features,
            eps=eps,
            device=device,
            dtype=dtype,
        )
