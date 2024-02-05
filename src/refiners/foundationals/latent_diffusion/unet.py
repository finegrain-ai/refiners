from torch import Tensor, device as Device, dtype as DType

import refiners.fluxion.layers as fl


class ResidualBlock(fl.Sum):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 32,
        eps: float = 1e-5,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        if in_channels % num_groups != 0 or out_channels % num_groups != 0:
            raise ValueError("Number of input and output channels must be divisible by num_groups.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups
        self.eps = eps
        shortcut = (
            fl.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, device=device, dtype=dtype)
            if in_channels != out_channels
            else fl.Identity()
        )
        super().__init__(
            fl.Chain(
                fl.GroupNorm(channels=in_channels, num_groups=num_groups, eps=eps, device=device, dtype=dtype),
                fl.SiLU(),
                fl.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
                fl.GroupNorm(channels=out_channels, num_groups=num_groups, eps=eps, device=device, dtype=dtype),
                fl.SiLU(),
                fl.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
            ),
            shortcut,
        )


class ResidualAccumulator(fl.Passthrough):
    def __init__(self, n: int) -> None:
        self.n = n

        super().__init__(
            fl.Residual(
                fl.UseContext(context="unet", key="residuals").compose(func=lambda residuals: residuals[self.n])
            ),
            fl.SetContext(context="unet", key="residuals", callback=self.update),
        )

    def update(self, residuals: list[Tensor | float], x: Tensor) -> None:
        residuals[self.n] = x


class ResidualConcatenator(fl.Chain):
    def __init__(self, n: int) -> None:
        self.n = n

        super().__init__(
            fl.Concatenate(
                fl.Identity(),
                fl.UseContext(context="unet", key="residuals").compose(lambda residuals: residuals[self.n]),
                dim=1,
            ),
        )
