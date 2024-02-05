from typing import Callable

from torch import Size, Tensor, device as Device, dtype as DType
from torch.nn.functional import pad

from refiners.fluxion.layers.basics import Identity
from refiners.fluxion.layers.chain import Chain, Lambda, Parallel, SetContext, UseContext
from refiners.fluxion.layers.conv import Conv2d
from refiners.fluxion.layers.module import Module
from refiners.fluxion.utils import interpolate


class Interpolate(Module):
    """Interpolate layer.

    This layer wraps [`torch.nn.functional.interpolate`][torch.nn.functional.interpolate].
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        x: Tensor,
        shape: Size,
    ) -> Tensor:
        return interpolate(x, shape)


class Downsample(Chain):
    """Downsample layer.

    This layer downsamples the input by the given scale factor.

    Raises:
        RuntimeError: If the context sampling is not set or if the context does not contain a list.
    """

    def __init__(
        self,
        channels: int,
        scale_factor: int,
        padding: int = 0,
        register_shape: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        """Initializes the Downsample layer.

        Args:
            channels: The number of input and output channels.
            scale_factor: The factor by which to downsample the input.
            padding: The amount of zero-padding added to both sides of the input.
            register_shape: If True, registers the input shape in the context.
            device: The device to use for the convolutional layer.
            dtype: The dtype to use for the convolutional layer.
        """
        self.channels = channels
        self.in_channels = channels
        self.out_channels = channels
        self.scale_factor = scale_factor
        self.padding = padding

        super().__init__(
            Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=scale_factor,
                padding=padding,
                device=device,
                dtype=dtype,
            ),
        )

        if padding == 0:
            zero_pad: Callable[[Tensor], Tensor] = lambda x: pad(x, (0, 1, 0, 1))
            self.insert(
                index=0,
                module=Lambda(func=zero_pad),
            )

        if register_shape:
            self.insert(
                index=0,
                module=SetContext(
                    context="sampling",
                    key="shapes",
                    callback=self.register_shape,
                ),
            )

    def register_shape(
        self,
        shapes: list[Size],
        x: Tensor,
    ) -> None:
        shapes.append(x.shape[2:])


class Upsample(Chain):
    """Upsample layer.

    This layer upsamples the input by the given scale factor.

    Raises:
        RuntimeError: If the context sampling is not set or if the context is empty.
    """

    def __init__(
        self,
        channels: int,
        upsample_factor: int | None = None,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        """Initializes the Upsample layer.

        Args:
            channels: The number of input and output channels.
            upsample_factor: The factor by which to upsample the input.
                If None, the input shape is taken from the context.
            device: The device to use for the convolutional layer.
            dtype: The dtype to use for the convolutional layer.
        """
        self.channels = channels
        self.upsample_factor = upsample_factor
        super().__init__(
            Parallel(
                Identity(),
                (
                    Lambda(self._get_static_shape)
                    if upsample_factor is not None
                    else UseContext(context="sampling", key="shapes").compose(lambda x: x.pop())
                ),
            ),
            Interpolate(),
            Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                device=device,
                dtype=dtype,
            ),
        )

    def _get_static_shape(self, x: Tensor) -> Size:
        assert self.upsample_factor is not None
        return Size([size * self.upsample_factor for size in x.shape[2:]])
