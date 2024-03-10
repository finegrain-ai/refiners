from torch import device as Device, dtype as DType, nn

from refiners.fluxion.layers.module import WeightedModule


class Conv2d(nn.Conv2d, WeightedModule):
    """2D Convolutional layer.

    This layer wraps [`torch.nn.Conv2d`][torch.nn.Conv2d].

    Receives:
        (Real[Tensor, "batch in_channels in_height in_width"]):

    Returns:
        (Real[Tensor, "batch out_channels out_height out_width"]):

    Example:
        ```py
        conv2d = fl.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        tensor = torch.randn(2, 3, 128, 128)
        output = conv2d(tensor)

        assert output.shape == (2, 32, 128, 128)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = (1, 1),
        padding: int | tuple[int, int] | str = (0, 0),
        groups: int = 1,
        use_bias: bool = True,
        dilation: int | tuple[int, int] = (1, 1),
        padding_mode: str = "zeros",
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(  # type: ignore
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.use_bias = use_bias


class ConvTranspose2d(nn.ConvTranspose2d, WeightedModule):
    """2D Transposed Convolutional layer.

    This layer wraps [`torch.nn.ConvTranspose2d`][torch.nn.ConvTranspose2d].

    Receives:
        (Real[Tensor, "batch in_channels in_height in_width"]):

    Returns:
        (Real[Tensor, "batch out_channels out_height out_width"]):

    Example:
        ```py
        conv2d = fl.ConvTranspose2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        tensor = torch.randn(2, 3, 128, 128)
        output = conv2d(tensor)

        assert output.shape == (2, 32, 128, 128)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        groups: int = 1,
        use_bias: bool = True,
        dilation: int | tuple[int, int] = 1,
        padding_mode: str = "zeros",
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(  # type: ignore
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.use_bias = use_bias
