from torch.nn import Conv2d as _Conv2d, Conv1d as _Conv1d
from torch import device as Device, dtype as DType
from refiners.fluxion.layers.module import WeightedModule


class Conv2d(_Conv2d, WeightedModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] | str = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "zeros",
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(  # type: ignore
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            padding_mode,
            device,
            dtype,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = (padding,) if isinstance(padding, int) else padding
        self.dilation = (dilation,) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.use_bias = use_bias
        self.padding_mode = padding_mode


class Conv1d(_Conv1d, WeightedModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] | str = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "zeros",
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(  # type: ignore
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            padding_mode,
            device,
            dtype,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
