from torch.nn import MaxPool1d as _MaxPool1d, MaxPool2d as _MaxPool2d

from refiners.fluxion.layers.module import Module


class MaxPool1d(_MaxPool1d, Module):
    """MaxPool1d layer.

    This layer wraps [`torch.nn.MaxPool1d`][torch.nn.MaxPool1d].

    Receives:
        (Float[Tensor, "batch channels in_length"]):

    Returns:
        (Float[Tensor, "batch channels out_length"]):
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        """Initializes the MaxPool1d layer.

        Args:
            kernel_size: The size of the sliding window.
            stride: The stride of the sliding window.
            padding: The amount of zero-padding added to both sides of the input.
            dilation: The spacing between kernel elements.
            return_indices: If True, returns the max indices along with the outputs.
            ceil_mode: If True, uses ceil instead of floor to compute the output shape.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )


class MaxPool2d(_MaxPool2d, Module):
    """MaxPool2d layer.

    This layer wraps [`torch.nn.MaxPool2d`][torch.nn.MaxPool2d].

    Receives:
        (Float[Tensor, "batch channels in_height in_width"]):

    Returns:
        (Float[Tensor, "batch channels out_height out_width"]):
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = (0, 0),
        dilation: int | tuple[int, int] = (1, 1),
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        """Initializes the MaxPool2d layer.

        Args:
            kernel_size: The size of the sliding window.
            stride: The stride of the sliding window.
            padding: The amount of zero-padding added to both sides of the input.
            dilation: The spacing between kernel elements.
            return_indices: If True, returns the max indices along with the outputs.
            ceil_mode: If True, uses ceil instead of floor to compute the output shape.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )
