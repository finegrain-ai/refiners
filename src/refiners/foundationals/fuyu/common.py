import torch
from torch import Tensor, device as Device, dtype as DType
from torch.nn.functional import relu, softmax

import refiners.fluxion.layers as fl


class SquaredReLU(fl.Activation):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.pow(relu(x), 2)
    
class Softmax(fl.Module):
    def forward(self, x: Tensor) -> Tensor:
        return softmax(x, dim=0)

class CustomReshape(fl.Module):
    """Reshape operation layer.

    This layer reshapes the input tensor to a specific shape (which must be compatible with the original shape).
    See also [torch.reshape][torch.reshape].

    Warning:
        The first dimension and seconde dimension (batch dimension and 
        sequence lenght) are forcefully preserved.

    Example:
        ```py
        reshape = fl.Reshape(5, 2)

        tensor = torch.randn(2, 6, 10, 1)
        output = reshape(tensor)

        assert output.shape == (2, 6, 5, 2)
        ```
    """

    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return torch.reshape(
            input=x,
            shape=(x.shape[0], x.shape[1], *self.shape),
        )

class Padding(fl.Module):
    def __init__(
        self,
        patch_size: int = 30,
        padding_value: int = 1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.padding_value = padding_value

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[2:]

        pad_h = h % self.patch_size
        pad_w = w % self.patch_size

        padded_x = torch.nn.functional.pad(
            input=x,
            pad=(0, pad_h, 0, pad_w),
            mode="constant",
            value=self.padding_value,
        )
        return padded_x