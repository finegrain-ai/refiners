from torch.nn import ReflectionPad2d as _ReflectionPad2d

from refiners.fluxion.layers.module import Module


class ReflectionPad2d(_ReflectionPad2d, Module):
    """Reflection padding layer.

    This layer wraps [`torch.nn.ReflectionPad2d`][torch.nn.ReflectionPad2d].

    Receives:
        (Float[Tensor, "batch channels in_height in_width"]):

    Returns:
        (Float[Tensor, "batch channels out_height out_width"]):
    """

    def __init__(self, padding: int) -> None:
        super().__init__(padding=padding)
