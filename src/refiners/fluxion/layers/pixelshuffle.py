from torch.nn import PixelUnshuffle as _PixelUnshuffle

from refiners.fluxion.layers.module import Module


class PixelUnshuffle(_PixelUnshuffle, Module):
    """Pixel Unshuffle layer.

    This layer wraps [`torch.nn.PixelUnshuffle`][torch.nn.PixelUnshuffle].

    Receives:
        (Float[Tensor, "batch in_channels in_height in_width"]):

    Returns:
        (Float[Tensor, "batch out_channels out_height out_width"]):
    """

    def __init__(self, downscale_factor: int):
        _PixelUnshuffle.__init__(self, downscale_factor=downscale_factor)
