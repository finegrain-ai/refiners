from refiners.fluxion.layers.module import Module
from torch.nn import PixelUnshuffle as _PixelUnshuffle


class PixelUnshuffle(_PixelUnshuffle, Module):
    def __init__(self, downscale_factor: int):
        _PixelUnshuffle.__init__(self, downscale_factor=downscale_factor)
