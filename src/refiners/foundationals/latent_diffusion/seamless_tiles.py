"""Code for enabling single direction circular padding on Conv2d layers."""

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters import Adapter
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet

TileModeType = Literal["", "x", "y", "xy"]
PaddingModeType = Literal["circular", "constant"]


class SeamlessConv2dWrapper(fl.Module):
    def __init__(self, target: fl.Conv2d, tile_adapter: "TilingAdapter") -> None:
        super().__init__()
        self.target = target
        self.tile_adapter = tile_adapter

    def _conv_forward(self, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        tile_mode = self.tile_adapter.tile_mode
        padding_mode_x = "circular" if "x" in tile_mode else "constant"
        padding_mode_y = "circular" if "y" in tile_mode else "constant"

        if padding_mode_y == padding_mode_x:
            original_padding_mode = self.target.padding_mode
            try:
                self.target.padding_mode = padding_mode_x
                return self.target._conv_forward(input, weight, bias)  # type: ignore[protected-access]
            finally:
                self.target.padding_mode = original_padding_mode

        rprt = self.target._reversed_padding_repeated_twice  # type: ignore[protected-access]
        padding_x = (rprt[0], rprt[1], 0, 0)
        padding_y = (0, 0, rprt[2], rprt[3])

        w1 = F.pad(input, padding_x, mode=padding_mode_x)
        del input

        w2 = F.pad(w1, padding_y, mode=padding_mode_y)
        del w1

        return F.conv2d(w2, weight, bias, self.target.stride, (0, 0), self.target.dilation, self.target.groups)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(input, self.target.weight, self.target.bias)


class TilingConv2dAdapter(fl.Chain, Adapter[fl.Conv2d]):
    def __init__(self, target: fl.Conv2d, tile_adapter: "TilingAdapter") -> None:
        with self.setup_adapter(target):
            super().__init__(
                SeamlessConv2dWrapper(target, tile_adapter=tile_adapter),
            )


def find_parent(model, child: nn.Module) -> fl.Chain | None:
    for module in model.modules():
        if child in module.children():
            return module  # type: ignore[no-any-return]
    return None


class TilingAdapter(fl.Passthrough, Adapter[SD1UNet]):
    def __init__(self, target: SD1UNet, tile_mode: TileModeType = "") -> None:
        self.tile_mode = tile_mode
        with self.setup_adapter(target):
            super().__init__(target)

    def inject(self, parent: fl.Chain | None = None) -> "TilingAdapter":
        super().inject(parent)
        for module in self.target.modules():
            if isinstance(module, fl.Conv2d):
                cparent = find_parent(self.target, module)
                TilingConv2dAdapter(module, tile_adapter=self).inject(cparent)
        return self

    def eject(self):
        for module in self.target.modules():
            if isinstance(module, TilingConv2dAdapter):
                module.eject()
        super().eject()

    def set_tile_mode(self, tile_mode: TileModeType) -> None:
        self.tile_mode = tile_mode
