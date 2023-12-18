import math
from typing import Any, Generic, TypeVar

import torch.functional as F
from torch import Tensor

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.adapter import Adapter
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import  SD1UNet
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet

T = TypeVar("T", bound="SD1UNet | SDXLUNet")
TSDScaleCrafterAdapter = TypeVar("TSDScaleCrafterAdapter", bound="SDScaleCrafterAdapter[Any]")  # Self (see PEP 673)

class ReDilatedConv(fl.Module):
    def __init__(self, conv: fl.Conv2d, dilation_factor: float = 1.0, kernel_inflated: bool = False, progressive: bool = False, inflate_timestep: int = 0, dilation_timestep: int = 700, interpolation_mode: str = "bilinear"):
        self.conv = conv
        self.dilation_factor = dilation_factor
        self.interpolation_mode = interpolation_mode
        self.progressive = progressive
        self.kernel_inflated = kernel_inflated
        self.inflate_timestep = inflate_timestep
        self.dilation_timestep = dilation_timestep
    def forward(self, x: Tensor, timestep: Tensor, activate: bool = True) -> Tensor:
        # activate mainly on certain timesteps in diffusion process
        if not activate:
            return self.conv(x)
        dilation_factor = self.dilation_factor
        inflated_timestep = timestep > self.inflate_timestep
        if self.progressive:
            dilation_factor = max(math.ceil(dilation_factor * ((self.dilation_timestep - timestep) / self.dilation_timestep)), 2)
        if inflated_timestep and self.kernel_inflated:
             dilation_factor /= 2
        dilation = math.ceil(dilation_factor)
        scale = dilation/dilation_factor
        original_dilation, original_padding = self.conv.module.dilation, self.conv.module.padding
        original_kernel_size = self.conv.module.weight.shape[-1]
        inflation_kernel_size = (original_kernel_size - 1) // 2
        width, height = x.shape[-2], x.shape[-1]
        strides = (self.conv.module.stride[0], self.conv.module.stride[1])
        self.module.dilation, self.module.padding = dilation, (
            dilation * inflation_kernel_size, dilation * inflation_kernel_size
        )
        original_size = (int(width/strides[0]), int(height/strides[1]))
        intermediate_size = (round(width*scale), round(height*scale))
        x = F.interpolate(x, size=intermediate_size, mode=self.interpolation_mode)
        x = self.module._conv_forward(x, self.conv.module.weight, self.conv.module.bias)
        self.module.dilation, self.module.padding = original_dilation, original_padding
        x = F.interpolate(x, size=original_size, mode=self.interpolation_mode)
        return x


class SDScaleCrafterAdapter(Generic[T], fl.Chain, Adapter[T]):
    def __init__(self, target: T, dilation_setting: dict[str, float], inflate_settings: dict[str, str], inflate_timestep: int = 0, dilation_timestep: int = 700, progressive: bool =False) -> None:
        self.dilation_setting = dilation_setting
        self.inflate_settings = inflate_settings
        self.inflate_timestep = inflate_timestep
        self.dilation_timestep = dilation_timestep
        self.progressive = progressive
        with self.setup_adapter(target):
            super().__init__(target)

    def inject(self: TSDScaleCrafterAdapter, parent: fl.Chain | None = None) -> TSDScaleCrafterAdapter:
         for name, module in self.target.named_modules():
             if name in self.dilation_setting.keys():
                dilation_factor = self.dilate_settings[name]
                kernel_inflated = name in self.inflate_settings
                wrapped_module = ReDilatedConv(module, dilation_factor, kernel_inflated, self.progressive, self.inflate_timestep, self.dilation_timestep)
                # TODO: Replace module with wrapped module
        return super().inject(parent)

    def eject(self) -> None:
        super().eject()
