import math
from typing import Any, Generic, TypeVar

import torch.functional as F
from torch import Tensor, einsum

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.adapter import Adapter
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import  SD1UNet
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet

T = TypeVar("T", bound="SD1UNet | SDXLUNet")
TSDScaleCrafterAdapter = TypeVar("TSDScaleCrafterAdapter", bound="SDScaleCrafterAdapter[Any]")  # Self (see PEP 673)


class ConvSwitch(fl.Module):
    def __init__(self, conv1: fl.Conv2d, conv2: fl.Conv2d | None = None):
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2
    def forward(self, x: Tensor, condition: bool = True) -> Tensor:
        if condition or (self.conv2 is None):
            return self.conv1(x)
        return self.conv2(x)

TConvSwitch = TypeVar("TConvSwitch", bound="ConvSwitch")

class InflatedConvAdapter(Generic[TConvSwitch], fl.Chain, Adapter[TConvSwitch]):
    def __init__(self, target: ConvSwitch, inflation_transform: Tensor):
        self.target = target
        self.inflation_transform = inflation_transform
        with self.setup_adapter(target):
            super().__init__(target)
    def inject(self: TConvSwitch, parent: fl.Chain | None = None) -> TConvSwitch:
        # In this code, if we want to inflate, we must do it before we dilate or the dilation setting is overwritten
        if(isinstance(self.target.conv2, ReDilatedConv)) {
            print("Dilation must be done after inflation not before")
        }
        inflated_weight = self.target.conv1.weight.clone()
        out_channels, in_channels = inflated_weight.shape[0], inflated_weight.shape[1]
        inflated_kernel_size = int(math.sqrt(self.inflation_transform.shape[0]))
        transformed_weight = einsum(
            "mn, ion -> iom", self.inflation_transform.to(dtype=self.target.dtype), inflated_weight.view(out_channels, in_channels, -1))
        inflated_target = fl.Conv2d(in_channels, out_channels, inflated_kernel_size, self.target.stride, self.target.padding, self.target.groups, self.target.use_bias, self.target.dilation[0], self.target.padding_mode, self.target.device, self.target.dtype)
        inflated_target.weight.detach().copy_(transformed_weight.view(out_channels, in_channels, inflated_kernel_size, inflated_kernel_size))
        if inflated_target.bias is not None:
            inflated_target.bias.detach().copy_(self.target.bias.detach())
        self.target.conv2 = inflated_target
        return super().inject(parent)
    def eject(self) -> None:
        self.target.conv2 = None
        super().eject()

class ReDilatedConv(fl.Module):
    def __init__(self, conv: fl.Conv2d, dilation_factor: float = 1.0, kernel_inflated: bool = False, progressive: bool = False, inflate_timestep: int = 0, dilation_timestep: int = 700, interpolation_mode: str = "bilinear")
        self.conv = conv
        self.dilation_factor = dilation_factor
        self.interpolation_mode = interpolation_mode
        self.progressive = progressive
        self.kernel_inflated = kernel_inflated
        self.inflate_timestep = inflate_timestep
        self.dilation_timestep = dilation_timestep
    def forward(self, x: Tensor, timestep: Tensor) -> Tensor:
        dilation_factor = self.dilation_factor
        inflated_timestep = timestep > self.inflate_timestep
        if self.progressive:
            dilation_factor = max(math.ceil(dilation_factor * ((self.dilation_timestep - timestep) / self.dilation_timestep)), 2)
        if inflated_timestep and self.kernel_inflated:
             dilation_factor /= 2
        dilation = math.ceil(dilation_factor)
        scale = dilation/dilation_factor
        original_dilation, original_padding = self.conv.dilation, self.conv.padding
        original_kernel_size = self.conv.weight.shape[-1]
        inflation_kernel_size = (original_kernel_size - 1) // 2
        width, height = x.shape[-2], x.shape[-1]
        strides = (self.conv.stride[0], self.conv.stride[1])
        self.module.dilation, self.module.padding = dilation, (
            dilation * inflation_kernel_size, dilation * inflation_kernel_size
        )
        original_size = (int(width/strides[0]), int(height/strides[1]))
        intermediate_size = (round(width*scale), round(height*scale))
        x = F.interpolate(x, size=intermediate_size, mode=self.interpolation_mode)
        x = self.module._conv_forward(x, self.conv.weight, self.conv.bias)
        self.module.dilation, self.module.padding = original_dilation, original_padding
        x = F.interpolate(x, size=original_size, mode=self.interpolation_mode)
        return x
class ReDilatedConvAdapter(Generic[T], fl.Chain, Adapter[T]):
    def __init__(self, target: TConvSwitch, dilation_factor: float = 1.0, kernel_inflated: bool = False, progressive: bool = False, inflate_timestep: int = 0, dilation_timestep: int = 700, interpolation_mode: str = "bilinear"):
        self.target = target
        self.dilation_factor = dilation_factor
        self.interpolation_mode = interpolation_mode
        self.progressive = progressive
        self.kernel_inflated = kernel_inflated
        self.inflate_timestep = inflate_timestep
        self.dilation_timestep = dilation_timestep
        with self.setup_adapter(target):
            super().__init__(target)
    def inject(self: TConvSwitch, parent: fl.Chain | None = None) -> TConvSwitch:
        if self.target.conv2 is not None:
            redilation_conv = self.target.conv2
        else:
            redilation_conv = self.target.conv1
        self.target.conv2 = ReDilatedConv(redilation_conv, self.dilation_factor, self.kernel_inflated, self.progressive, self.inflate_timestep, self.dilation_timestep, self.interpolation_mode)
    def eject(self) -> None:
        super().eject()

class SDScaleCrafterAdapter(Generic[T], fl.Chain, Adapter[T]):
    def __init__(self, target: T, dilation_setting: dict[str, float], inflate_settings: dict[str, str], inflate_timestep: int = 0, dilation_timestep: int = 700, progressive: bool =False) -> None:
        self.dilation_setting = dilation_setting
        self.inflate_settings = inflate_settings
        self.inflate_timestep = inflate_timestep
        self.dilation_timestep = dilation_timestep
        self.progressive = progressive
        # TODO: Reset all convoltions in unet with convswitch above
        with self.setup_adapter(target):
            super().__init__(target)

    def inject(self: TSDScaleCrafterAdapter, parent: fl.Chain | None = None) -> TSDScaleCrafterAdapter:
        # for each name first inject kernel and then inject dilation
        # add a 
        return super().inject(parent)

    def eject(self) -> None:
        super().eject()
