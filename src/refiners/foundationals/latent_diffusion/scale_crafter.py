import math
from typing import TYPE_CHECKING, Any, Generic, TypeVar, List

import torch.functional as F
from torch import Tensor, einsum

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.adapter import Adapter
if TYPE_CHECKING:
    from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import  SD1UNet
    from refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet
from refiners.fluxion.context import Contexts

T = TypeVar("T", bound="SD1UNet | SDXLUNet")
TSDScaleCrafterAdapter = TypeVar("TSDScaleCrafterAdapter", bound="SDScaleCrafterAdapter[Any]")  # Self (see PEP 673)
TConvMap = TypeVar("TConvMap", bound="ConvMap")

class ReDilatedConv(fl.Module):
    def __init__(self, conv: fl.Conv2d, dilation_factor: float = 1.0, kernel_inflated: bool = False, progressive: bool = False, inflate_timestep: int = 0, dilation_timestep: int = 700):
        self.conv = conv
        self.dilation_factor = dilation_factor
        self.progressive = progressive
        self.kernel_inflated = kernel_inflated
        self.inflate_timestep = inflate_timestep
        self.dilation_timestep = dilation_timestep
    def forward(self, x: Tensor, timestep: Tensor) -> Tensor:
        if timestep >= self.dilation_timestep:
            return self.conv(x)
        dilation_factor = self.dilation_factor
        inflated_timestep = timestep < self.inflate_timestep
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
        x = F.interpolate(x, size=intermediate_size, mode="bilinear")
        x = self.module._conv_forward(x, self.conv.weight, self.conv.bias)
        self.module.dilation, self.module.padding = original_dilation, original_padding
        x = F.interpolate(x, size=original_size, mode="bilinear")
        return x

class RedilatedConvChain(fl.Chain):
    def __init__(self, redilated_conv: ReDilatedConv):
        self.redilated_conv = redilated_conv
        super().__init__(
            fl.UseContext("diffusion", "timestep"),
            redilated_conv
        )

class ConvMap(fl.Module):
    def __init__(self, convs: List[fl.Conv2d | RedilatedConvChain]):
        super().__init__()
        self.convs = convs
    def forward(self, x: Tensor, index: int = 0) -> Tensor:
        return self.convs[index](x)

class ConvNode(fl.Module):
    def __init__(self, conv_map1: ConvMap, conv_map2: ConvMap | None = None):
        super().__init__()
        self.conv_map1 = conv_map1
        self.conv_map2 = conv_map2
    def forward(self, x: Tensor, nodeCondition: bool = True, switchCondition: bool = True) -> Tensor:
        if nodeCondition or (self.conv_map2 is None):
            return self.conv_map1(x, switchCondition)
        return self.conv_map2(x, switchCondition)

class ScaleCrafterRouter(fl.Chain):
    def __init__(self, conv_node: ConvNode, inflate_timestep: int = 0, noise_damped_timestep: int = 700):
        self.conv_node = conv_node
        self.inflate_timestep = inflate_timestep
        self.noise_damped_timestep = noise_damped_timestep
        super().__init__(
            fl.Parallel(
                fl.Chain(
                    fl.Parallel(
                        fl.UseContext(context="diffusion", key="timestep"),
                        fl.UseContext(context="scale_crafter", key="noise_damped"),
                    ),
                    fl.Lambda(self.route_noise_damped)
                ),
                fl.Chain(
                    fl.UseContext(context="diffusion", key="timestep"),
                    fl.Lambda(self.route_inflate)
                )
            ),
            conv_node
        )
    def route_inflate(self, timestep: Tensor):
        return int(timestep < self.inflate_timestep)
    def route_noise_damped(self, timestep: Tensor, noise_damped: bool) -> int:
        return int(timestep < self.noise_damped_timestep)
class ReDilatedConvAdapter(Generic[TConvMap], fl.Chain, Adapter[TConvMap]):
    def __init__(self, target: ConvMap, dilation_factor: float = 1.0, kernel_inflated: bool = False, progressive: bool = False, inflate_timestep: int = 0, dilation_timestep: int = 700):
        self.target = target
        self.dilation_factor = dilation_factor
        self.progressive = progressive
        self.kernel_inflated = kernel_inflated
        self.inflate_timestep = inflate_timestep
        self.dilation_timestep = dilation_timestep
        with self.setup_adapter(target):
            super().__init__(target)
    def inject(self: TConvMap, parent: fl.Chain | None = None) -> TConvMap:
        redilated_convs = []
        for conv in self.target.convs:
            redilated_conv = ReDilatedConv(conv, self.dilation_factor, self.kernel_inflated, self.progressive, self.inflate_timestep, self.dilation_timestep)
            redilated_convs.append(RedilatedConvChain(redilated_conv))
        self.target.convs = redilated_convs
        return super().inject(parent)
    def eject(self) -> None:
        convs = []
        for redilated_conv_chain in self.target.convs:
            conv = redilated_conv_chain.redilated_conv.conv
            convs.append(conv)
        self.target.convs = convs
        super().eject()

class InflatedConvAdapter(Generic[TConvMap], fl.Chain, Adapter[TConvMap]):
    def __init__(self, target: ConvMap, inflation_transform: Tensor):
        self.target = target
        self.inflation_transform = inflation_transform
        with self.setup_adapter(target):
            super().__init__(target)
    def inject(self: TConvMap, parent: fl.Chain | None = None) -> TConvMap:
        if isinstance(self.target.convs[-1], RedilatedConvChain):
            print("Dilation must be done after inflation not before")
        inflated_weight = self.target.convs[0].weight.clone()
        out_channels, in_channels = inflated_weight.shape[0], inflated_weight.shape[1]
        inflated_kernel_size = int(math.sqrt(self.inflation_transform.shape[0]))
        transformed_weight = einsum(
            "mn, ion -> iom", self.inflation_transform.to(dtype=self.target.dtype), inflated_weight.view(out_channels, in_channels, -1))
        inflated_target = fl.Conv2d(in_channels, out_channels, inflated_kernel_size, self.target.stride, self.target.padding, self.target.groups, self.target.use_bias, self.target.dilation[0], self.target.padding_mode, self.target.device, self.target.dtype)
        inflated_target.weight.detach().copy_(transformed_weight.view(out_channels, in_channels, inflated_kernel_size, inflated_kernel_size))
        if inflated_target.bias is not None:
            inflated_target.bias.detach().copy_(self.target.bias.detach())
        self.target.convs.append(inflated_target)
        return super().inject(parent)
    def eject(self) -> None:
        self.target.convs.pop(-1)
        super().eject()

class SDScaleCrafterAdapter(Generic[T], fl.Chain, Adapter[T]):
    def __init__(self, target: T, dilation_settings: dict[str, float], inflate_settings: list[str], noise_damped_dilation_settings: dict[str, float], noise_damped_inflate_settings: dict[str, str], inflate_transform: Tensor | None = None, inflate_timestep: int = 0, dilation_timestep: int = 700, noise_damped_timestep: int = 700, progressive: bool =False) -> None:
        self.dilation_settings = dilation_settings
        self.noise_damped_dilation_settings = noise_damped_dilation_settings
        self.inflate_transform = inflate_transform
        self.inflate_settings = inflate_settings
        self.noise_damped_inflate_settings = noise_damped_inflate_settings
        self.inflate_timestep = inflate_timestep
        self.dilation_timestep = dilation_timestep
        self.noise_damped_timestep = noise_damped_timestep
        self.progressive = progressive
        with self.setup_adapter(target):
            super().__init__(target)
    def init_context(self) -> Contexts:
        return {
            "scale_crafter": {"noise_damped": False}
        }
    def set_noise_damped(self, noise_damped: bool) -> None:
        self.set_context("scale_crafter", {"noise_damped": noise_damped})
    def inject(self: TSDScaleCrafterAdapter, parent: fl.Chain | None = None) -> TSDScaleCrafterAdapter:
        for name, module in self.target.named_modules():
            if isinstance(module, fl.Conv2d):
                conv1 = module
                parent = conv1._parent[0]
                conv_switch = ConvMap([conv1])
                kernel_inflated = False
                if name in self.inflate_settings:
                    assert name in self.dilation_settings
                    assert self.inflate_transform is not None
                    kernel_inflated = True
                    inflate_adapter = InflatedConvAdapter(conv_switch, self.inflate_transform)
                    inflate_adapter.inject()
                if name in self.dilation_settings:
                    dilation = self.dilation_settings[name]
                    redilated_conv_adapter = ReDilatedConvAdapter(conv_switch, dilation, kernel_inflated, self.progressive, self.inflate_timestep, self.dilation_timestep)
                    redilated_conv_adapter.inject()
                noise_damped_conv_switch = ConvMap([conv1])
                noise_dampled_kernel_inflated = False
                if name in self.noise_damped_inflate_settings:
                    assert name in self.noise_damped_dilation_settings
                    assert self.inflate_transform is not None
                    noise_dampled_kernel_inflated = True
                    noise_damped_inflate_adapter = InflatedConvAdapter(noise_damped_conv_switch, self.inflate_transform)
                    noise_damped_inflate_adapter.inject()
                if name in self.noise_damped_dilation_settings:
                    dilation = self.noise_damped_dilation_settings[name]
                    noise_damped_redilated_conv_adapter = ReDilatedConvAdapter(noise_damped_conv_switch, dilation, noise_dampled_kernel_inflated, self.progressive, self.inflate_timestep, self.dilation_timestep)
                    noise_damped_redilated_conv_adapter.inject()
                conv_node = ConvNode(conv_switch, noise_damped_conv_switch)
                parent.replace(conv1, ScaleCrafterRouter(conv_node))
        return super().inject(parent)

    def eject(self) -> None:
        for module in self.target.modules():
            if isinstance(module, ScaleCrafterRouter):
                conv_chain = module
                parent = conv_chain._parent[0]
                parent.replace(conv_chain, conv_chain.conv_node.ConvMap1.convs[0])
        super().eject()
