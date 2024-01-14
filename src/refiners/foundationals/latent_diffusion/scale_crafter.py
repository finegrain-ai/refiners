import math
from typing import TYPE_CHECKING, Any, Generic, TypeVar, List

import torch.nn.functional as F
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

def Tensor2int(tensor: Tensor) -> int:
    return int(tensor[0].cpu().detach().numpy())
class ReDilatedConv(fl.Module):
    def __init__(self, conv: fl.Conv2d, dilation_factor: float = 1.0, kernel_inflated: bool = False, progressive: bool = False, inflate_timestep: int = 1000, dilation_timestep: int = 300):
        super().__init__()
        self.conv = conv
        self.dilation_factor = dilation_factor
        self.progressive = progressive
        self.kernel_inflated = kernel_inflated
        self.inflate_timestep = inflate_timestep
        self.dilation_timestep = dilation_timestep
    def forward(self, x: Tensor, timestep: int) -> Tensor:
        if timestep < self.dilation_timestep:
            return self.conv(x)
        dilation_factor = self.dilation_factor
        inflated_timestep = timestep < self.inflate_timestep
        if self.progressive:
            dilation_factor = max(math.ceil(dilation_factor * (timestep-self.dilation_timestep) / (1000-self.dilation_timestep)), 2)
        if inflated_timestep and self.kernel_inflated:
             dilation_factor /= 2
        dilation = math.ceil(dilation_factor)
        scale = dilation/dilation_factor
        original_dilation, original_padding = self.conv.dilation, self.conv.padding
        original_kernel_size = self.conv.weight.shape[-1]
        inflation_kernel_size = (original_kernel_size - 1) // 2
        width, height = x.shape[-2], x.shape[-1]
        strides = (self.conv.stride[0], self.conv.stride[1])
        self.conv.dilation, self.conv.padding = (dilation, dilation), (
            dilation * inflation_kernel_size, dilation * inflation_kernel_size
        )
        original_size = (int(width/strides[0]), int(height/strides[1]))
        intermediate_size = (round(width*scale), round(height*scale))
        x = F.interpolate(x, size=intermediate_size, mode="bilinear")
        x = self.conv._conv_forward(x, self.conv.weight, self.conv.bias)
        self.conv.dilation, self.conv.padding = original_dilation, original_padding
        x = F.interpolate(x, size=original_size, mode="bilinear")
        return x

class ConvMap(fl.Module):
    def __init__(self, convs: List[fl.Conv2d | ReDilatedConv]):
        super().__init__()
        assert len(convs) > 0
        self.convs = convs
    def forward(self, x: Tensor, timestep: int, index: int = 0) -> Tensor:
        conv = self.convs[index]
        if isinstance(conv, fl.Conv2d):
            return conv(x)
        return conv(x, timestep)

class ConvNode(fl.Module):
    def __init__(self, conv_map1: ConvMap, conv_map2: ConvMap | None = None):
        super().__init__()
        self.conv_map1 = conv_map1
        self.conv_map2 = conv_map2
    def forward(self, x: Tensor, timestep_tensor: Tensor, nodeCondition: bool = True, switchCondition: int = 0) -> Tensor:
        timestep = Tensor2int(timestep_tensor)
        if nodeCondition or (self.conv_map2 is None):
            return self.conv_map1(x, timestep, switchCondition)
        return self.conv_map2(x, timestep, switchCondition)

class ScaleCrafterRouter(fl.Chain):
    def __init__(self, conv_node: ConvNode, inflate_timestep: int = 1000, noise_damped_timestep: int = 1000):
        super().__init__()
        self.inflate_timestep = inflate_timestep
        self.noise_damped_timestep = noise_damped_timestep
        super().__init__(
            fl.Parallel(
                fl.Identity(),
                fl.UseContext(context="diffusion", key="timestep"),
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
    def route_inflate(self, timestep_tensor: Tensor) -> int:
        timestep = Tensor2int(timestep_tensor)
        return int(timestep >= self.inflate_timestep)
    def route_noise_damped(self, timestep_tensor: Tensor, noise_damped: bool) -> bool:
        timestep = Tensor2int(timestep_tensor)
        return not ((timestep >= self.noise_damped_timestep) and noise_damped)
class ReDilatedConvAdapter(Generic[TConvMap], fl.Chain, Adapter[TConvMap]):
    def __init__(self, target: ConvMap, dilation_factor: float = 1.0, kernel_inflated: bool = False, progressive: bool = False, inflate_timestep: int = 1000, dilation_timestep: int = 300):
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
            redilated_convs.append(redilated_conv)
        self.target.convs = redilated_convs
        return super().inject(parent)
    def eject(self) -> None:
        convs = []
        for redilated_conv in self.target.convs:
            conv = redilated_conv.conv
            convs.append(conv)
        self.target.convs = convs
        super().eject()

class InflatedConvAdapter(Generic[TConvMap], fl.Chain, Adapter[TConvMap]):
    def __init__(self, target: ConvMap, inflation_transform: Tensor):
        self.inflation_transform = inflation_transform
        with self.setup_adapter(target):
            super().__init__(target)
    def inject(self: TConvMap, parent: fl.Chain | None = None) -> TConvMap:
        target_conv = self.target.convs[0]
        if isinstance(target_conv, ReDilatedConv):
            raise Exception("Dilation must be done after inflation not before")
        inflated_weight = target_conv.weight.clone()
        out_channels, in_channels = inflated_weight.shape[0], inflated_weight.shape[1]
        inflated_kernel_size = int(math.sqrt(self.inflation_transform.shape[0]))
        transformed_weight = einsum(
            "mn, ion -> iom", self.inflation_transform.to(dtype=target_conv.dtype), inflated_weight.view(out_channels, in_channels, -1))
        inflated_target = fl.Conv2d(in_channels, out_channels, inflated_kernel_size, target_conv.stride, target_conv.padding, target_conv.groups, target_conv.use_bias, target_conv.dilation[0], target_conv.padding_mode, target_conv.device, target_conv.dtype)
        inflated_target.weight.detach().copy_(transformed_weight.view(out_channels, in_channels, inflated_kernel_size, inflated_kernel_size))
        if inflated_target.bias is not None:
            inflated_target.bias.detach().copy_(target_conv.bias.detach())
        self.target.convs.append(inflated_target)
        return super().inject(parent)
    def eject(self) -> None:
        self.target.convs.pop(-1)
        super().eject()

class SDScaleCrafterAdapter(Generic[T], fl.Chain, Adapter[T]):
    def __init__(self, target: T, dilation_settings: dict[str, float] = {}, inflate_settings: list[str] = [], noise_damped_dilation_settings: dict[str, float] = {}, noise_damped_inflate_settings: list[str] = [], inflate_transform: Tensor | None = None, inflate_timestep: int = 1000, dilation_timestep: int = 300, noise_damped_timestep: int = 1000, progressive: bool =False) -> None:
        self.dilation_settings = dilation_settings
        self.noise_damped_dilation_settings = noise_damped_dilation_settings
        self.inflate_transform = inflate_transform
        self.inflate_settings = inflate_settings
        self.noise_damped_inflate_settings = noise_damped_inflate_settings
        self.inflate_timestep = inflate_timestep
        self.dilation_timestep = dilation_timestep
        self.noise_damped_timestep = noise_damped_timestep
        self.progressive = progressive
        self.sub_adapters = {}
        with self.setup_adapter(target):
            super().__init__(target)
    def init_context(self) -> Contexts:
        return {
            "scale_crafter": {"noise_damped": False}
        }
    def set_noise_damped(self, noise_damped: bool) -> None:
        self.set_context("scale_crafter", {"noise_damped": noise_damped})
    def uses_noise_damped(self) -> bool:
        return self.noise_damped_timestep < 1000
    def inject(self: TSDScaleCrafterAdapter, parent: fl.Chain | None = None) -> TSDScaleCrafterAdapter:
        print(self.inflate_settings)
        for name, module in self.target.named_modules():
            if isinstance(module, fl.Conv2d):
                self.sub_adapters[name] = self.sub_adapters.get(name, {})
                conv1 = module
                conv_chain = conv1._parent[0]
                conv_switch = ConvMap([conv1])
                kernel_inflated = False
                print("conv", name)
                if name in self.inflate_settings:
                    assert name in self.dilation_settings
                    assert self.inflate_transform is not None
                    kernel_inflated = True
                    inflate_adapter = InflatedConvAdapter(conv_switch, self.inflate_transform)
                    self.sub_adapters[name]["inflate_adapter"] = inflate_adapter
                    inflate_adapter.inject()
                    print("inflate conv map", len(conv_switch.convs))
                if name in self.dilation_settings:
                    dilation = self.dilation_settings[name]
                    redilated_conv_adapter = ReDilatedConvAdapter(conv_switch, dilation, kernel_inflated, self.progressive, self.inflate_timestep, self.dilation_timestep)
                    self.sub_adapters[name]["redilated_conv_adapter"] = redilated_conv_adapter
                    redilated_conv_adapter.inject()
                    print("conv map", len(conv_switch.convs))
                noise_damped_conv_switch = ConvMap([conv1])
                noise_dampled_kernel_inflated = False
                if name in self.noise_damped_inflate_settings:
                    assert name in self.noise_damped_dilation_settings
                    assert self.inflate_transform is not None
                    noise_dampled_kernel_inflated = True
                    noise_damped_inflate_adapter = InflatedConvAdapter(noise_damped_conv_switch, self.inflate_transform)
                    self.sub_adapters[name]["noise_damped_inflate_adapter"] = noise_damped_inflate_adapter
                    noise_damped_inflate_adapter.inject()
                    print("inflate noise damped conv map", len(noise_damped_conv_switch.convs))
                if name in self.noise_damped_dilation_settings:
                    dilation = self.noise_damped_dilation_settings[name]
                    noise_damped_redilated_conv_adapter = ReDilatedConvAdapter(noise_damped_conv_switch, dilation, noise_dampled_kernel_inflated, self.progressive, self.inflate_timestep, self.dilation_timestep)
                    self.sub_adapters[name]["noise_damped_redilated_conv_adapter"] = noise_damped_redilated_conv_adapter
                    noise_damped_redilated_conv_adapter.inject()
                    print("noise damped conv map", len(noise_damped_conv_switch.convs))
                conv_node = ConvNode(conv_switch, noise_damped_conv_switch)
                conv_chain.replace(conv1, ScaleCrafterRouter(conv_node, self.inflate_timestep, self.noise_damped_timestep))
        return super().inject(parent)

    def eject(self) -> None:
        for name, module in self.target.named_modules():
            if isinstance(module, ScaleCrafterRouter):
                conv_sub_adapters = self.sub_adapters[name.replace("ScaleCrafterRouter", "Conv2d")]
                for key in conv_sub_adapters:
                    conv_sub_adapters[key].eject()
                conv_chain = module
                parent = conv_chain._parent[0]
                conv = conv_chain[-1].conv_map1.convs[0]
                conv._parent = [parent]
                parent.replace(conv_chain, conv)
        super().eject()
