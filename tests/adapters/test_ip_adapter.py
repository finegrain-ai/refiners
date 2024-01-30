from typing import overload

import pytest
import torch

import refiners.fluxion.layers as fl
from refiners.fluxion.utils import no_grad
from refiners.foundationals.latent_diffusion import SD1IPAdapter, SD1UNet, SDXLIPAdapter, SDXLUNet
from refiners.foundationals.latent_diffusion.image_prompt import ImageCrossAttention


@overload
def new_adapter(target: SD1UNet) -> SD1IPAdapter:
    ...


@overload
def new_adapter(target: SDXLUNet) -> SDXLIPAdapter:
    ...


def new_adapter(target: SD1UNet | SDXLUNet) -> SD1IPAdapter | SDXLIPAdapter:
    if isinstance(target, SD1UNet):
        return SD1IPAdapter(target=target)
    else:
        return SDXLIPAdapter(target=target)


@no_grad()
@pytest.mark.parametrize("k_unet", [SD1UNet, SDXLUNet])
def test_inject_eject(k_unet: type[SD1UNet] | type[SDXLUNet], test_device: torch.device):
    unet = k_unet(in_channels=4, device=test_device, dtype=torch.float16)
    initial_repr = repr(unet)
    adapter = new_adapter(unet)
    assert repr(unet) == initial_repr
    adapter.inject()
    assert repr(unet) != initial_repr
    adapter.eject()
    assert repr(unet) == initial_repr


@no_grad()
@pytest.mark.parametrize("k_unet", [SD1UNet, SDXLUNet])
def test_scale(k_unet: type[SD1UNet] | type[SDXLUNet], test_device: torch.device):
    unet = k_unet(in_channels=4, device=test_device, dtype=torch.float16)
    adapter = new_adapter(unet).inject()

    def predicate(m: fl.Module, p: fl.Chain) -> bool:
        return isinstance(p, ImageCrossAttention) and isinstance(m, fl.Multiply)

    for m, _ in unet.walk(predicate):
        assert isinstance(m, fl.Multiply)
        assert m.scale == 1.0

    adapter.scale = 0.42
    assert adapter.scale == 0.42
    for m, _ in unet.walk(predicate):
        assert isinstance(m, fl.Multiply)
        assert m.scale == 0.42
