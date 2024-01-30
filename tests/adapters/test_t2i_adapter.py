from typing import overload

import pytest
import torch

from refiners.fluxion.utils import no_grad
from refiners.foundationals.latent_diffusion import SD1T2IAdapter, SD1UNet, SDXLT2IAdapter, SDXLUNet
from refiners.foundationals.latent_diffusion.t2i_adapter import T2IFeatures


@overload
def new_adapter(target: SD1UNet, name: str) -> SD1T2IAdapter:
    ...


@overload
def new_adapter(target: SDXLUNet, name: str) -> SDXLT2IAdapter:
    ...


def new_adapter(target: SD1UNet | SDXLUNet, name: str) -> SD1T2IAdapter | SDXLT2IAdapter:
    if isinstance(target, SD1UNet):
        return SD1T2IAdapter(target=target, name=name)
    else:
        return SDXLT2IAdapter(target=target, name=name)


@no_grad()
@pytest.mark.parametrize("k_unet", [SD1UNet, SDXLUNet])
def test_inject_eject(k_unet: type[SD1UNet] | type[SDXLUNet], test_device: torch.device):
    unet = k_unet(in_channels=4, device=test_device, dtype=torch.float16)
    initial_repr = repr(unet)
    adapter_1 = new_adapter(unet, "adapter 1")
    assert repr(unet) == initial_repr
    adapter_1.inject()
    assert repr(unet) != initial_repr

    with pytest.raises(AssertionError) as already_injected_error:
        new_adapter(unet, "adapter 1").inject()

    assert str(already_injected_error.value) == "T2I-Adapter named adapter 1 is already injected"

    adapter_2 = new_adapter(unet, "adapter 2").inject()

    adapter_1.eject()

    new_adapter_1 = new_adapter(unet, "adapter 1").inject()
    new_adapter_1.eject()

    assert unet.parent == adapter_2
    assert unet.find(T2IFeatures) is not None

    adapter_2.eject()

    assert unet.parent is None
    assert unet.find(T2IFeatures) is None
    assert repr(unet) == initial_repr
