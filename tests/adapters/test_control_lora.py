import torch

import refiners.fluxion.layers as fl
from refiners.foundationals.latent_diffusion import ControlLoraAdapter, SDXLUNet
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.control_lora import ZeroConvolution


def test_inject_eject(test_device: torch.device):
    unet = SDXLUNet(in_channels=4, device=test_device, dtype=torch.float16)
    initial_repr = repr(unet)
    adapter = ControlLoraAdapter(name="foo", target=unet)
    assert repr(unet) == initial_repr
    adapter.inject()
    assert repr(unet) != initial_repr
    adapter.eject()
    assert repr(unet) == initial_repr


def test_scale(test_device: torch.device):
    unet = SDXLUNet(in_channels=4, device=test_device, dtype=torch.float16)
    adapter = ControlLoraAdapter(name="foo", target=unet, scale=0.75).inject()

    def predicate(m: fl.Module, p: fl.Chain) -> bool:
        return isinstance(p, ZeroConvolution) and isinstance(m, fl.Multiply)

    for m, _ in unet.walk(predicate):
        assert isinstance(m, fl.Multiply)
        assert m.scale == 0.75

    adapter.scale = 0.42
    assert adapter.scale == 0.42
    for m, _ in unet.walk(predicate):
        assert isinstance(m, fl.Multiply)
        assert m.scale == 0.42
