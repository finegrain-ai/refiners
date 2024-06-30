import torch

import refiners.fluxion.layers as fl
from refiners.fluxion.utils import no_grad
from refiners.foundationals.latent_diffusion import SD1ELLAAdapter, SD1UNet
from refiners.foundationals.latent_diffusion.ella_adapter import ELLACrossAttentionAdapter


def new_adapter(target: SD1UNet) -> SD1ELLAAdapter:
    return SD1ELLAAdapter(target=target)


@no_grad()
def test_inject_eject(test_device: torch.device):
    unet = SD1UNet(in_channels=4, device=test_device, dtype=torch.float16)
    initial_repr = repr(unet)
    adapter = new_adapter(unet)
    assert repr(unet) == initial_repr
    adapter.inject()
    assert repr(unet) != initial_repr
    adapter.eject()
    assert repr(unet) == initial_repr
    adapter.inject()
    assert repr(unet) != initial_repr
    adapter.eject()
    assert repr(unet) == initial_repr


@no_grad()
def test_ella_cross_attention(test_device: torch.device):
    unet = SD1UNet(in_channels=4, device=test_device, dtype=torch.float16)
    adapter = new_adapter(unet).inject()

    def predicate(m: fl.Module, p: fl.Chain) -> bool:
        return isinstance(p, ELLACrossAttentionAdapter) and isinstance(m, fl.UseContext)

    for m, _ in unet.walk(predicate):
        assert isinstance(m, fl.UseContext)
        assert m.context == "ella"
        assert m.key == "latents"
    assert len(adapter.sub_adapters) == 32
