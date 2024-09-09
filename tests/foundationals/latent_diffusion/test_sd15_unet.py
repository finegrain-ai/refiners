import pytest
import torch

from refiners.fluxion import manual_seed
from refiners.fluxion.utils import no_grad
from refiners.foundationals.latent_diffusion import SD1UNet


@pytest.fixture(scope="module")
def refiners_sd15_unet(test_device: torch.device) -> SD1UNet:
    unet = SD1UNet(in_channels=4, device=test_device)
    return unet


def test_unet_context_flush(refiners_sd15_unet: SD1UNet):
    manual_seed(0)
    text_embedding = torch.randn(1, 77, 768, device=refiners_sd15_unet.device, dtype=refiners_sd15_unet.dtype)
    timestep = torch.randint(0, 999, size=(1, 1), device=refiners_sd15_unet.device)
    x = torch.randn(1, 4, 32, 32, device=refiners_sd15_unet.device, dtype=refiners_sd15_unet.dtype)

    refiners_sd15_unet.set_clip_text_embedding(clip_text_embedding=text_embedding)  # not flushed between forward-s

    with no_grad():
        refiners_sd15_unet.set_timestep(timestep=timestep)
        y_1 = refiners_sd15_unet(x.clone())

    with no_grad():
        refiners_sd15_unet.set_timestep(timestep=timestep)
        y_2 = refiners_sd15_unet(x.clone())

    assert torch.equal(y_1, y_2)
