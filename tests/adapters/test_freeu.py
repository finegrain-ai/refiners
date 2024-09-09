from typing import Iterator

import pytest
import torch

from refiners.fluxion import manual_seed
from refiners.fluxion.layers import Chain
from refiners.fluxion.utils import no_grad
from refiners.foundationals.latent_diffusion import SD1UNet, SDXLUNet
from refiners.foundationals.latent_diffusion.freeu import FreeUResidualConcatenator, SDFreeUAdapter


@pytest.fixture(scope="module", params=[True, False])
def unet(request: pytest.FixtureRequest) -> Iterator[SD1UNet | SDXLUNet]:
    xl: bool = request.param
    unet = SDXLUNet(in_channels=4) if xl else SD1UNet(in_channels=4)
    yield unet


def test_freeu_adapter(unet: SD1UNet | SDXLUNet) -> None:
    freeu = SDFreeUAdapter(unet, backbone_scales=[1.2, 1.2], skip_scales=[0.9, 0.9])

    assert len(list(unet.walk(FreeUResidualConcatenator))) == 0

    with pytest.raises(AssertionError) as exc:
        freeu.eject()
    assert "could not find" in str(exc.value)

    freeu.inject()
    assert len(list(unet.walk(FreeUResidualConcatenator))) == 2

    freeu.eject()
    assert len(list(unet.walk(FreeUResidualConcatenator))) == 0


def test_freeu_adapter_too_many_scales(unet: SD1UNet | SDXLUNet) -> None:
    num_blocks = len(unet.layer("UpBlocks", Chain))

    with pytest.raises(AssertionError):
        SDFreeUAdapter(unet, backbone_scales=[1.2] * (num_blocks + 1), skip_scales=[0.9] * (num_blocks + 1))


def test_freeu_adapter_inconsistent_scales(unet: SD1UNet | SDXLUNet) -> None:
    with pytest.raises(AssertionError):
        SDFreeUAdapter(unet, backbone_scales=[1.2, 1.2], skip_scales=[0.9, 0.9, 0.9])


def test_freeu_identity_scales() -> None:
    manual_seed(0)
    text_embedding = torch.randn(1, 77, 768)
    timestep = torch.randint(0, 999, size=(1, 1))
    x = torch.randn(1, 4, 32, 32)

    unet = SD1UNet(in_channels=4)
    unet.set_clip_text_embedding(clip_text_embedding=text_embedding)  # not flushed between forward-s

    with no_grad():
        unet.set_timestep(timestep=timestep)
        y_1 = unet(x.clone())

    freeu = SDFreeUAdapter(unet, backbone_scales=[1.0, 1.0], skip_scales=[1.0, 1.0])
    freeu.inject()

    with no_grad():
        unet.set_timestep(timestep=timestep)
        y_2 = unet(x.clone())

    # The FFT -> inverse FFT sequence (skip features) introduces small numerical differences
    assert torch.allclose(y_1, y_2, atol=1e-5)
