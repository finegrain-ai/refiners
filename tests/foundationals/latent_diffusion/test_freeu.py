from typing import Iterator

import pytest

from refiners.foundationals.latent_diffusion import SD1UNet, SDXLUNet
from refiners.foundationals.latent_diffusion.freeu import SDFreeUAdapter, FreeUResidualConcatenator


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
    num_blocks = len(unet.UpBlocks)

    with pytest.raises(AssertionError):
        SDFreeUAdapter(unet, backbone_scales=[1.2] * (num_blocks + 1), skip_scales=[0.9] * (num_blocks + 1))


def test_freeu_adapter_inconsistent_scales(unet: SD1UNet | SDXLUNet) -> None:
    with pytest.raises(AssertionError):
        SDFreeUAdapter(unet, backbone_scales=[1.2, 1.2], skip_scales=[0.9, 0.9, 0.9])
