import pytest

from refiners.fluxion.layers.attentions import SelfAttention
from refiners.foundationals.latent_diffusion import SD1UNet, SDXLUNet
from refiners.foundationals.latent_diffusion.style_aligned import (
    SharedSelfAttentionAdapter,
    StyleAligned,
    StyleAlignedAdapter,
)


@pytest.fixture(scope="module", params=[1.0, 100.0])
def scale(request: pytest.FixtureRequest) -> float:
    return request.param


@pytest.fixture(scope="module", params=[SD1UNet, SDXLUNet])
def unet(request: pytest.FixtureRequest) -> SD1UNet | SDXLUNet:
    return request.param(in_channels=4)


@pytest.fixture(scope="module")
def self_attention() -> SelfAttention:
    return SelfAttention(embedding_dim=100)


@pytest.fixture(scope="module")
def adapter_SAA(unet: SD1UNet | SDXLUNet, scale: float) -> StyleAlignedAdapter[SD1UNet | SDXLUNet]:
    return StyleAlignedAdapter(target=unet, scale=scale)


@pytest.fixture(scope="module")
def adapter_SSA(self_attention: SelfAttention, scale: float) -> SharedSelfAttentionAdapter:
    return SharedSelfAttentionAdapter(target=self_attention, scale=scale)


def test_inject_eject_SharedSelfAttentionAdapter(
    self_attention: SD1UNet | SDXLUNet, adapter_SSA: SharedSelfAttentionAdapter
):
    initial_repr = repr(self_attention)

    assert self_attention.parent is None
    assert self_attention.find(StyleAligned) is None
    assert repr(self_attention) == initial_repr

    adapter_SSA.inject()
    assert self_attention.parent is not None
    assert self_attention.find(StyleAligned) is not None
    assert repr(self_attention) != initial_repr

    adapter_SSA.eject()
    assert self_attention.parent is None
    assert self_attention.find(StyleAligned) is None
    assert repr(self_attention) == initial_repr


def test_inject_eject_StyleAlignedAdapter(
    unet: SD1UNet | SDXLUNet, adapter_SAA: StyleAlignedAdapter[SD1UNet | SDXLUNet]
):
    initial_repr = repr(unet)

    assert unet.parent is None
    assert unet.find(SharedSelfAttentionAdapter) is None
    assert repr(unet) == initial_repr

    adapter_SAA.inject()
    assert unet.parent is not None
    assert unet.find(SharedSelfAttentionAdapter) is not None
    assert repr(unet) != initial_repr

    adapter_SAA.eject()
    assert unet.parent is None
    assert unet.find(SharedSelfAttentionAdapter) is None
    assert repr(unet) == initial_repr
