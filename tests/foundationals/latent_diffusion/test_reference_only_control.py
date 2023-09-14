import torch
import pytest


from refiners.foundationals.latent_diffusion import SD1UNet
from refiners.foundationals.latent_diffusion.reference_only_control import (
    ReferenceOnlyControlAdapter,
    SaveLayerNormAdapter,
    SelfAttentionInjectionAdapter,
    SelfAttentionInjectionPassthrough,
)
from refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock


@torch.no_grad()
def test_refonly_inject_eject() -> None:
    unet = SD1UNet(in_channels=9)
    adapter = ReferenceOnlyControlAdapter(unet)

    nb_cross_attention_blocks = len(list(unet.walk(CrossAttentionBlock)))
    assert nb_cross_attention_blocks > 0

    assert unet.parent is None
    assert len(list(unet.walk(SelfAttentionInjectionPassthrough))) == 0
    assert len(list(unet.walk(SaveLayerNormAdapter))) == 0
    assert len(list(unet.walk(SelfAttentionInjectionAdapter))) == 0

    with pytest.raises(AssertionError) as exc:
        adapter.eject()
    assert "not the first element" in str(exc.value)

    adapter.inject()

    assert unet.parent == adapter
    assert len(list(unet.walk(SelfAttentionInjectionPassthrough))) == 1
    assert len(list(unet.walk(SaveLayerNormAdapter))) == nb_cross_attention_blocks
    assert len(list(unet.walk(SelfAttentionInjectionAdapter))) == nb_cross_attention_blocks

    with pytest.raises(AssertionError) as exc:
        adapter.inject()
    assert "already injected" in str(exc.value)

    adapter.eject()

    assert unet.parent is None
    assert len(list(unet.walk(SelfAttentionInjectionPassthrough))) == 0
    assert len(list(unet.walk(SaveLayerNormAdapter))) == 0
    assert len(list(unet.walk(SelfAttentionInjectionAdapter))) == 0
