from typing import Iterator

import pytest

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.adapter import lookup_top_adapter
from refiners.fluxion.utils import no_grad
from refiners.foundationals.latent_diffusion import SD1ControlnetAdapter, SD1UNet
from refiners.foundationals.latent_diffusion.stable_diffusion_1.controlnet import Controlnet


@pytest.fixture(scope="module", params=[True, False])
def unet(request: pytest.FixtureRequest) -> Iterator[SD1UNet]:
    with_parent: bool = request.param
    unet = SD1UNet(in_channels=9)
    if with_parent:
        fl.Chain(unet)
    yield unet


@no_grad()
def test_single_controlnet(unet: SD1UNet) -> None:
    original_parent = unet.parent
    cn = SD1ControlnetAdapter(unet, name="cn")

    assert unet.parent == original_parent
    assert len(list(unet.walk(Controlnet))) == 0

    with pytest.raises(ValueError) as exc:
        cn.eject()
    assert "not in" in str(exc.value)

    cn.inject()
    assert unet.parent == cn
    assert len(list(unet.walk(Controlnet))) == 1

    with pytest.raises(AssertionError) as exc:
        cn.inject()
    assert "already injected" in str(exc.value)

    cn.eject()
    assert unet.parent == original_parent
    assert len(list(unet.walk(Controlnet))) == 0


@no_grad()
def test_two_controlnets_eject_bottom_up(unet: SD1UNet) -> None:
    original_parent = unet.parent
    cn1 = SD1ControlnetAdapter(unet, name="cn1").inject()
    cn2 = SD1ControlnetAdapter(unet, name="cn2").inject()

    assert unet.parent == cn2
    assert unet in cn2
    assert unet not in cn1
    assert cn2.parent == cn1
    assert cn2 in cn1
    assert cn1.parent == original_parent
    assert len(list(unet.walk(Controlnet))) == 2
    assert cn1.target == unet
    assert lookup_top_adapter(cn1, cn1.target) == cn2

    cn2.eject()
    assert unet.parent == cn1
    assert unet in cn2
    assert cn2 not in cn1
    assert unet in cn1
    assert len(list(unet.walk(Controlnet))) == 1

    cn1.eject()
    assert unet.parent == original_parent
    assert len(list(unet.walk(Controlnet))) == 0


@no_grad()
def test_two_controlnets_eject_top_down(unet: SD1UNet) -> None:
    original_parent = unet.parent
    cn1 = SD1ControlnetAdapter(unet, name="cn1").inject()
    cn2 = SD1ControlnetAdapter(unet, name="cn2").inject()

    cn1.eject()
    assert cn2.parent == original_parent
    assert unet.parent == cn2

    cn2.eject()
    assert unet.parent == original_parent
    assert len(list(unet.walk(Controlnet))) == 0


@no_grad()
def test_two_controlnets_same_name(unet: SD1UNet) -> None:
    SD1ControlnetAdapter(unet, name="cnx").inject()
    cn2 = SD1ControlnetAdapter(unet, name="cnx")

    with pytest.raises(AssertionError) as exc:
        cn2.inject()
    assert "Controlnet named cnx is already injected" in str(exc.value)
