import pytest
import torch

from refiners.fluxion.utils import no_grad
from refiners.foundationals.latent_diffusion import StableDiffusion_1, StableDiffusion_XL


@no_grad()
@pytest.mark.parametrize("k_sd", [StableDiffusion_1, StableDiffusion_XL])
def test_set_self_attention_guidance(
    k_sd: type[StableDiffusion_1] | type[StableDiffusion_XL], test_device: torch.device
):
    sd = k_sd(device=test_device, dtype=torch.float16)

    assert sd._find_sag_adapter() is None  # type: ignore
    sd.set_self_attention_guidance(enable=True, scale=0.42)
    adapter = sd._find_sag_adapter()  # type: ignore
    assert adapter is not None
    assert adapter.scale == 0.42

    sd.set_self_attention_guidance(enable=True, scale=0.75)
    assert sd._find_sag_adapter() == adapter  # type: ignore
    assert adapter.scale == 0.75

    sd.set_self_attention_guidance(enable=False)
    assert sd._find_sag_adapter() is None  # type: ignore
