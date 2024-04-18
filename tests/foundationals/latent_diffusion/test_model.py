import torch

from refiners.fluxion.utils import manual_seed, no_grad
from refiners.foundationals.latent_diffusion.model import LatentDiffusionModel


@no_grad()
def test_sample_noise():
    manual_seed(2)
    latents_0 = LatentDiffusionModel.sample_noise(size=(1, 4, 64, 64))
    manual_seed(2)
    latents_1 = LatentDiffusionModel.sample_noise(size=(1, 4, 64, 64), offset_noise=0.0)

    assert torch.allclose(latents_0, latents_1, atol=1e-6, rtol=0)
