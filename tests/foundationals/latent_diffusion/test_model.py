import torch
from PIL import Image

from refiners.fluxion.utils import manual_seed, no_grad
from refiners.foundationals.latent_diffusion import StableDiffusion_1_Inpainting
from refiners.foundationals.latent_diffusion.model import LatentDiffusionModel


@no_grad()
def test_sample_noise():
    manual_seed(2)
    latents_0 = LatentDiffusionModel.sample_noise(size=(1, 4, 64, 64))
    manual_seed(2)
    latents_1 = LatentDiffusionModel.sample_noise(size=(1, 4, 64, 64), offset_noise=0.0)

    assert torch.allclose(latents_0, latents_1, atol=1e-6, rtol=0)


@no_grad()
def test_sd1_inpainting(test_device: torch.device) -> None:
    sd = StableDiffusion_1_Inpainting(device=test_device)

    latent_noise = torch.randn(1, 4, 64, 64, device=test_device)
    target_image = Image.new("RGB", (512, 512))
    mask = Image.new("L", (512, 512))

    sd.set_inpainting_conditions(target_image=target_image, mask=mask)
    text_embedding = sd.compute_clip_text_embedding("")
    output = sd(latent_noise, step=0, clip_text_embedding=text_embedding)

    assert output.shape == (1, 4, 64, 64)
