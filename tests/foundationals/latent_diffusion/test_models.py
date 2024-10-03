import torch
from PIL import Image

from refiners.fluxion.utils import manual_seed, no_grad
from refiners.foundationals.latent_diffusion import StableDiffusion_1, StableDiffusion_1_Inpainting, StableDiffusion_XL
from refiners.foundationals.latent_diffusion.model import LatentDiffusionModel


@no_grad()
def test_sample_noise_zero_offset(test_device: torch.device, test_dtype_fp32_bf16_fp16: torch.dtype) -> None:
    manual_seed(2)
    latents_0 = LatentDiffusionModel.sample_noise(
        size=(1, 4, 64, 64),
        device=test_device,
        dtype=test_dtype_fp32_bf16_fp16,
    )
    manual_seed(2)
    latents_1 = LatentDiffusionModel.sample_noise(
        size=(1, 4, 64, 64),
        offset_noise=0.0,  # should be no-op
        device=test_device,
        dtype=test_dtype_fp32_bf16_fp16,
    )

    assert torch.allclose(latents_0, latents_1, atol=1e-6, rtol=0)


@no_grad()
def test_sd15_one_step(test_device: torch.device, test_dtype_fp32_bf16_fp16: torch.dtype) -> None:
    sd = StableDiffusion_1(device=test_device, dtype=test_dtype_fp32_bf16_fp16)

    # prepare inputs
    latent_noise = torch.randn(1, 4, 64, 64, device=test_device, dtype=test_dtype_fp32_bf16_fp16)
    text_embedding = sd.compute_clip_text_embedding("")

    # run the pipeline of models, for a single step
    output = sd(latent_noise, step=0, clip_text_embedding=text_embedding)

    assert output.shape == (1, 4, 64, 64)


@no_grad()
def test_sd15_inpainting_one_step(test_device: torch.device, test_dtype_fp32_bf16_fp16: torch.dtype) -> None:
    sd = StableDiffusion_1_Inpainting(device=test_device, dtype=test_dtype_fp32_bf16_fp16)

    # prepare inputs
    latent_noise = torch.randn(1, 4, 64, 64, device=test_device, dtype=test_dtype_fp32_bf16_fp16)
    target_image = Image.new("RGB", (512, 512))
    mask = Image.new("L", (512, 512))
    sd.set_inpainting_conditions(target_image=target_image, mask=mask)
    text_embedding = sd.compute_clip_text_embedding("")

    # run the pipeline of models, for a single step
    output = sd(latent_noise, step=0, clip_text_embedding=text_embedding)

    assert output.shape == (1, 4, 64, 64)


@no_grad()
def test_sdxl_one_step(test_device: torch.device, test_dtype_fp32_bf16_fp16: torch.dtype) -> None:
    sd = StableDiffusion_XL(device=test_device, dtype=test_dtype_fp32_bf16_fp16)

    # prepare inputs
    latent_noise = torch.randn(1, 4, 128, 128, device=test_device, dtype=test_dtype_fp32_bf16_fp16)
    text_embedding, pooled_text_embedding = sd.compute_clip_text_embedding("")
    time_ids = sd.default_time_ids

    # run the pipeline of models, for a single step
    output = sd(
        latent_noise,
        step=0,
        clip_text_embedding=text_embedding,
        pooled_text_embedding=pooled_text_embedding,
        time_ids=time_ids,
    )

    assert output.shape == (1, 4, 128, 128)
