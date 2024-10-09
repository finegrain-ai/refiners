import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from torch import Tensor

from refiners.fluxion.utils import manual_seed, no_grad
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL


@no_grad()
def test_text_encoder(
    diffusers_sd15_pipeline: StableDiffusionPipeline,
    refiners_sd15_text_encoder: CLIPTextEncoderL,
) -> None:
    """Compare our refiners implementation with the diffusers implementation."""
    manual_seed(seed=0)  # unnecessary, but just in case
    prompt = "A photo of a pizza."
    negative_prompt = ""
    atol = 1e-2  # FIXME: very high tolerance, figure out why

    (  # encode text prompts using diffusers pipeline
        diffusers_embeds,  # type: ignore
        diffusers_negative_embeds,  # type: ignore
    ) = diffusers_sd15_pipeline.encode_prompt(  # type: ignore
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        device=diffusers_sd15_pipeline.device,
    )
    assert isinstance(diffusers_embeds, Tensor)
    assert isinstance(diffusers_negative_embeds, Tensor)

    # encode text prompts using refiners model
    refiners_embeds = refiners_sd15_text_encoder(prompt)
    refiners_negative_embeds = refiners_sd15_text_encoder("")

    # check that the shapes are the same
    assert diffusers_embeds.shape == refiners_embeds.shape == (1, 77, 768)
    assert diffusers_negative_embeds.shape == refiners_negative_embeds.shape == (1, 77, 768)

    # check that the values are close
    assert torch.allclose(input=refiners_embeds, other=diffusers_embeds, atol=atol)
    assert torch.allclose(input=refiners_negative_embeds, other=diffusers_negative_embeds, atol=atol)


@no_grad()
def test_text_encoder_batched(refiners_sd15_text_encoder: CLIPTextEncoderL) -> None:
    """Check that encoding two prompts works as expected whether batched or not."""
    manual_seed(seed=0)  # unnecessary, but just in case
    prompt1 = "A photo of a pizza."
    prompt2 = "A giant duck."
    atol = 1e-6

    # encode the two prompts at once
    embeds_batched = refiners_sd15_text_encoder([prompt1, prompt2])
    assert embeds_batched.shape == (2, 77, 768)

    # encode the prompts one by one
    embeds_1 = refiners_sd15_text_encoder(prompt1)
    embeds_2 = refiners_sd15_text_encoder(prompt2)
    assert embeds_1.shape == embeds_2.shape == (1, 77, 768)

    # check that the values are close
    assert torch.allclose(input=embeds_1, other=embeds_batched[0].unsqueeze(0), atol=atol)
    assert torch.allclose(input=embeds_2, other=embeds_batched[1].unsqueeze(0), atol=atol)
