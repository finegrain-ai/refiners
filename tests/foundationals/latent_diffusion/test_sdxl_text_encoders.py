import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from torch import Tensor

from refiners.fluxion.utils import manual_seed, no_grad
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.text_encoder import DoubleTextEncoder


@no_grad()
def test_double_text_encoder(
    diffusers_sdxl_pipeline: StableDiffusionXLPipeline,
    refiners_sdxl_text_encoder: DoubleTextEncoder,
) -> None:
    """Compare our refiners implementation with the diffusers implementation."""
    manual_seed(seed=0)  # unnecessary, but just in case
    prompt = "A photo of a pizza."
    negative_prompt = ""
    atol = 1e-6

    (  # encode text prompts using diffusers pipeline
        diffusers_embeds,
        diffusers_negative_embeds,  # type: ignore
        diffusers_pooled_embeds,  # type: ignore
        diffusers_negative_pooled_embeds,  # type: ignore
    ) = diffusers_sdxl_pipeline.encode_prompt(prompt=prompt, negative_prompt=negative_prompt)
    assert diffusers_negative_embeds is not None
    assert isinstance(diffusers_pooled_embeds, Tensor)
    assert isinstance(diffusers_negative_pooled_embeds, Tensor)

    # encode text prompts using refiners model
    refiners_embeds, refiners_pooled_embeds = refiners_sdxl_text_encoder(prompt)
    refiners_negative_embeds, refiners_negative_pooled_embeds = refiners_sdxl_text_encoder("")

    # check that the shapes are the same
    assert diffusers_embeds.shape == refiners_embeds.shape == (1, 77, 2048)
    assert diffusers_pooled_embeds.shape == refiners_pooled_embeds.shape == (1, 1280)
    assert diffusers_negative_embeds.shape == refiners_negative_embeds.shape == (1, 77, 2048)
    assert diffusers_negative_pooled_embeds.shape == refiners_negative_pooled_embeds.shape == (1, 1280)

    # check that the values are close
    assert torch.allclose(input=refiners_embeds, other=diffusers_embeds, atol=atol)
    assert torch.allclose(input=refiners_negative_embeds, other=diffusers_negative_embeds, atol=atol)
    assert torch.allclose(input=refiners_negative_pooled_embeds, other=diffusers_negative_pooled_embeds, atol=atol)
    assert torch.allclose(input=refiners_pooled_embeds, other=diffusers_pooled_embeds, atol=atol)


@no_grad()
def test_double_text_encoder_batched(refiners_sdxl_text_encoder: DoubleTextEncoder) -> None:
    """Check that encoding two prompts works as expected whether batched or not."""
    manual_seed(seed=0)  # unnecessary, but just in case
    prompt1 = "A photo of a pizza."
    prompt2 = "A giant duck."
    atol = 1e-6

    # encode the two prompts at once
    embeds_batched, pooled_embeds_batched = refiners_sdxl_text_encoder([prompt1, prompt2])
    assert embeds_batched.shape == (2, 77, 2048)
    assert pooled_embeds_batched.shape == (2, 1280)

    # encode the prompts one by one
    embeds_1, pooled_embeds_1 = refiners_sdxl_text_encoder(prompt1)
    embeds_2, pooled_embeds_2 = refiners_sdxl_text_encoder(prompt2)
    assert embeds_1.shape == embeds_2.shape == (1, 77, 2048)
    assert pooled_embeds_1.shape == pooled_embeds_2.shape == (1, 1280)

    # check that the values are close
    assert torch.allclose(input=embeds_1, other=embeds_batched[0].unsqueeze(0), atol=atol)
    assert torch.allclose(input=pooled_embeds_1, other=pooled_embeds_batched[0].unsqueeze(0), atol=atol)
    assert torch.allclose(input=embeds_2, other=embeds_batched[1].unsqueeze(0), atol=atol)
    assert torch.allclose(input=pooled_embeds_2, other=pooled_embeds_batched[1].unsqueeze(0), atol=atol)
