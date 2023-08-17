from typing import Any, Protocol, cast
from pathlib import Path
from warnings import warn
import pytest
import torch
from torch import Tensor

from refiners.fluxion.utils import manual_seed
import refiners.fluxion.layers as fl
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderG, CLIPTextEncoderL
from refiners.foundationals.latent_diffusion.sdxl_text_encoder import DoubleTextEncoder


class DiffusersSDXL(Protocol):
    unet: fl.Module
    text_encoder: fl.Module
    text_encoder_2: fl.Module
    tokenizer: fl.Module
    tokenizer_2: fl.Module
    vae: fl.Module

    def __call__(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
        ...

    def encode_prompt(
        self,
        prompt: str,
        prompt_2: str | None = None,
        negative_prompt: str | None = None,
        negative_prompt_2: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ...


@pytest.fixture(scope="module")
def stabilityai_sdxl_base_path(test_weights_path: Path) -> Path:
    r = test_weights_path / "stabilityai" / "stable-diffusion-xl-base-1.0"
    if not r.is_dir():
        warn(message=f"could not find Stability SDXL base weights at {r}, skipping")
        pytest.skip(allow_module_level=True)
    return r


@pytest.fixture(scope="module")
def diffusers_sdxl(stabilityai_sdxl_base_path: Path) -> Any:
    from diffusers import DiffusionPipeline  # type: ignore

    return DiffusionPipeline.from_pretrained(pretrained_model_name_or_path=stabilityai_sdxl_base_path)  # type: ignore


@pytest.fixture(scope="module")
def double_text_encoder(test_weights_path: Path) -> DoubleTextEncoder:
    text_encoder_l = CLIPTextEncoderL()
    text_encoder_g_with_projection = CLIPTextEncoderG()
    text_encoder_g_with_projection.append(module=fl.Linear(in_features=1280, out_features=1280, bias=False))

    text_encoder_l_path = test_weights_path / "CLIPTextEncoderL.safetensors"
    text_encdoer_g_path = test_weights_path / "CLIPTextEncoderGWithProjection.safetensors"

    text_encoder_l.load_from_safetensors(tensors_path=text_encoder_l_path)
    text_encoder_g_with_projection.load_from_safetensors(tensors_path=text_encdoer_g_path)

    linear = text_encoder_g_with_projection.pop(index=-1)
    assert isinstance(linear, fl.Linear)

    double_text_encoder = DoubleTextEncoder(
        text_encoder_l=text_encoder_l, text_encoder_g=text_encoder_g_with_projection, projection=linear
    )

    return double_text_encoder


@torch.no_grad()
def test_double_text_encoder(diffusers_sdxl: DiffusersSDXL, double_text_encoder: DoubleTextEncoder) -> None:
    manual_seed(seed=0)
    prompt = "A photo of a pizza."

    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
        diffusers_sdxl.encode_prompt(prompt=prompt, negative_prompt="")
    )

    double_embedding, pooled_embedding = double_text_encoder(prompt)

    assert double_embedding.shape == torch.Size([1, 77, 2048])
    assert pooled_embedding.shape == torch.Size([1, 1280])

    embedding_1, embedding_2 = cast(
        tuple[Tensor, Tensor], prompt_embeds.split(split_size=[768, 1280], dim=-1)  # type: ignore
    )

    rembedding_1, rembedding_2 = cast(
        tuple[Tensor, Tensor], double_embedding.split(split_size=[768, 1280], dim=-1)  # type: ignore
    )

    assert torch.allclose(input=embedding_1, other=rembedding_1, rtol=1e-3, atol=1e-3)
    assert torch.allclose(input=embedding_2, other=rembedding_2, rtol=1e-3, atol=1e-3)
    assert torch.allclose(input=pooled_embedding, other=pooled_prompt_embeds, rtol=1e-3, atol=1e-3)

    negative_double_embedding, negative_pooled_embedding = double_text_encoder("")

    assert torch.allclose(input=negative_double_embedding, other=negative_prompt_embeds, rtol=1e-3, atol=1e-3)
    assert torch.allclose(input=negative_pooled_embedding, other=negative_pooled_prompt_embeds, rtol=1e-3, atol=1e-3)
