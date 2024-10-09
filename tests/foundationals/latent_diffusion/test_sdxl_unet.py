from typing import Any

import pytest
import torch

from refiners.conversion.model_converter import ConversionStage, ModelConverter
from refiners.fluxion.utils import manual_seed, no_grad
from refiners.foundationals.latent_diffusion.stable_diffusion_xl import SDXLUNet


@pytest.fixture(scope="module")
def refiners_sdxl_unet() -> SDXLUNet:
    unet = SDXLUNet(in_channels=4)
    return unet


@no_grad()
def test_sdxl_unet(
    diffusers_sdxl_unet: Any,
    refiners_sdxl_unet: SDXLUNet,
) -> None:
    source = diffusers_sdxl_unet
    target = refiners_sdxl_unet

    manual_seed(seed=0)
    x = torch.randn(1, 4, 32, 32)
    timestep = torch.tensor(data=[0])
    clip_text_embeddings = torch.randn(1, 77, 2048)
    added_cond_kwargs = {"text_embeds": torch.randn(1, 1280), "time_ids": torch.randn(1, 6)}

    target_args = (x,)
    source_args = {
        "positional": (x, timestep, clip_text_embeddings),
        "keyword": {"added_cond_kwargs": added_cond_kwargs},
    }

    old_forward = target.forward

    def forward_with_context(self: Any, *args: Any, **kwargs: Any) -> Any:
        target.set_timestep(timestep=timestep)
        target.set_clip_text_embedding(clip_text_embedding=clip_text_embeddings)
        target.set_time_ids(time_ids=added_cond_kwargs["time_ids"])
        target.set_pooled_text_embedding(pooled_text_embedding=added_cond_kwargs["text_embeds"])
        return old_forward(self, *args, **kwargs)

    target.forward = forward_with_context

    converter = ModelConverter(source_model=source, target_model=target, verbose=True, threshold=1e-2)

    assert converter.run(
        source_args=source_args,
        target_args=target_args,
    )
    assert converter.stage == ConversionStage.MODELS_OUTPUT_AGREE
