from typing import Any
from pathlib import Path
from warnings import warn
import pytest
import torch

from refiners.foundationals.latent_diffusion.sdxl_unet import SDXLUNet
from refiners.fluxion.utils import compare_models


@pytest.fixture(scope="module")
def stabilityai_sdxl_base_path(test_weights_path: Path) -> Path:
    r = test_weights_path / "stabilityai" / "stable-diffusion-xl-base-0.9"
    if not r.is_dir():
        warn(f"could not find Stability SDXL base weights at {r}, skipping")
        pytest.skip(allow_module_level=True)
    return r


@pytest.fixture(scope="module")
def diffusers_sdxl(stabilityai_sdxl_base_path: Path) -> Any:
    from diffusers import DiffusionPipeline  # type: ignore

    return DiffusionPipeline.from_pretrained(pretrained_model_name_or_path=stabilityai_sdxl_base_path)  # type: ignore


@pytest.fixture(scope="module")
def diffusers_sdxl_unet(diffusers_sdxl: Any) -> Any:
    return diffusers_sdxl.unet


@pytest.fixture(scope="module")
def sdxl_unet_weights_std(test_weights_path: Path) -> Path:
    unet_weights_std = test_weights_path / "sdxl-unet.safetensors"
    if not unet_weights_std.is_file():
        warn(message=f"could not find weights at {unet_weights_std}, skipping")
        pytest.skip(allow_module_level=True)
    return unet_weights_std


@pytest.fixture(scope="module")
def refiners_sdxl_unet(sdxl_unet_weights_std: Path) -> SDXLUNet:
    unet = SDXLUNet(in_channels=4)
    unet.load_from_safetensors(tensors_path=sdxl_unet_weights_std)
    return unet


@torch.no_grad()
def test_sdxl_unet(diffusers_sdxl_unet: Any, refiners_sdxl_unet: SDXLUNet) -> None:
    torch.manual_seed(seed=0)  # type: ignore
    x = torch.randn(1, 4, 32, 32)
    timestep = torch.tensor(data=[0])
    clip_text_embeddings = torch.randn(1, 77, 2048)
    added_cond_kwargs = {"text_embeds": torch.randn(1, 1280), "time_ids": torch.randn(1, 6)}
    source_args = (x, timestep, clip_text_embeddings, None, None, None, None, added_cond_kwargs)

    refiners_sdxl_unet.set_timestep(timestep=timestep)
    refiners_sdxl_unet.set_clip_text_embedding(clip_text_embedding=clip_text_embeddings)
    refiners_sdxl_unet.set_time_ids(time_ids=added_cond_kwargs["time_ids"])
    refiners_sdxl_unet.set_pooled_text_embedding(pooled_text_embedding=added_cond_kwargs["text_embeds"])
    target_args = (x,)

    assert compare_models(
        source_model=diffusers_sdxl_unet,
        target_model=refiners_sdxl_unet,
        source_args=source_args,
        target_args=target_args,
        threshold=1e-2,
    )
