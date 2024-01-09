from pathlib import Path
from warnings import warn

import pytest
import torch
from transformers import CLIPVisionModelWithProjection  # type: ignore

from refiners.fluxion.utils import load_from_safetensors, no_grad
from refiners.foundationals.clip.image_encoder import CLIPImageEncoderH


@pytest.fixture(scope="module")
def our_encoder(test_weights_path: Path, test_device: torch.device) -> CLIPImageEncoderH:
    weights = test_weights_path / "CLIPImageEncoderH.safetensors"
    if not weights.is_file():
        warn(f"could not find weights at {weights}, skipping")
        pytest.skip(allow_module_level=True)
    encoder = CLIPImageEncoderH(device=test_device)
    tensors = load_from_safetensors(weights)
    encoder.load_state_dict(tensors)
    return encoder


@pytest.fixture(scope="module")
def stabilityai_unclip_weights_path(test_weights_path: Path):
    r = test_weights_path / "stabilityai" / "stable-diffusion-2-1-unclip"
    if not r.is_dir():
        warn(f"could not find Stability AI weights at {r}, skipping")
        pytest.skip(allow_module_level=True)
    return r


@pytest.fixture(scope="module")
def ref_encoder(stabilityai_unclip_weights_path: Path, test_device: torch.device) -> CLIPVisionModelWithProjection:
    return CLIPVisionModelWithProjection.from_pretrained(stabilityai_unclip_weights_path, subfolder="image_encoder").to(  # type: ignore
        test_device  # type: ignore
    )


def test_encoder(
    ref_encoder: CLIPVisionModelWithProjection,
    our_encoder: CLIPImageEncoderH,
    test_device: torch.device,
):
    x = torch.randn(1, 3, 224, 224).to(test_device)

    with no_grad():
        ref_embeddings = ref_encoder(x).image_embeds
        our_embeddings = our_encoder(x)

    assert ref_embeddings.shape == (1, 1024)
    assert our_embeddings.shape == (1, 1024)

    assert (our_embeddings - ref_embeddings).abs().max() < 0.01
