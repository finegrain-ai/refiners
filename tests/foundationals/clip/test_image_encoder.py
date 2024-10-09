from pathlib import Path

import pytest
import torch
from transformers import CLIPVisionModelWithProjection  # type: ignore

from refiners.fluxion.utils import load_from_safetensors, no_grad
from refiners.foundationals.clip.image_encoder import CLIPImageEncoderH


@pytest.fixture(scope="module")
def our_encoder(
    clip_image_encoder_huge_weights_path: Path,
    test_device: torch.device,
    test_dtype_fp32_bf16_fp16: torch.dtype,
) -> CLIPImageEncoderH:
    encoder = CLIPImageEncoderH(device=test_device, dtype=test_dtype_fp32_bf16_fp16)
    tensors = load_from_safetensors(clip_image_encoder_huge_weights_path)
    encoder.load_state_dict(tensors)
    return encoder


@pytest.fixture(scope="module")
def ref_encoder(
    unclip21_transformers_stabilityai_path: str,
    test_device: torch.device,
    test_dtype_fp32_bf16_fp16: torch.dtype,
    use_local_weights: bool,
) -> CLIPVisionModelWithProjection:
    return CLIPVisionModelWithProjection.from_pretrained(  # type: ignore
        unclip21_transformers_stabilityai_path,
        local_files_only=use_local_weights,
        subfolder="image_encoder",
    ).to(device=test_device, dtype=test_dtype_fp32_bf16_fp16)  # type: ignore


@no_grad()
@pytest.mark.flaky(reruns=3)
def test_encoder(
    ref_encoder: CLIPVisionModelWithProjection,
    our_encoder: CLIPImageEncoderH,
):
    assert ref_encoder.dtype == our_encoder.dtype
    assert ref_encoder.device == our_encoder.device
    x = torch.randn((1, 3, 224, 224), dtype=ref_encoder.dtype, device=ref_encoder.device)

    ref_embeddings = ref_encoder(x).image_embeds
    our_embeddings = our_encoder(x)

    assert ref_embeddings.shape == (1, 1024)
    assert our_embeddings.shape == (1, 1024)

    assert torch.allclose(our_embeddings, ref_embeddings, atol=0.05)
