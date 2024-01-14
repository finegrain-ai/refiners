from math import isclose
from pathlib import Path
from warnings import warn

import pytest
import torch
from transformers import AutoModel  # type: ignore
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model  # type: ignore

from refiners.fluxion.utils import load_from_safetensors, manual_seed, no_grad
from refiners.foundationals.dinov2 import (
    DINOv2_base,
    DINOv2_base_reg,
    DINOv2_large,
    DINOv2_large_reg,
    DINOv2_small,
    DINOv2_small_reg,
)
from refiners.foundationals.dinov2.vit import ViT

FLAVORS = [
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vits14_reg4",
    "dinov2_vitb14_reg4",
    "dinov2_vitl14_reg4",
]


@pytest.fixture(scope="module", params=FLAVORS)
def flavor(request: pytest.FixtureRequest) -> str:
    return request.param


# Temporary: see comments in `test_encoder_only`
@pytest.fixture(scope="module")
def seed_expected_norm(flavor: str) -> tuple[int, float]:
    match flavor:
        case "dinov2_vits14":
            return (42, 1977.9213867)
        case "dinov2_vitb14":
            return (42, 1902.6384277)
        case "dinov2_vitl14":
            return (42, 1763.9187011)
        case "dinov2_vits14_reg4":
            return (42, 989.2380981)
        case "dinov2_vitb14_reg4":
            return (42, 974.4362182)
        case "dinov2_vitl14_reg4":
            return (42, 924.8797607)
        case _:
            raise ValueError(f"Unexpected DINOv2 flavor: {flavor}")


@pytest.fixture(scope="module")
def our_backbone(test_weights_path: Path, flavor: str, test_device: torch.device) -> ViT:
    weights = test_weights_path / f"{flavor}_pretrain.safetensors"
    if not weights.is_file():
        warn(f"could not find weights at {weights}, skipping")
        pytest.skip(allow_module_level=True)
    match flavor:
        case "dinov2_vits14":
            backbone = DINOv2_small(device=test_device)
        case "dinov2_vitb14":
            backbone = DINOv2_base(device=test_device)
        case "dinov2_vitl14":
            backbone = DINOv2_large(device=test_device)
        case "dinov2_vits14_reg4":
            backbone = DINOv2_small_reg(device=test_device)
        case "dinov2_vitb14_reg4":
            backbone = DINOv2_base_reg(device=test_device)
        case "dinov2_vitl14_reg4":
            backbone = DINOv2_large_reg(device=test_device)
        case _:
            raise ValueError(f"Unexpected DINOv2 flavor: {flavor}")
    tensors = load_from_safetensors(weights)
    backbone.load_state_dict(tensors)
    return backbone


@pytest.fixture(scope="module")
def dinov2_weights_path(test_weights_path: Path, flavor: str):
    # TODO: At the time of writing, those are not yet supported in transformers
    # (https://github.com/huggingface/transformers/issues/27379). Alternatively, it is also possible to use
    # facebookresearch/dinov2 directly (https://github.com/finegrain-ai/refiners/pull/132).
    if flavor.endswith("_reg4"):
        warn(f"DINOv2 with registers are not yet supported in Hugging Face, skipping")
        pytest.skip(allow_module_level=True)
    match flavor:
        case "dinov2_vits14":
            name = "dinov2-small"
        case "dinov2_vitb14":
            name = "dinov2-base"
        case "dinov2_vitl14":
            name = "dinov2-large"
        case _:
            raise ValueError(f"Unexpected DINOv2 flavor: {flavor}")
    r = test_weights_path / "facebook" / name
    if not r.is_dir():
        warn(f"could not find DINOv2 weights at {r}, skipping")
        pytest.skip(allow_module_level=True)
    return r


@pytest.fixture(scope="module")
def ref_backbone(dinov2_weights_path: Path, test_device: torch.device) -> Dinov2Model:
    backbone = AutoModel.from_pretrained(dinov2_weights_path)  # type: ignore
    assert isinstance(backbone, Dinov2Model)
    return backbone.to(test_device)  # type: ignore


def test_encoder(
    ref_backbone: Dinov2Model,
    our_backbone: ViT,
    test_device: torch.device,
):
    manual_seed(42)

    # Position encoding interpolation [1] at runtime is not supported yet. So stick to the default image resolution
    # e.g. using (224, 224) pixels as input would give a runtime error (sequence size mismatch)
    # [1]: https://github.com/facebookresearch/dinov2/blob/2302b6b/dinov2/models/vision_transformer.py#L179
    assert our_backbone.image_size == 518

    x = torch.randn(1, 3, 518, 518).to(test_device)

    with no_grad():
        ref_features = ref_backbone(x).last_hidden_state
        our_features = our_backbone(x)

    assert (our_features - ref_features).abs().max() < 1e-3


# Mainly for DINOv2 + registers coverage (this test can be removed once `test_encoder` supports all flavors)
def test_encoder_only(
    our_backbone: ViT,
    seed_expected_norm: tuple[int, float],
    test_device: torch.device,
):
    seed, expected_norm = seed_expected_norm
    manual_seed(seed)

    x = torch.randn(1, 3, 518, 518).to(test_device)

    our_features = our_backbone(x)

    assert isclose(our_features.norm().item(), expected_norm, rel_tol=1e-04)
