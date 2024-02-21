from pathlib import Path
from warnings import warn

import pytest
import torch
from transformers import (  # type: ignore
    AutoModel,  # type: ignore
    VitMatteForImageMatting,  # type: ignore
)

from refiners.fluxion.utils import load_from_safetensors, manual_seed, no_grad
from refiners.foundationals.vit_matte import ViTMatteH


@pytest.fixture(scope="module")
def our_backbone(test_weights_path: Path, test_device: torch.device) -> ViTMatteH:
    weights = test_weights_path / f"vitmatte_h.safetensors"
    if not weights.is_file():
        warn(f"could not find weights at {weights}, skipping")
        pytest.skip(allow_module_level=True)
    backbone = ViTMatteH(device=test_device)
    tensors = load_from_safetensors(weights)
    backbone.load_state_dict(tensors)
    return backbone


@pytest.fixture(scope="module")
def vit_matte_weights_path(test_weights_path: Path):
    r = test_weights_path
    if not r.is_dir():
        pytest.skip(allow_module_level=True)
    return r


@pytest.fixture(scope="module")
def ref_backbone(vit_matte_weights_path: Path, test_device: torch.device) -> VitMatteForImageMatting:
    backbone = AutoModel.from_pretrained(vit_matte_weights_path)  # type: ignore
    assert isinstance(backbone, VitMatteForImageMatting)
    return backbone.to(test_device)  # type: ignore


def test_encoder(
    ref_backbone: VitMatteForImageMatting,
    our_backbone: ViTMatteH,
    test_device: torch.device,
):
    manual_seed(42)

    x = torch.randn(1, 4, 512, 512).to(test_device)

    with no_grad():
        ref_features = ref_backbone(x).last_hidden_state
        our_features = our_backbone(x)

    assert (our_features - ref_features).abs().max() < 1e-3
