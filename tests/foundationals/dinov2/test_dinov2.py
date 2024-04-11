from pathlib import Path
from typing import Any
from warnings import warn

import pytest
import torch

from refiners.fluxion.utils import load_from_safetensors, load_tensors, manual_seed, no_grad
from refiners.foundationals.dinov2.dinov2 import (
    DINOv2_base,
    DINOv2_base_reg,
    DINOv2_giant,
    DINOv2_giant_reg,
    DINOv2_large,
    DINOv2_large_reg,
    DINOv2_small,
    DINOv2_small_reg,
)
from refiners.foundationals.dinov2.vit import ViT

FLAVORS_MAP = {
    "dinov2_vits14": DINOv2_small,
    "dinov2_vits14_reg": DINOv2_small_reg,
    "dinov2_vitb14": DINOv2_base,
    "dinov2_vitb14_reg": DINOv2_base_reg,
    "dinov2_vitl14": DINOv2_large,
    "dinov2_vitl14_reg": DINOv2_large_reg,
    "dinov2_vitg14": DINOv2_giant,
    "dinov2_vitg14_reg": DINOv2_giant_reg,
}


@pytest.fixture(scope="module", params=[224, 518])
def resolution(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(scope="module", params=FLAVORS_MAP.keys())
def flavor(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def dinov2_repo_path(test_repos_path: Path) -> Path:
    repo = test_repos_path / "dinov2"
    if not repo.exists():
        warn(f"could not find DINOv2 GitHub repo at {repo}, skipping")
        pytest.skip(allow_module_level=True)
    return repo


@pytest.fixture(scope="module")
def ref_model(
    flavor: str,
    dinov2_repo_path: Path,
    test_weights_path: Path,
    test_device: torch.device,
) -> torch.nn.Module:
    kwargs: dict[str, Any] = {}
    if "reg" not in flavor:
        kwargs["interpolate_offset"] = 0.0

    model = torch.hub.load(  # type: ignore
        model=flavor,
        repo_or_dir=str(dinov2_repo_path),
        source="local",
        pretrained=False,  # to turn off automatic weights download (see load_state_dict below)
        **kwargs,
    ).to(device=test_device)

    flavor = flavor.replace("_reg", "_reg4")
    weights = test_weights_path / f"{flavor}_pretrain.pth"
    if not weights.is_file():
        warn(f"could not find weights at {weights}, skipping")
        pytest.skip(allow_module_level=True)
    model.load_state_dict(load_tensors(weights, device=test_device))

    assert isinstance(model, torch.nn.Module)
    return model


@pytest.fixture(scope="module")
def our_model(
    test_weights_path: Path,
    flavor: str,
    test_device: torch.device,
) -> ViT:
    model = FLAVORS_MAP[flavor](device=test_device)

    flavor = flavor.replace("_reg", "_reg4")
    weights = test_weights_path / f"{flavor}_pretrain.safetensors"
    if not weights.is_file():
        warn(f"could not find weights at {weights}, skipping")
        pytest.skip(allow_module_level=True)

    tensors = load_from_safetensors(weights)
    model.load_state_dict(tensors)

    return model


@no_grad()
def test_dinov2_facebook_weights(
    ref_model: torch.nn.Module,
    our_model: ViT,
    resolution: int,
    test_device: torch.device,
) -> None:
    manual_seed(2)
    input_data = torch.randn(
        (1, 3, resolution, resolution),
        device=test_device,
    )

    ref_output = ref_model(input_data, is_training=True)
    ref_cls = ref_output["x_norm_clstoken"]
    ref_reg = ref_output["x_norm_regtokens"]
    ref_patch = ref_output["x_norm_patchtokens"]

    our_output = our_model(input_data)
    our_cls = our_output[:, 0]
    our_reg = our_output[:, 1 : our_model.num_registers + 1]
    our_patch = our_output[:, our_model.num_registers + 1 :]

    assert torch.allclose(ref_cls, our_cls, atol=1e-4)
    assert torch.allclose(ref_reg, our_reg, atol=1e-4)
    assert torch.allclose(ref_patch, our_patch, atol=3e-3)


@no_grad()
def test_dinov2_float16(
    resolution: int,
    test_device: torch.device,
) -> None:
    if test_device.type == "cpu":
        warn("not running on CPU, skipping")
        pytest.skip()

    model = DINOv2_small(device=test_device, dtype=torch.float16)

    manual_seed(2)
    input_data = torch.randn(
        (1, 3, resolution, resolution),
        device=test_device,
        dtype=torch.float16,
    )

    output = model(input_data)
    sequence_length = (resolution // model.patch_size) ** 2 + 1
    assert output.shape == (1, sequence_length, model.embedding_dim)
    assert output.dtype == torch.float16


@no_grad()
def test_dinov2_batch_size(
    resolution: int,
    test_device: torch.device,
) -> None:
    model = DINOv2_small(device=test_device)

    batch_size = 4
    manual_seed(2)
    input_data = torch.randn(
        (batch_size, 3, resolution, resolution),
        device=test_device,
    )

    output = model(input_data)
    sequence_length = (resolution // model.patch_size) ** 2 + 1
    assert output.shape == (batch_size, sequence_length, model.embedding_dim)
