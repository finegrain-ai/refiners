from pathlib import Path
from typing import Any
from warnings import warn

import pytest
import torch
from huggingface_hub import hf_hub_download  # type: ignore

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

FLAVORS_MAP_REFINERS = {
    "dinov2_vits14": DINOv2_small,
    "dinov2_vits14_reg": DINOv2_small_reg,
    "dinov2_vitb14": DINOv2_base,
    "dinov2_vitb14_reg": DINOv2_base_reg,
    "dinov2_vitl14": DINOv2_large,
    "dinov2_vitl14_reg": DINOv2_large_reg,
    "dinov2_vitg14": DINOv2_giant,
    "dinov2_vitg14_reg": DINOv2_giant_reg,
}
FLAVORS_MAP_HUB = {
    "dinov2_vits14": "refiners/dinov2.small.patch_14",
    "dinov2_vits14_reg": "refiners/dinov2.small.patch_14.reg_4",
    "dinov2_vitb14": "refiners/dinov2.base.patch_14",
    "dinov2_vitb14_reg": "refiners/dinov2.base.patch_14.reg_4",
    "dinov2_vitl14": "refiners/dinov2.large.patch_14",
    "dinov2_vitl14_reg": "refiners/dinov2.large.patch_14.reg_4",
    "dinov2_vitg14": "refiners/dinov2.giant.patch_14",
    "dinov2_vitg14_reg": "refiners/dinov2.giant.patch_14.reg_4",
}


@pytest.fixture(scope="module", params=["float16", "bfloat16"])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    match request.param:
        case "float16":
            return torch.float16
        case "bfloat16":
            return torch.bfloat16
        case _ as dtype:
            raise ValueError(f"unsupported dtype: {dtype}")


@pytest.fixture(scope="module", params=[224, 518])
def resolution(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(scope="module", params=FLAVORS_MAP_REFINERS.keys())
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
    dinov2_small_unconverted_weights_path: Path,
    dinov2_small_reg4_unconverted_weights_path: Path,
    dinov2_base_unconverted_weights_path: Path,
    dinov2_base_reg4_unconverted_weights_path: Path,
    dinov2_large_unconverted_weights_path: Path,
    dinov2_large_reg4_unconverted_weights_path: Path,
    dinov2_giant_unconverted_weights_path: Path,
    dinov2_giant_reg4_unconverted_weights_path: Path,
    test_device: torch.device,
) -> torch.nn.Module:
    kwargs: dict[str, Any] = {}
    if "reg" not in flavor:
        kwargs["interpolate_offset"] = 0.0

    model: torch.nn.Module = torch.hub.load(  # type: ignore
        model=flavor,
        repo_or_dir=str(dinov2_repo_path),
        source="local",
        pretrained=False,  # to turn off automatic weights download (see load_state_dict below)
        **kwargs,
    )
    model = model.to(device=test_device)

    weight_map = {
        "dinov2_vits14": dinov2_small_unconverted_weights_path,
        "dinov2_vits14_reg": dinov2_small_reg4_unconverted_weights_path,
        "dinov2_vitb14": dinov2_base_unconverted_weights_path,
        "dinov2_vitb14_reg": dinov2_base_reg4_unconverted_weights_path,
        "dinov2_vitl14": dinov2_large_unconverted_weights_path,
        "dinov2_vitl14_reg": dinov2_large_reg4_unconverted_weights_path,
        "dinov2_vitg14": dinov2_giant_unconverted_weights_path,
        "dinov2_vitg14_reg": dinov2_giant_reg4_unconverted_weights_path,
    }
    weights_path = weight_map[flavor]

    model.load_state_dict(load_tensors(weights_path, device=test_device))
    assert isinstance(model, torch.nn.Module)
    return model


@pytest.fixture(scope="module")
def our_model(
    flavor: str,
    dinov2_small_weights_path: Path,
    dinov2_small_reg4_weights_path: Path,
    dinov2_base_weights_path: Path,
    dinov2_base_reg4_weights_path: Path,
    dinov2_large_weights_path: Path,
    dinov2_large_reg4_weights_path: Path,
    dinov2_giant_weights_path: Path,
    dinov2_giant_reg4_weights_path: Path,
    test_device: torch.device,
) -> ViT:
    weight_map = {
        "dinov2_vits14": dinov2_small_weights_path,
        "dinov2_vits14_reg": dinov2_small_reg4_weights_path,
        "dinov2_vitb14": dinov2_base_weights_path,
        "dinov2_vitb14_reg": dinov2_base_reg4_weights_path,
        "dinov2_vitl14": dinov2_large_weights_path,
        "dinov2_vitl14_reg": dinov2_large_reg4_weights_path,
        "dinov2_vitg14": dinov2_giant_weights_path,
        "dinov2_vitg14_reg": dinov2_giant_reg4_weights_path,
    }
    weights_path = weight_map[flavor]

    model = FLAVORS_MAP_REFINERS[flavor](device=test_device)
    tensors = load_from_safetensors(weights_path)
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
        size=(1, 3, resolution, resolution),
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
def test_dinov2(
    resolution: int,
    test_dtype_fp32_bf16_fp16: torch.dtype,
    test_device: torch.device,
) -> None:
    if test_device.type == "cpu":
        warn("not running on CPU, skipping")
        pytest.skip()

    model = DINOv2_small(device=test_device, dtype=test_dtype_fp32_bf16_fp16)

    manual_seed(2)
    input_data = torch.randn(
        size=(1, 3, resolution, resolution),
        device=test_device,
        dtype=test_dtype_fp32_bf16_fp16,
    )

    output = model(input_data)
    sequence_length = (resolution // model.patch_size) ** 2 + 1
    assert output.shape == (1, sequence_length, model.embedding_dim)
    assert output.dtype == test_dtype_fp32_bf16_fp16


@no_grad()
def test_dinov2_batch_size(
    resolution: int,
    test_device: torch.device,
) -> None:
    model = DINOv2_small(device=test_device)

    batch_size = 4
    manual_seed(2)
    input_data = torch.randn(
        size=(batch_size, 3, resolution, resolution),
        device=test_device,
    )

    output = model(input_data)
    sequence_length = (resolution // model.patch_size) ** 2 + 1
    assert output.shape == (batch_size, sequence_length, model.embedding_dim)
