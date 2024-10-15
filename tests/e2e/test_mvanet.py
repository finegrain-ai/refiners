from pathlib import Path
from warnings import warn

import pytest
import torch
from PIL import Image
from tests.utils import ensure_similar_images

from refiners.fluxion.utils import image_to_tensor, no_grad, normalize, tensor_to_image
from refiners.foundationals.swin.mvanet import MVANet


@pytest.fixture(scope="module")
def ref_path(test_e2e_path: Path) -> Path:
    return test_e2e_path / "test_mvanet_ref"


@pytest.fixture(scope="module")
def ref_cactus(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "cactus.png").convert("RGB")


@pytest.fixture
def expected_cactus_mask(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_cactus_mask.png")


@pytest.fixture
def mvanet_model(mvanet_weights_path: Path, test_device: torch.device) -> MVANet:
    model = MVANet(device=test_device).eval()  # .eval() is important!
    model.load_from_safetensors(mvanet_weights_path)
    return model


@no_grad()
def test_mvanet(
    mvanet_model: MVANet,
    ref_cactus: Image.Image,
    expected_cactus_mask: Image.Image,
    test_device: torch.device,
):
    in_t = image_to_tensor(ref_cactus.resize((1024, 1024), Image.Resampling.BILINEAR)).squeeze()
    in_t = normalize(in_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)
    prediction: torch.Tensor = mvanet_model(in_t.to(test_device)).sigmoid()
    cactus_mask = tensor_to_image(prediction).resize(ref_cactus.size, Image.Resampling.BILINEAR)
    ensure_similar_images(cactus_mask.convert("RGB"), expected_cactus_mask.convert("RGB"))


@no_grad()
def test_mvanet_to(
    mvanet_weights_path: Path,
    ref_cactus: Image.Image,
    expected_cactus_mask: Image.Image,
    test_device: torch.device,
):
    if test_device.type == "cpu":
        warn("not running on CPU, skipping")
        pytest.skip()

    model = MVANet(device=torch.device("cpu")).eval()
    model.load_from_safetensors(mvanet_weights_path)
    model.to(test_device)

    in_t = image_to_tensor(ref_cactus.resize((1024, 1024), Image.Resampling.BILINEAR)).squeeze()
    in_t = normalize(in_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)
    prediction: torch.Tensor = model(in_t.to(test_device)).sigmoid()
    cactus_mask = tensor_to_image(prediction).resize(ref_cactus.size, Image.Resampling.BILINEAR)
    ensure_similar_images(cactus_mask.convert("RGB"), expected_cactus_mask.convert("RGB"))
