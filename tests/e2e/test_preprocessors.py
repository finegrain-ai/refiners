from pathlib import Path

import pytest
import torch
from PIL import Image
from tests.utils import ensure_similar_images

from refiners.fluxion.utils import image_to_tensor, no_grad, tensor_to_image
from refiners.foundationals.latent_diffusion.preprocessors.informative_drawings import InformativeDrawings


def _img_open(path: Path) -> Image.Image:
    return Image.open(path)  # type: ignore


@pytest.fixture(scope="module")
def diffusion_ref_path(test_e2e_path: Path) -> Path:
    return test_e2e_path / "test_diffusion_ref"


@pytest.fixture(scope="module")
def cutecat_init(diffusion_ref_path: Path) -> Image.Image:
    return _img_open(diffusion_ref_path / "cutecat_init.png").convert("RGB")


@pytest.fixture
def expected_image_informative_drawings(diffusion_ref_path: Path) -> Image.Image:
    return _img_open(diffusion_ref_path / "cutecat_guide_lineart.png").convert("RGB")


@pytest.fixture
def informative_drawings_model(
    controlnet_preprocessor_info_drawings_weights_path: Path,
    test_device: torch.device,
) -> InformativeDrawings:
    model = InformativeDrawings(device=test_device)
    model.load_from_safetensors(controlnet_preprocessor_info_drawings_weights_path)
    return model


@no_grad()
def test_preprocessor_informative_drawing(
    informative_drawings_model: InformativeDrawings,
    cutecat_init: Image.Image,
    expected_image_informative_drawings: Image.Image,
    test_device: torch.device,
):
    in_tensor = image_to_tensor(cutecat_init.convert("RGB"), device=test_device)
    out_tensor = informative_drawings_model(in_tensor)
    rgb_tensor = out_tensor.repeat(1, 3, 1, 1)  # grayscale to RGB
    image = tensor_to_image(rgb_tensor)
    ensure_similar_images(image, expected_image_informative_drawings)
