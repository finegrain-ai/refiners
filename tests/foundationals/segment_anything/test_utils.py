import pytest
import torch
from PIL import Image

from refiners.foundationals.segment_anything.utils import (
    compute_scaled_size,
    image_to_scaled_tensor,
    pad_image_tensor,
    preprocess_image,
)


@pytest.fixture
def image_encoder_resolution() -> int:
    return 1024


def test_compute_scaled_size(image_encoder_resolution: int) -> None:
    w, h = (1536, 768)
    scaled_size = compute_scaled_size((h, w), image_encoder_resolution)

    assert scaled_size == (512, 1024)


def test_rgb_image_to_scaled_tensor() -> None:
    image = Image.new("RGB", (1536, 768))
    tensor = image_to_scaled_tensor(image, (512, 1024))
    assert tensor.shape == (1, 3, 512, 1024)


def test_grayscale_image_to_scaled_tensor() -> None:
    image = Image.new("L", (1536, 768))
    tensor = image_to_scaled_tensor(image, (512, 1024))
    assert tensor.shape == (1, 1, 512, 1024)


def test_preprocess_image(image_encoder_resolution: int) -> None:
    image = Image.new("RGB", (1536, 768))
    preprocessed = preprocess_image(image, image_encoder_resolution)

    assert preprocessed.shape == (1, 3, 1024, 1024)


def test_pad_image_tensor(image_encoder_resolution: int) -> None:
    w, h = (1536, 768)
    image = Image.new("RGB", (w, h), color="white")
    scaled_size = compute_scaled_size((h, w), image_encoder_resolution)
    scaled_image_tensor = image_to_scaled_tensor(image, scaled_size)
    padded_image_tensor = pad_image_tensor(scaled_image_tensor, scaled_size, image_encoder_resolution)

    assert padded_image_tensor.shape == (1, 3, 1024, 1024)
    assert torch.all(padded_image_tensor[:, :, 512:, :] == 0)
