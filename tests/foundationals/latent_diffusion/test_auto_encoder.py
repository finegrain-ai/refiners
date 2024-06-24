from pathlib import Path
from warnings import warn

import pytest
import torch
from PIL import Image
from tests.utils import ensure_similar_images

from refiners.fluxion.utils import load_from_safetensors, no_grad
from refiners.foundationals.latent_diffusion.auto_encoder import LatentDiffusionAutoencoder


@pytest.fixture(scope="module")
def ref_path() -> Path:
    return Path(__file__).parent / "test_auto_encoder_ref"


@pytest.fixture(scope="module")
def lda(test_weights_path: Path, test_device: torch.device) -> LatentDiffusionAutoencoder:
    lda_weights = test_weights_path / "lda.safetensors"
    if not lda_weights.is_file():
        warn(f"could not find weights at {lda_weights}, skipping")
        pytest.skip(allow_module_level=True)
    encoder = LatentDiffusionAutoencoder(device=test_device)
    tensors = load_from_safetensors(lda_weights)
    encoder.load_state_dict(tensors)
    return encoder


@pytest.fixture(scope="module")
def sample_image(ref_path: Path) -> Image.Image:
    test_image = ref_path / "macaw.png"
    if not test_image.is_file():
        warn(f"could not reference image at {test_image}, skipping")
        pytest.skip(allow_module_level=True)
    img = Image.open(test_image)  # type: ignore
    assert img.size == (512, 512)
    return img


@no_grad()
def test_encode_decode_image(lda: LatentDiffusionAutoencoder, sample_image: Image.Image):
    encoded = lda.image_to_latents(sample_image)
    decoded = lda.latents_to_image(encoded)

    assert decoded.mode == "RGB"  # type: ignore

    # Ensure no saturation. The green channel (band = 1) must not max out.
    assert max(iter(decoded.getdata(band=1))) < 255  # type: ignore

    ensure_similar_images(sample_image, decoded, min_psnr=20, min_ssim=0.9)


@no_grad()
def test_encode_decode_images(lda: LatentDiffusionAutoencoder, sample_image: Image.Image):
    encoded = lda.images_to_latents([sample_image, sample_image])
    images = lda.latents_to_images(encoded)
    assert isinstance(images, list)
    assert len(images) == 2
    ensure_similar_images(sample_image, images[1], min_psnr=20, min_ssim=0.9)


@no_grad()
def test_tiled_autoencoder(lda: LatentDiffusionAutoencoder, sample_image: Image.Image):
    sample_image = sample_image.resize((2048, 2048))  # type: ignore

    with lda.tiled_inference(sample_image, tile_size=(512, 512)):
        encoded = lda.tiled_image_to_latents(sample_image)
        result = lda.tiled_latents_to_image(encoded)

    ensure_similar_images(sample_image, result, min_psnr=35, min_ssim=0.985)


@no_grad()
def test_tiled_autoencoder_rectangular_tiles(lda: LatentDiffusionAutoencoder, sample_image: Image.Image):
    sample_image = sample_image.resize((2048, 2048))  # type: ignore

    with lda.tiled_inference(sample_image, tile_size=(512, 1024)):
        encoded = lda.tiled_image_to_latents(sample_image)
        result = lda.tiled_latents_to_image(encoded)

    ensure_similar_images(sample_image, result, min_psnr=35, min_ssim=0.985)


@no_grad()
def test_tiled_autoencoder_large_tile(lda: LatentDiffusionAutoencoder, sample_image: Image.Image):
    sample_image = sample_image.resize((1024, 1024))  # type: ignore

    with lda.tiled_inference(sample_image, tile_size=(2048, 2048)):
        encoded = lda.tiled_image_to_latents(sample_image)
        result = lda.tiled_latents_to_image(encoded)

    ensure_similar_images(sample_image, result, min_psnr=34, min_ssim=0.975)


@no_grad()
def test_tiled_autoencoder_rectangular_image(lda: LatentDiffusionAutoencoder, sample_image: Image.Image):
    sample_image = sample_image.crop((0, 0, 300, 500))
    sample_image = sample_image.resize((sample_image.width * 4, sample_image.height * 4))  # type: ignore

    with lda.tiled_inference(sample_image, tile_size=(512, 512)):
        encoded = lda.tiled_image_to_latents(sample_image)
        result = lda.tiled_latents_to_image(encoded)

    ensure_similar_images(sample_image, result, min_psnr=37, min_ssim=0.985)


def test_value_error_tile_encode_no_context(lda: LatentDiffusionAutoencoder, sample_image: Image.Image) -> None:
    with pytest.raises(ValueError):
        lda.tiled_image_to_latents(sample_image)

    with pytest.raises(ValueError):
        lda.tiled_latents_to_image(torch.randn(1, 8, 16, 16, device=lda.device))
