from pathlib import Path
from warnings import warn

import pytest
import torch
from PIL import Image
from tests.utils import ensure_similar_images

from refiners.fluxion.utils import no_grad
from refiners.foundationals.latent_diffusion.auto_encoder import LatentDiffusionAutoencoder


@pytest.fixture(scope="module")
def sample_image() -> Image.Image:
    test_image = Path(__file__).parent / "test_auto_encoder_ref" / "macaw.png"
    if not test_image.is_file():
        warn(f"could not reference image at {test_image}, skipping")
        pytest.skip(allow_module_level=True)
    img = Image.open(test_image)  # type: ignore
    assert img.size == (512, 512)
    return img


@pytest.fixture(scope="module")
def autoencoder(
    refiners_autoencoder: LatentDiffusionAutoencoder,
    test_device: torch.device,
) -> LatentDiffusionAutoencoder:
    return refiners_autoencoder.to(test_device)


@no_grad()
def test_encode_decode_image(autoencoder: LatentDiffusionAutoencoder, sample_image: Image.Image):
    encoded = autoencoder.image_to_latents(sample_image)
    decoded = autoencoder.latents_to_image(encoded)

    assert decoded.mode == "RGB"  # type: ignore

    # Ensure no saturation. The green channel (band = 1) must not max out.
    assert max(iter(decoded.getdata(band=1))) < 255  # type: ignore

    ensure_similar_images(sample_image, decoded, min_psnr=20, min_ssim=0.9)


@no_grad()
def test_encode_decode_images(autoencoder: LatentDiffusionAutoencoder, sample_image: Image.Image):
    encoded = autoencoder.images_to_latents([sample_image, sample_image])
    images = autoencoder.latents_to_images(encoded)
    assert isinstance(images, list)
    assert len(images) == 2
    ensure_similar_images(sample_image, images[1], min_psnr=20, min_ssim=0.9)


@no_grad()
def test_tiled_autoencoder(autoencoder: LatentDiffusionAutoencoder, sample_image: Image.Image):
    sample_image = sample_image.resize((2048, 2048))  # type: ignore

    with autoencoder.tiled_inference(sample_image, tile_size=(512, 512)):
        encoded = autoencoder.tiled_image_to_latents(sample_image)
        result = autoencoder.tiled_latents_to_image(encoded)

    ensure_similar_images(sample_image, result, min_psnr=35, min_ssim=0.985)


@no_grad()
def test_tiled_autoencoder_rectangular_tiles(autoencoder: LatentDiffusionAutoencoder, sample_image: Image.Image):
    sample_image = sample_image.resize((2048, 2048))  # type: ignore

    with autoencoder.tiled_inference(sample_image, tile_size=(512, 1024)):
        encoded = autoencoder.tiled_image_to_latents(sample_image)
        result = autoencoder.tiled_latents_to_image(encoded)

    ensure_similar_images(sample_image, result, min_psnr=35, min_ssim=0.985)


@no_grad()
def test_tiled_autoencoder_large_tile(autoencoder: LatentDiffusionAutoencoder, sample_image: Image.Image):
    sample_image = sample_image.resize((1024, 1024))  # type: ignore

    with autoencoder.tiled_inference(sample_image, tile_size=(2048, 2048)):
        encoded = autoencoder.tiled_image_to_latents(sample_image)
        result = autoencoder.tiled_latents_to_image(encoded)

    ensure_similar_images(sample_image, result, min_psnr=34, min_ssim=0.975)


@no_grad()
def test_tiled_autoencoder_rectangular_image(autoencoder: LatentDiffusionAutoencoder, sample_image: Image.Image):
    sample_image = sample_image.crop((0, 0, 300, 500))
    sample_image = sample_image.resize((sample_image.width * 4, sample_image.height * 4))  # type: ignore

    with autoencoder.tiled_inference(sample_image, tile_size=(512, 512)):
        encoded = autoencoder.tiled_image_to_latents(sample_image)
        result = autoencoder.tiled_latents_to_image(encoded)

    ensure_similar_images(sample_image, result, min_psnr=37, min_ssim=0.985)


def test_value_error_tile_encode_no_context(autoencoder: LatentDiffusionAutoencoder, sample_image: Image.Image) -> None:
    with pytest.raises(ValueError):
        autoencoder.tiled_image_to_latents(sample_image)

    with pytest.raises(ValueError):
        autoencoder.tiled_latents_to_image(torch.randn(1, 8, 16, 16, device=autoencoder.device))
