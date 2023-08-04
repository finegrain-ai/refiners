import torch
import pytest

from warnings import warn
from PIL import Image
from pathlib import Path

from refiners.fluxion.utils import load_from_safetensors
from refiners.foundationals.latent_diffusion.auto_encoder import LatentDiffusionAutoencoder

from tests.utils import ensure_similar_images


@pytest.fixture(scope="module")
def ref_path() -> Path:
    return Path(__file__).parent / "test_auto_encoder_ref"


@pytest.fixture(scope="module")
def encoder(test_weights_path: Path, test_device: torch.device) -> LatentDiffusionAutoencoder:
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
    img = Image.open(test_image)
    assert img.size == (512, 512)
    return img


@torch.no_grad()
def test_encode_decode(encoder: LatentDiffusionAutoencoder, sample_image: Image.Image):
    encoded = encoder.encode_image(sample_image)
    decoded = encoder.decode_latents(encoded)

    assert decoded.mode == "RGB"

    # Ensure no saturation. The green channel (band = 1) must not max out.
    assert max(iter(decoded.getdata(band=1))) < 255  # type: ignore

    ensure_similar_images(sample_image, decoded, min_psnr=20, min_ssim=0.9)
