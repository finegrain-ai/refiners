import gc
from pathlib import Path
from warnings import warn

import pytest
import torch
from PIL import Image

from refiners.fluxion.utils import load_from_safetensors, manual_seed, no_grad
from refiners.foundationals.latent_diffusion import SDXLIPAdapter
from refiners.foundationals.latent_diffusion.lora import SDLoraManager
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.model import StableDiffusion_XL
from tests.utils import ensure_similar_images


@pytest.fixture(autouse=True)
def ensure_gc():
    # Avoid GPU OOMs
    # See https://github.com/pytest-dev/pytest/discussions/8153#discussioncomment-214812
    gc.collect()


@pytest.fixture(scope="module")
def ref_path(test_e2e_path: Path) -> Path:
    return test_e2e_path / "test_doc_examples_ref"


@pytest.fixture(scope="module")
def sdxl_text_encoder_weights(test_weights_path: Path) -> Path:
    path = test_weights_path / "DoubleCLIPTextEncoder.safetensors"
    if not path.is_file():
        warn(message=f"could not find weights at {path}, skipping")
        pytest.skip(allow_module_level=True)
    return path


@pytest.fixture(scope="module")
def sdxl_lda_fp16_fix_weights(test_weights_path: Path) -> Path:
    path = test_weights_path / "sdxl-lda-fp16-fix.safetensors"
    if not path.is_file():
        warn(message=f"could not find weights at {path}, skipping")
        pytest.skip(allow_module_level=True)
    return path


@pytest.fixture(scope="module")
def sdxl_unet_weights(test_weights_path: Path) -> Path:
    path = test_weights_path / "sdxl-unet.safetensors"
    if not path.is_file():
        warn(message=f"could not find weights at {path}, skipping")
        pytest.skip(allow_module_level=True)
    return path


@pytest.fixture(scope="module")
def sdxl_ip_adapter_plus_weights(test_weights_path: Path) -> Path:
    path = test_weights_path / "ip-adapter-plus_sdxl_vit-h.safetensors"
    if not path.is_file():
        warn(f"could not find weights at {path}, skipping")
        pytest.skip(allow_module_level=True)
    return path


@pytest.fixture(scope="module")
def image_encoder_weights(test_weights_path: Path) -> Path:
    path = test_weights_path / "CLIPImageEncoderH.safetensors"
    if not path.is_file():
        warn(f"could not find weights at {path}, skipping")
        pytest.skip(allow_module_level=True)
    return path


@pytest.fixture
def scifi_lora_weights(test_weights_path: Path) -> Path:
    path = test_weights_path / "loras" / "Sci-fi_Environments_sdxl.safetensors"
    if not path.is_file():
        warn(message=f"could not find weights at {path}, skipping")
        pytest.skip(allow_module_level=True)
    return path


@pytest.fixture
def pixelart_lora_weights(test_weights_path: Path) -> Path:
    path = test_weights_path / "loras" / "pixel-art-xl-v1.1.safetensors"
    if not path.is_file():
        warn(message=f"could not find weights at {path}, skipping")
        pytest.skip(allow_module_level=True)
    return path


@pytest.fixture
def sdxl(
    sdxl_text_encoder_weights: Path,
    sdxl_lda_fp16_fix_weights: Path,
    sdxl_unet_weights: Path,
    test_device: torch.device,
) -> StableDiffusion_XL:
    if test_device.type == "cpu":
        warn(message="not running on CPU, skipping")
        pytest.skip()

    sdxl = StableDiffusion_XL(device=test_device, dtype=torch.float16)

    sdxl.clip_text_encoder.load_from_safetensors(tensors_path=sdxl_text_encoder_weights)
    sdxl.lda.load_from_safetensors(tensors_path=sdxl_lda_fp16_fix_weights)
    sdxl.unet.load_from_safetensors(tensors_path=sdxl_unet_weights)

    return sdxl


@pytest.fixture
def image_prompt_german_castle(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "german-castle.jpg").convert("RGB")


@pytest.fixture
def expected_image_guide_adapting_sdxl_vanilla(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_image_guide_adapting_sdxl_vanilla.png").convert("RGB")


@pytest.fixture
def expected_image_guide_adapting_sdxl_single_lora(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_image_guide_adapting_sdxl_single_lora.png").convert("RGB")


@pytest.fixture
def expected_image_guide_adapting_sdxl_multiple_loras(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_image_guide_adapting_sdxl_multiple_loras.png").convert("RGB")


@pytest.fixture
def expected_image_guide_adapting_sdxl_loras_ip_adapter(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_image_guide_adapting_sdxl_loras_ip_adapter.png").convert("RGB")


@no_grad()
def test_guide_adapting_sdxl_vanilla(
    test_device: torch.device,
    sdxl: StableDiffusion_XL,
    expected_image_guide_adapting_sdxl_vanilla: Image.Image,
) -> None:
    if test_device.type == "cpu":
        warn(message="not running on CPU, skipping")
        pytest.skip()

    expected_image = expected_image_guide_adapting_sdxl_vanilla

    prompt = "a futuristic castle surrounded by a forest, mountains in the background"
    seed = 42
    sdxl.set_inference_steps(50, first_step=0)
    sdxl.set_self_attention_guidance(enable=True, scale=0.75)

    clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
        text=prompt + ", best quality, high quality",
        negative_text="monochrome, lowres, bad anatomy, worst quality, low quality",
    )
    time_ids = sdxl.default_time_ids

    manual_seed(seed)
    # The guide uses 2048x2048 but it is too slow for tests.
    x = sdxl.init_latents((1024, 1024)).to(sdxl.device, sdxl.dtype)
    for step in sdxl.steps:
        x = sdxl(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            time_ids=time_ids,
        )

    predicted_image = sdxl.lda.decode_latents(x)
    ensure_similar_images(predicted_image, expected_image)


@no_grad()
def test_guide_adapting_sdxl_single_lora(
    test_device: torch.device,
    sdxl: StableDiffusion_XL,
    scifi_lora_weights: Path,
    expected_image_guide_adapting_sdxl_single_lora: Image.Image,
) -> None:
    if test_device.type == "cpu":
        warn(message="not running on CPU, skipping")
        pytest.skip()

    expected_image = expected_image_guide_adapting_sdxl_single_lora

    prompt = "a futuristic castle surrounded by a forest, mountains in the background"
    seed = 42
    sdxl.set_inference_steps(50, first_step=0)
    sdxl.set_self_attention_guidance(enable=True, scale=0.75)

    manager = SDLoraManager(sdxl)
    manager.add_loras("scifi-lora", load_from_safetensors(scifi_lora_weights))

    clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
        text=prompt + ", best quality, high quality",
        negative_text="monochrome, lowres, bad anatomy, worst quality, low quality",
    )
    time_ids = sdxl.default_time_ids

    manual_seed(seed)
    x = sdxl.init_latents((1024, 1024)).to(sdxl.device, sdxl.dtype)
    for step in sdxl.steps:
        x = sdxl(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            time_ids=time_ids,
        )

    predicted_image = sdxl.lda.decode_latents(x)
    ensure_similar_images(predicted_image, expected_image)


@no_grad()
def test_guide_adapting_sdxl_multiple_loras(
    test_device: torch.device,
    sdxl: StableDiffusion_XL,
    scifi_lora_weights: Path,
    pixelart_lora_weights: Path,
    expected_image_guide_adapting_sdxl_multiple_loras: Image.Image,
) -> None:
    if test_device.type == "cpu":
        warn(message="not running on CPU, skipping")
        pytest.skip()

    expected_image = expected_image_guide_adapting_sdxl_multiple_loras

    prompt = "a futuristic castle surrounded by a forest, mountains in the background"
    seed = 42
    sdxl.set_inference_steps(50, first_step=0)
    sdxl.set_self_attention_guidance(enable=True, scale=0.75)

    manager = SDLoraManager(sdxl)
    manager.add_loras("scifi-lora", load_from_safetensors(scifi_lora_weights))
    manager.add_loras("pixel-art-lora", load_from_safetensors(pixelart_lora_weights), scale=1.4)

    clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
        text=prompt + ", best quality, high quality",
        negative_text="monochrome, lowres, bad anatomy, worst quality, low quality",
    )
    time_ids = sdxl.default_time_ids

    manual_seed(seed)
    x = sdxl.init_latents((1024, 1024)).to(sdxl.device, sdxl.dtype)
    for step in sdxl.steps:
        x = sdxl(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            time_ids=time_ids,
        )

    predicted_image = sdxl.lda.decode_latents(x)
    ensure_similar_images(predicted_image, expected_image)


@no_grad()
def test_guide_adapting_sdxl_loras_ip_adapter(
    test_device: torch.device,
    sdxl: StableDiffusion_XL,
    sdxl_ip_adapter_plus_weights: Path,
    image_encoder_weights: Path,
    scifi_lora_weights: Path,
    pixelart_lora_weights: Path,
    image_prompt_german_castle: Image.Image,
    expected_image_guide_adapting_sdxl_loras_ip_adapter: Image.Image,
) -> None:
    if test_device.type == "cpu":
        warn(message="not running on CPU, skipping")
        pytest.skip()

    expected_image = expected_image_guide_adapting_sdxl_loras_ip_adapter

    prompt = "a futuristic castle surrounded by a forest, mountains in the background"
    seed = 42
    sdxl.set_inference_steps(50, first_step=0)
    sdxl.set_self_attention_guidance(enable=True, scale=0.75)

    manager = SDLoraManager(sdxl)
    manager.add_loras("scifi-lora", load_from_safetensors(scifi_lora_weights), scale=1.5)
    manager.add_loras("pixel-art-lora", load_from_safetensors(pixelart_lora_weights), scale=1.55)

    ip_adapter = SDXLIPAdapter(
        target=sdxl.unet,
        weights=load_from_safetensors(sdxl_ip_adapter_plus_weights),
        scale=1.0,
        fine_grained=True,
    )
    ip_adapter.clip_image_encoder.load_from_safetensors(image_encoder_weights)
    ip_adapter.inject()

    clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
        text=prompt + ", best quality, high quality",
        negative_text="monochrome, lowres, bad anatomy, worst quality, low quality",
    )
    time_ids = sdxl.default_time_ids

    image_prompt_preprocessed = ip_adapter.preprocess_image(image_prompt_german_castle)
    clip_image_embedding = ip_adapter.compute_clip_image_embedding(image_prompt_preprocessed)
    ip_adapter.set_clip_image_embedding(clip_image_embedding)

    manual_seed(seed)
    x = sdxl.init_latents((1024, 1024)).to(sdxl.device, sdxl.dtype)
    for step in sdxl.steps:
        x = sdxl(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            time_ids=time_ids,
        )

    predicted_image = sdxl.lda.decode_latents(x)
    ensure_similar_images(predicted_image, expected_image)


# We do not (yet) test the last example using T2i-Adapter with Zoe Depth.
