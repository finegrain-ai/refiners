import torch
import pytest

from typing import Iterator

from warnings import warn
from PIL import Image
from pathlib import Path

from refiners.fluxion.utils import load_from_safetensors, image_to_tensor, manual_seed
from refiners.foundationals.latent_diffusion import (
    StableDiffusion_1,
    StableDiffusion_1_Inpainting,
    SD1UNet,
    SD1ControlnetAdapter,
    SD1IPAdapter,
    SD1T2IAdapter,
    SDXLIPAdapter,
    SDXLT2IAdapter,
)
from refiners.foundationals.latent_diffusion.lora import SD1LoraAdapter
from refiners.foundationals.latent_diffusion.multi_diffusion import DiffusionTarget
from refiners.foundationals.latent_diffusion.schedulers import DDIM
from refiners.foundationals.latent_diffusion.reference_only_control import ReferenceOnlyControlAdapter
from refiners.foundationals.clip.concepts import ConceptExtender
from refiners.foundationals.latent_diffusion.stable_diffusion_1.multi_diffusion import SD1MultiDiffusion
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.model import StableDiffusion_XL

from tests.utils import ensure_similar_images


@pytest.fixture(scope="module")
def ref_path(test_e2e_path: Path) -> Path:
    return test_e2e_path / "test_diffusion_ref"


@pytest.fixture(scope="module")
def cutecat_init(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "cutecat_init.png").convert("RGB")


@pytest.fixture(scope="module")
def kitchen_dog(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "kitchen_dog.png").convert("RGB")


@pytest.fixture(scope="module")
def kitchen_dog_mask(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "kitchen_dog_mask.png").convert("RGB")


@pytest.fixture(scope="module")
def woman_image(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "woman.png").convert("RGB")


@pytest.fixture(scope="module")
def statue_image(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "statue.png").convert("RGB")


@pytest.fixture
def expected_image_std_random_init(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_std_random_init.png").convert("RGB")


@pytest.fixture
def expected_image_std_random_init_sag(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_std_random_init_sag.png").convert("RGB")


@pytest.fixture
def expected_image_std_init_image(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_std_init_image.png").convert("RGB")


@pytest.fixture
def expected_image_std_inpainting(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_std_inpainting.png").convert("RGB")


@pytest.fixture
def expected_image_controlnet_stack(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_controlnet_stack.png").convert("RGB")


@pytest.fixture
def expected_image_ip_adapter_woman(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_image_ip_adapter_woman.png").convert("RGB")


@pytest.fixture
def expected_image_ip_adapter_plus_statue(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_image_ip_adapter_plus_statue.png").convert("RGB")


@pytest.fixture
def expected_image_sdxl_ip_adapter_woman(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_image_sdxl_ip_adapter_woman.png").convert("RGB")


@pytest.fixture
def expected_image_sdxl_ip_adapter_plus_woman(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_image_sdxl_ip_adapter_plus_woman.png").convert("RGB")


@pytest.fixture
def expected_image_ip_adapter_controlnet(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_ip_adapter_controlnet.png").convert("RGB")


@pytest.fixture
def expected_sdxl_ddim_random_init(ref_path: Path) -> Image.Image:
    return Image.open(fp=ref_path / "expected_cutecat_sdxl_ddim_random_init.png").convert(mode="RGB")


@pytest.fixture
def expected_sdxl_ddim_random_init_sag(ref_path: Path) -> Image.Image:
    return Image.open(fp=ref_path / "expected_cutecat_sdxl_ddim_random_init_sag.png").convert(mode="RGB")


@pytest.fixture(scope="module", params=["canny", "depth", "lineart", "normals", "sam"])
def controlnet_data(
    ref_path: Path, test_weights_path: Path, request: pytest.FixtureRequest
) -> Iterator[tuple[str, Image.Image, Image.Image, Path]]:
    cn_name: str = request.param
    condition_image = Image.open(ref_path / f"cutecat_guide_{cn_name}.png").convert("RGB")
    expected_image = Image.open(ref_path / f"expected_controlnet_{cn_name}.png").convert("RGB")
    weights_fn = {
        "depth": "lllyasviel_control_v11f1p_sd15_depth",
        "canny": "lllyasviel_control_v11p_sd15_canny",
        "lineart": "lllyasviel_control_v11p_sd15_lineart",
        "normals": "lllyasviel_control_v11p_sd15_normalbae",
        "sam": "mfidabel_controlnet-segment-anything",
    }

    weights_path = test_weights_path / "controlnet" / f"{weights_fn[cn_name]}.safetensors"
    yield (cn_name, condition_image, expected_image, weights_path)


@pytest.fixture(scope="module")
def controlnet_data_canny(ref_path: Path, test_weights_path: Path) -> tuple[str, Image.Image, Image.Image, Path]:
    cn_name = "canny"
    condition_image = Image.open(ref_path / f"cutecat_guide_{cn_name}.png").convert("RGB")
    expected_image = Image.open(ref_path / f"expected_controlnet_{cn_name}.png").convert("RGB")
    weights_path = test_weights_path / "controlnet" / "lllyasviel_control_v11p_sd15_canny.safetensors"
    return cn_name, condition_image, expected_image, weights_path


@pytest.fixture(scope="module")
def controlnet_data_depth(ref_path: Path, test_weights_path: Path) -> tuple[str, Image.Image, Image.Image, Path]:
    cn_name = "depth"
    condition_image = Image.open(ref_path / f"cutecat_guide_{cn_name}.png").convert("RGB")
    expected_image = Image.open(ref_path / f"expected_controlnet_{cn_name}.png").convert("RGB")
    weights_path = test_weights_path / "controlnet" / "lllyasviel_control_v11f1p_sd15_depth.safetensors"
    return cn_name, condition_image, expected_image, weights_path


@pytest.fixture(scope="module")
def t2i_adapter_data_depth(ref_path: Path, test_weights_path: Path) -> tuple[str, Image.Image, Image.Image, Path]:
    name = "depth"
    condition_image = Image.open(ref_path / f"cutecat_guide_{name}.png").convert("RGB")
    expected_image = Image.open(ref_path / f"expected_t2i_adapter_{name}.png").convert("RGB")
    weights_path = test_weights_path / "T2I-Adapter" / "t2iadapter_depth_sd15v2.safetensors"
    return name, condition_image, expected_image, weights_path


@pytest.fixture(scope="module")
def t2i_adapter_xl_data_canny(ref_path: Path, test_weights_path: Path) -> tuple[str, Image.Image, Image.Image, Path]:
    name = "canny"
    condition_image = Image.open(ref_path / f"fairy_guide_{name}.png").convert("RGB")
    expected_image = Image.open(ref_path / f"expected_t2i_adapter_xl_{name}.png").convert("RGB")
    weights_path = test_weights_path / "T2I-Adapter" / "t2i-adapter-canny-sdxl-1.0.safetensors"
    return name, condition_image, expected_image, weights_path


@pytest.fixture(scope="module")
def lora_data_pokemon(ref_path: Path, test_weights_path: Path) -> tuple[Image.Image, Path]:
    expected_image = Image.open(ref_path / "expected_lora_pokemon.png").convert("RGB")
    weights_path = test_weights_path / "loras" / "pcuenq_pokemon_lora.safetensors"
    return expected_image, weights_path


@pytest.fixture
def scene_image_inpainting_refonly(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "inpainting-scene.png").convert("RGB")


@pytest.fixture
def mask_image_inpainting_refonly(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "inpainting-mask.png").convert("RGB")


@pytest.fixture
def target_image_inpainting_refonly(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "inpainting-target.png").convert("RGB")


@pytest.fixture
def expected_image_inpainting_refonly(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_inpainting_refonly.png").convert("RGB")


@pytest.fixture
def expected_image_refonly(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_refonly.png").convert("RGB")


@pytest.fixture
def condition_image_refonly(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "cyberpunk_guide.png").convert("RGB")


@pytest.fixture
def expected_image_textual_inversion_random_init(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_textual_inversion_random_init.png").convert("RGB")


@pytest.fixture
def expected_multi_diffusion(ref_path: Path) -> Image.Image:
    return Image.open(fp=ref_path / "expected_multi_diffusion.png").convert(mode="RGB")


@pytest.fixture
def text_embedding_textual_inversion(test_textual_inversion_path: Path) -> torch.Tensor:
    return torch.load(test_textual_inversion_path / "gta5-artwork" / "learned_embeds.bin")["<gta5-artwork>"]  # type: ignore


@pytest.fixture(scope="module")
def text_encoder_weights(test_weights_path: Path) -> Path:
    text_encoder_weights = test_weights_path / "CLIPTextEncoderL.safetensors"
    if not text_encoder_weights.is_file():
        warn(f"could not find weights at {text_encoder_weights}, skipping")
        pytest.skip(allow_module_level=True)
    return text_encoder_weights


@pytest.fixture(scope="module")
def lda_weights(test_weights_path: Path) -> Path:
    lda_weights = test_weights_path / "lda.safetensors"
    if not lda_weights.is_file():
        warn(f"could not find weights at {lda_weights}, skipping")
        pytest.skip(allow_module_level=True)
    return lda_weights


@pytest.fixture(scope="module")
def unet_weights_std(test_weights_path: Path) -> Path:
    unet_weights_std = test_weights_path / "unet.safetensors"
    if not unet_weights_std.is_file():
        warn(f"could not find weights at {unet_weights_std}, skipping")
        pytest.skip(allow_module_level=True)
    return unet_weights_std


@pytest.fixture(scope="module")
def unet_weights_inpainting(test_weights_path: Path) -> Path:
    unet_weights_inpainting = test_weights_path / "inpainting" / "unet.safetensors"
    if not unet_weights_inpainting.is_file():
        warn(f"could not find weights at {unet_weights_inpainting}, skipping")
        pytest.skip(allow_module_level=True)
    return unet_weights_inpainting


@pytest.fixture(scope="module")
def lda_ft_mse_weights(test_weights_path: Path) -> Path:
    lda_weights = test_weights_path / "lda_ft_mse.safetensors"
    if not lda_weights.is_file():
        warn(f"could not find weights at {lda_weights}, skipping")
        pytest.skip(allow_module_level=True)
    return lda_weights


@pytest.fixture(scope="module")
def ip_adapter_weights(test_weights_path: Path) -> Path:
    ip_adapter_weights = test_weights_path / "ip-adapter_sd15.safetensors"
    if not ip_adapter_weights.is_file():
        warn(f"could not find weights at {ip_adapter_weights}, skipping")
        pytest.skip(allow_module_level=True)
    return ip_adapter_weights


@pytest.fixture(scope="module")
def ip_adapter_plus_weights(test_weights_path: Path) -> Path:
    ip_adapter_weights = test_weights_path / "ip-adapter-plus_sd15.safetensors"
    if not ip_adapter_weights.is_file():
        warn(f"could not find weights at {ip_adapter_weights}, skipping")
        pytest.skip(allow_module_level=True)
    return ip_adapter_weights


@pytest.fixture(scope="module")
def sdxl_ip_adapter_weights(test_weights_path: Path) -> Path:
    ip_adapter_weights = test_weights_path / "ip-adapter_sdxl_vit-h.safetensors"
    if not ip_adapter_weights.is_file():
        warn(f"could not find weights at {ip_adapter_weights}, skipping")
        pytest.skip(allow_module_level=True)
    return ip_adapter_weights


@pytest.fixture(scope="module")
def sdxl_ip_adapter_plus_weights(test_weights_path: Path) -> Path:
    ip_adapter_weights = test_weights_path / "ip-adapter-plus_sdxl_vit-h.safetensors"
    if not ip_adapter_weights.is_file():
        warn(f"could not find weights at {ip_adapter_weights}, skipping")
        pytest.skip(allow_module_level=True)
    return ip_adapter_weights


@pytest.fixture(scope="module")
def image_encoder_weights(test_weights_path: Path) -> Path:
    image_encoder_weights = test_weights_path / "CLIPImageEncoderH.safetensors"
    if not image_encoder_weights.is_file():
        warn(f"could not find weights at {image_encoder_weights}, skipping")
        pytest.skip(allow_module_level=True)
    return image_encoder_weights


@pytest.fixture
def sd15_std(
    text_encoder_weights: Path, lda_weights: Path, unet_weights_std: Path, test_device: torch.device
) -> StableDiffusion_1:
    if test_device.type == "cpu":
        warn("not running on CPU, skipping")
        pytest.skip()

    sd15 = StableDiffusion_1(device=test_device)

    sd15.clip_text_encoder.load_from_safetensors(text_encoder_weights)
    sd15.lda.load_from_safetensors(lda_weights)
    sd15.unet.load_from_safetensors(unet_weights_std)

    return sd15


@pytest.fixture
def sd15_std_float16(
    text_encoder_weights: Path, lda_weights: Path, unet_weights_std: Path, test_device: torch.device
) -> StableDiffusion_1:
    if test_device.type == "cpu":
        warn("not running on CPU, skipping")
        pytest.skip()

    sd15 = StableDiffusion_1(device=test_device, dtype=torch.float16)

    sd15.clip_text_encoder.load_from_safetensors(text_encoder_weights)
    sd15.lda.load_from_safetensors(lda_weights)
    sd15.unet.load_from_safetensors(unet_weights_std)

    return sd15


@pytest.fixture
def sd15_inpainting(
    text_encoder_weights: Path, lda_weights: Path, unet_weights_inpainting: Path, test_device: torch.device
) -> StableDiffusion_1_Inpainting:
    if test_device.type == "cpu":
        warn("not running on CPU, skipping")
        pytest.skip()

    unet = SD1UNet(in_channels=9)
    sd15 = StableDiffusion_1_Inpainting(unet=unet, device=test_device)

    sd15.clip_text_encoder.load_from_safetensors(text_encoder_weights)
    sd15.lda.load_from_safetensors(lda_weights)
    sd15.unet.load_from_safetensors(unet_weights_inpainting)

    return sd15


@pytest.fixture
def sd15_inpainting_float16(
    text_encoder_weights: Path, lda_weights: Path, unet_weights_inpainting: Path, test_device: torch.device
) -> StableDiffusion_1_Inpainting:
    if test_device.type == "cpu":
        warn("not running on CPU, skipping")
        pytest.skip()

    unet = SD1UNet(in_channels=9)
    sd15 = StableDiffusion_1_Inpainting(unet=unet, device=test_device, dtype=torch.float16)

    sd15.clip_text_encoder.load_from_safetensors(text_encoder_weights)
    sd15.lda.load_from_safetensors(lda_weights)
    sd15.unet.load_from_safetensors(unet_weights_inpainting)

    return sd15


@pytest.fixture
def sd15_ddim(
    text_encoder_weights: Path, lda_weights: Path, unet_weights_std: Path, test_device: torch.device
) -> StableDiffusion_1:
    if test_device.type == "cpu":
        warn("not running on CPU, skipping")
        pytest.skip()

    ddim_scheduler = DDIM(num_inference_steps=20)
    sd15 = StableDiffusion_1(scheduler=ddim_scheduler, device=test_device)

    sd15.clip_text_encoder.load_from_safetensors(text_encoder_weights)
    sd15.lda.load_from_safetensors(lda_weights)
    sd15.unet.load_from_safetensors(unet_weights_std)

    return sd15


@pytest.fixture
def sd15_ddim_lda_ft_mse(
    text_encoder_weights: Path, lda_ft_mse_weights: Path, unet_weights_std: Path, test_device: torch.device
) -> StableDiffusion_1:
    if test_device.type == "cpu":
        warn("not running on CPU, skipping")
        pytest.skip()

    ddim_scheduler = DDIM(num_inference_steps=20)
    sd15 = StableDiffusion_1(scheduler=ddim_scheduler, device=test_device)

    sd15.clip_text_encoder.load_state_dict(load_from_safetensors(text_encoder_weights))
    sd15.lda.load_state_dict(load_from_safetensors(lda_ft_mse_weights))
    sd15.unet.load_state_dict(load_from_safetensors(unet_weights_std))

    return sd15


@pytest.fixture
def sdxl_lda_weights(test_weights_path: Path) -> Path:
    sdxl_lda_weights = test_weights_path / "sdxl-lda.safetensors"
    if not sdxl_lda_weights.is_file():
        warn(message=f"could not find weights at {sdxl_lda_weights}, skipping")
        pytest.skip(allow_module_level=True)
    return sdxl_lda_weights


@pytest.fixture
def sdxl_unet_weights(test_weights_path: Path) -> Path:
    sdxl_unet_weights = test_weights_path / "sdxl-unet.safetensors"
    if not sdxl_unet_weights.is_file():
        warn(message=f"could not find weights at {sdxl_unet_weights}, skipping")
        pytest.skip(allow_module_level=True)
    return sdxl_unet_weights


@pytest.fixture
def sdxl_text_encoder_weights(test_weights_path: Path) -> Path:
    sdxl_double_text_encoder_weights = test_weights_path / "DoubleCLIPTextEncoder.safetensors"
    if not sdxl_double_text_encoder_weights.is_file():
        warn(message=f"could not find weights at {sdxl_double_text_encoder_weights}, skipping")
        pytest.skip(allow_module_level=True)
    return sdxl_double_text_encoder_weights


@pytest.fixture
def sdxl_ddim(
    sdxl_text_encoder_weights: Path, sdxl_lda_weights: Path, sdxl_unet_weights: Path, test_device: torch.device
) -> StableDiffusion_XL:
    if test_device.type == "cpu":
        warn(message="not running on CPU, skipping")
        pytest.skip()

    scheduler = DDIM(num_inference_steps=30)
    sdxl = StableDiffusion_XL(scheduler=scheduler, device=test_device)

    sdxl.clip_text_encoder.load_from_safetensors(tensors_path=sdxl_text_encoder_weights)
    sdxl.lda.load_from_safetensors(tensors_path=sdxl_lda_weights)
    sdxl.unet.load_from_safetensors(tensors_path=sdxl_unet_weights)

    return sdxl


@torch.no_grad()
def test_diffusion_std_random_init(
    sd15_std: StableDiffusion_1, expected_image_std_random_init: Image.Image, test_device: torch.device
):
    sd15 = sd15_std
    n_steps = 30

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

    sd15.set_num_inference_steps(n_steps)

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.decode_latents(x)

    ensure_similar_images(predicted_image, expected_image_std_random_init)


@torch.no_grad()
def test_diffusion_std_random_init_float16(
    sd15_std_float16: StableDiffusion_1, expected_image_std_random_init: Image.Image, test_device: torch.device
):
    sd15 = sd15_std_float16
    n_steps = 30

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)
    assert clip_text_embedding.dtype == torch.float16

    sd15.set_num_inference_steps(n_steps)

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device, dtype=torch.float16)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.decode_latents(x)

    ensure_similar_images(predicted_image, expected_image_std_random_init, min_psnr=35, min_ssim=0.98)


@torch.no_grad()
def test_diffusion_std_random_init_sag(
    sd15_std: StableDiffusion_1, expected_image_std_random_init_sag: Image.Image, test_device: torch.device
):
    sd15 = sd15_std
    n_steps = 30

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

    sd15.set_num_inference_steps(n_steps)
    sd15.set_self_attention_guidance(enable=True, scale=0.75)

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.decode_latents(x)

    ensure_similar_images(predicted_image, expected_image_std_random_init_sag)


@torch.no_grad()
def test_diffusion_std_init_image(
    sd15_std: StableDiffusion_1,
    cutecat_init: Image.Image,
    expected_image_std_init_image: Image.Image,
):
    sd15 = sd15_std
    n_steps = 35
    first_step = 5

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

    sd15.set_num_inference_steps(n_steps)

    manual_seed(2)
    x = sd15.init_latents((512, 512), cutecat_init, first_step=first_step)

    for step in sd15.steps[first_step:]:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.decode_latents(x)

    ensure_similar_images(predicted_image, expected_image_std_init_image)


@torch.no_grad()
def test_rectangular_init_latents(
    sd15_std: StableDiffusion_1,
    cutecat_init: Image.Image,
):
    sd15 = sd15_std

    # Just check latents initialization with a non-square image (and not the entire diffusion)
    width, height = 512, 504
    rect_init_image = cutecat_init.crop((0, 0, width, height))
    x = sd15.init_latents((height, width), rect_init_image)

    assert sd15.lda.decode_latents(x).size == (width, height)


@torch.no_grad()
def test_diffusion_inpainting(
    sd15_inpainting: StableDiffusion_1_Inpainting,
    kitchen_dog: Image.Image,
    kitchen_dog_mask: Image.Image,
    expected_image_std_inpainting: Image.Image,
    test_device: torch.device,
):
    sd15 = sd15_inpainting
    n_steps = 30

    prompt = "a large white cat, detailed high-quality professional image, sitting on a chair, in a kitchen"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

    sd15.set_num_inference_steps(n_steps)
    sd15.set_inpainting_conditions(kitchen_dog, kitchen_dog_mask)

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.decode_latents(x)

    # PSNR and SSIM values are large because with float32 we get large differences even v.s. ourselves.
    ensure_similar_images(predicted_image, expected_image_std_inpainting, min_psnr=25, min_ssim=0.95)


@torch.no_grad()
def test_diffusion_inpainting_float16(
    sd15_inpainting_float16: StableDiffusion_1_Inpainting,
    kitchen_dog: Image.Image,
    kitchen_dog_mask: Image.Image,
    expected_image_std_inpainting: Image.Image,
    test_device: torch.device,
):
    sd15 = sd15_inpainting_float16
    n_steps = 30

    prompt = "a large white cat, detailed high-quality professional image, sitting on a chair, in a kitchen"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)
    assert clip_text_embedding.dtype == torch.float16

    sd15.set_num_inference_steps(n_steps)
    sd15.set_inpainting_conditions(kitchen_dog, kitchen_dog_mask)

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device, dtype=torch.float16)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.decode_latents(x)

    # PSNR and SSIM values are large because float16 is even worse than float32.
    ensure_similar_images(predicted_image, expected_image_std_inpainting, min_psnr=20, min_ssim=0.92)


@torch.no_grad()
def test_diffusion_controlnet(
    sd15_std: StableDiffusion_1,
    controlnet_data: tuple[str, Image.Image, Image.Image, Path],
    test_device: torch.device,
):
    sd15 = sd15_std
    n_steps = 30

    cn_name, condition_image, expected_image, cn_weights_path = controlnet_data

    if not cn_weights_path.is_file():
        warn(f"could not find weights at {cn_weights_path}, skipping")
        pytest.skip(allow_module_level=True)

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

    sd15.set_num_inference_steps(n_steps)

    controlnet = SD1ControlnetAdapter(
        sd15.unet, name=cn_name, scale=0.5, weights=load_from_safetensors(cn_weights_path)
    ).inject()

    cn_condition = image_to_tensor(condition_image.convert("RGB"), device=test_device)

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device)

    for step in sd15.steps:
        controlnet.set_controlnet_condition(cn_condition)
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.decode_latents(x)

    ensure_similar_images(predicted_image, expected_image, min_psnr=35, min_ssim=0.98)


@torch.no_grad()
def test_diffusion_controlnet_structural_copy(
    sd15_std: StableDiffusion_1,
    controlnet_data_canny: tuple[str, Image.Image, Image.Image, Path],
    test_device: torch.device,
):
    sd15_base = sd15_std
    sd15 = sd15_base.structural_copy()
    n_steps = 30

    cn_name, condition_image, expected_image, cn_weights_path = controlnet_data_canny

    if not cn_weights_path.is_file():
        warn(f"could not find weights at {cn_weights_path}, skipping")
        pytest.skip(allow_module_level=True)

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

    sd15.set_num_inference_steps(n_steps)

    controlnet = SD1ControlnetAdapter(
        sd15.unet, name=cn_name, scale=0.5, weights=load_from_safetensors(cn_weights_path)
    ).inject()

    cn_condition = image_to_tensor(condition_image.convert("RGB"), device=test_device)

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device)

    for step in sd15.steps:
        controlnet.set_controlnet_condition(cn_condition)
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.decode_latents(x)

    ensure_similar_images(predicted_image, expected_image, min_psnr=35, min_ssim=0.98)


@torch.no_grad()
def test_diffusion_controlnet_float16(
    sd15_std_float16: StableDiffusion_1,
    controlnet_data_canny: tuple[str, Image.Image, Image.Image, Path],
    test_device: torch.device,
):
    sd15 = sd15_std_float16
    n_steps = 30

    cn_name, condition_image, expected_image, cn_weights_path = controlnet_data_canny

    if not cn_weights_path.is_file():
        warn(f"could not find weights at {cn_weights_path}, skipping")
        pytest.skip(allow_module_level=True)

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

    sd15.set_num_inference_steps(n_steps)

    controlnet = SD1ControlnetAdapter(
        sd15.unet, name=cn_name, scale=0.5, weights=load_from_safetensors(cn_weights_path)
    ).inject()

    cn_condition = image_to_tensor(condition_image.convert("RGB"), device=test_device, dtype=torch.float16)

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device, dtype=torch.float16)

    for step in sd15.steps:
        controlnet.set_controlnet_condition(cn_condition)
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.decode_latents(x)

    ensure_similar_images(predicted_image, expected_image, min_psnr=35, min_ssim=0.98)


@torch.no_grad()
def test_diffusion_controlnet_stack(
    sd15_std: StableDiffusion_1,
    controlnet_data_depth: tuple[str, Image.Image, Image.Image, Path],
    controlnet_data_canny: tuple[str, Image.Image, Image.Image, Path],
    expected_image_controlnet_stack: Image.Image,
    test_device: torch.device,
):
    sd15 = sd15_std
    n_steps = 30

    _, depth_condition_image, _, depth_cn_weights_path = controlnet_data_depth
    _, canny_condition_image, _, canny_cn_weights_path = controlnet_data_canny

    if not canny_cn_weights_path.is_file():
        warn(f"could not find weights at {canny_cn_weights_path}, skipping")
        pytest.skip(allow_module_level=True)

    if not depth_cn_weights_path.is_file():
        warn(f"could not find weights at {depth_cn_weights_path}, skipping")
        pytest.skip(allow_module_level=True)

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

    sd15.set_num_inference_steps(n_steps)

    depth_controlnet = SD1ControlnetAdapter(
        sd15.unet, name="depth", scale=0.3, weights=load_from_safetensors(depth_cn_weights_path)
    ).inject()
    canny_controlnet = SD1ControlnetAdapter(
        sd15.unet, name="canny", scale=0.7, weights=load_from_safetensors(canny_cn_weights_path)
    ).inject()

    depth_cn_condition = image_to_tensor(depth_condition_image.convert("RGB"), device=test_device)
    canny_cn_condition = image_to_tensor(canny_condition_image.convert("RGB"), device=test_device)

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device)

    for step in sd15.steps:
        depth_controlnet.set_controlnet_condition(depth_cn_condition)
        canny_controlnet.set_controlnet_condition(canny_cn_condition)
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.decode_latents(x)

    ensure_similar_images(predicted_image, expected_image_controlnet_stack, min_psnr=35, min_ssim=0.98)


@torch.no_grad()
def test_diffusion_lora(
    sd15_std: StableDiffusion_1,
    lora_data_pokemon: tuple[Image.Image, Path],
    test_device: torch.device,
):
    sd15 = sd15_std
    n_steps = 30

    expected_image, lora_weights_path = lora_data_pokemon

    if not lora_weights_path.is_file():
        warn(f"could not find weights at {lora_weights_path}, skipping")
        pytest.skip(allow_module_level=True)

    prompt = "a cute cat"
    clip_text_embedding = sd15.compute_clip_text_embedding(prompt)

    sd15.set_num_inference_steps(n_steps)

    SD1LoraAdapter.from_safetensors(target=sd15, checkpoint_path=lora_weights_path, scale=1.0).inject()

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.decode_latents(x)

    ensure_similar_images(predicted_image, expected_image, min_psnr=35, min_ssim=0.98)


@torch.no_grad()
def test_diffusion_lora_float16(
    sd15_std_float16: StableDiffusion_1,
    lora_data_pokemon: tuple[Image.Image, Path],
    test_device: torch.device,
):
    sd15 = sd15_std_float16
    n_steps = 30

    expected_image, lora_weights_path = lora_data_pokemon

    if not lora_weights_path.is_file():
        warn(f"could not find weights at {lora_weights_path}, skipping")
        pytest.skip(allow_module_level=True)

    prompt = "a cute cat"
    clip_text_embedding = sd15.compute_clip_text_embedding(prompt)

    sd15.set_num_inference_steps(n_steps)

    SD1LoraAdapter.from_safetensors(target=sd15, checkpoint_path=lora_weights_path, scale=1.0).inject()

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device, dtype=torch.float16)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.decode_latents(x)

    ensure_similar_images(predicted_image, expected_image, min_psnr=33, min_ssim=0.98)


@torch.no_grad()
def test_diffusion_lora_twice(
    sd15_std: StableDiffusion_1,
    lora_data_pokemon: tuple[Image.Image, Path],
    test_device: torch.device,
):
    sd15 = sd15_std
    n_steps = 30

    expected_image, lora_weights_path = lora_data_pokemon

    if not lora_weights_path.is_file():
        warn(f"could not find weights at {lora_weights_path}, skipping")
        pytest.skip(allow_module_level=True)

    prompt = "a cute cat"
    clip_text_embedding = sd15.compute_clip_text_embedding(prompt)

    sd15.set_num_inference_steps(n_steps)

    # The same LoRA is used twice which is not a common use case: this is purely for testing purpose
    SD1LoraAdapter.from_safetensors(target=sd15, checkpoint_path=lora_weights_path, scale=0.4).inject()
    SD1LoraAdapter.from_safetensors(target=sd15, checkpoint_path=lora_weights_path, scale=0.6).inject()

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.decode_latents(x)

    ensure_similar_images(predicted_image, expected_image, min_psnr=35, min_ssim=0.98)


@torch.no_grad()
def test_diffusion_refonly(
    sd15_ddim: StableDiffusion_1,
    condition_image_refonly: Image.Image,
    expected_image_refonly: Image.Image,
    test_device: torch.device,
):
    sd15 = sd15_ddim

    prompt = "Chicken"
    clip_text_embedding = sd15.compute_clip_text_embedding(prompt)

    refonly_adapter = ReferenceOnlyControlAdapter(sd15.unet).inject()

    guide = sd15.lda.encode_image(condition_image_refonly)
    guide = torch.cat((guide, guide))

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device)

    for step in sd15.steps:
        noise = torch.randn(2, 4, 64, 64, device=test_device)
        noised_guide = sd15.scheduler.add_noise(guide, noise, step)
        refonly_adapter.set_controlnet_condition(noised_guide)
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
        torch.randn(2, 4, 64, 64, device=test_device)  # for SD Web UI reproductibility only
    predicted_image = sd15.lda.decode_latents(x)

    ensure_similar_images(predicted_image, expected_image_refonly, min_psnr=35, min_ssim=0.99)


@torch.no_grad()
def test_diffusion_inpainting_refonly(
    sd15_inpainting: StableDiffusion_1_Inpainting,
    scene_image_inpainting_refonly: Image.Image,
    target_image_inpainting_refonly: Image.Image,
    mask_image_inpainting_refonly: Image.Image,
    expected_image_inpainting_refonly: Image.Image,
    test_device: torch.device,
):
    sd15 = sd15_inpainting
    n_steps = 30

    prompt = ""  # unconditional
    clip_text_embedding = sd15.compute_clip_text_embedding(prompt)

    refonly_adapter = ReferenceOnlyControlAdapter(sd15.unet).inject()

    sd15.set_num_inference_steps(n_steps)
    sd15.set_inpainting_conditions(target_image_inpainting_refonly, mask_image_inpainting_refonly)

    guide = sd15.lda.encode_image(scene_image_inpainting_refonly)
    guide = torch.cat((guide, guide))

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device)

    for step in sd15.steps:
        noise = torch.randn_like(guide)
        noised_guide = sd15.scheduler.add_noise(guide, noise, step)
        # See https://github.com/Mikubill/sd-webui-controlnet/pull/1275 ("1.1.170 reference-only begin to support
        # inpaint variation models")
        noised_guide = torch.cat([noised_guide, torch.zeros_like(noised_guide)[:, 0:1, :, :], guide], dim=1)

        refonly_adapter.set_controlnet_condition(noised_guide)
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.decode_latents(x)

    ensure_similar_images(predicted_image, expected_image_inpainting_refonly, min_psnr=35, min_ssim=0.99)


@torch.no_grad()
def test_diffusion_textual_inversion_random_init(
    sd15_std: StableDiffusion_1,
    expected_image_textual_inversion_random_init: Image.Image,
    text_embedding_textual_inversion: torch.Tensor,
    test_device: torch.device,
):
    sd15 = sd15_std

    conceptExtender = ConceptExtender(sd15.clip_text_encoder)
    conceptExtender.add_concept("<gta5-artwork>", text_embedding_textual_inversion)
    conceptExtender.inject()

    n_steps = 30

    prompt = "a cute cat on a <gta5-artwork>"
    clip_text_embedding = sd15.compute_clip_text_embedding(prompt)

    sd15.set_num_inference_steps(n_steps)

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.decode_latents(x)

    ensure_similar_images(predicted_image, expected_image_textual_inversion_random_init, min_psnr=35, min_ssim=0.98)


@torch.no_grad()
def test_diffusion_ip_adapter(
    sd15_ddim_lda_ft_mse: StableDiffusion_1,
    ip_adapter_weights: Path,
    image_encoder_weights: Path,
    woman_image: Image.Image,
    expected_image_ip_adapter_woman: Image.Image,
    test_device: torch.device,
):
    sd15 = sd15_ddim_lda_ft_mse.to(dtype=torch.float16)
    n_steps = 50

    # See tencent-ailab/IP-Adapter best practices section:
    #
    #     If you only use the image prompt, you can set the scale=1.0 and text_prompt="" (or some generic text
    #     prompts, e.g. "best quality", you can also use any negative text prompt).
    #
    # The prompts below are the ones used by default by IPAdapter's generate method if none are specified
    prompt = "best quality, high quality"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    ip_adapter = SD1IPAdapter(target=sd15.unet, weights=load_from_safetensors(ip_adapter_weights))
    ip_adapter.clip_image_encoder.load_from_safetensors(image_encoder_weights)
    ip_adapter.inject()

    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)
    clip_image_embedding = ip_adapter.compute_clip_image_embedding(ip_adapter.preprocess_image(woman_image))

    negative_text_embedding, conditional_text_embedding = clip_text_embedding.chunk(2)
    negative_image_embedding, conditional_image_embedding = clip_image_embedding.chunk(2)

    clip_text_embedding = torch.cat(
        (
            torch.cat([negative_text_embedding, negative_image_embedding], dim=1),
            torch.cat([conditional_text_embedding, conditional_image_embedding], dim=1),
        )
    )

    sd15.set_num_inference_steps(n_steps)

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device, dtype=torch.float16)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.decode_latents(x)

    ensure_similar_images(predicted_image, expected_image_ip_adapter_woman)


@torch.no_grad()
def test_diffusion_sdxl_ip_adapter(
    sdxl_ddim: StableDiffusion_XL,
    sdxl_ip_adapter_weights: Path,
    image_encoder_weights: Path,
    woman_image: Image.Image,
    expected_image_sdxl_ip_adapter_woman: Image.Image,
    test_device: torch.device,
):
    sdxl = sdxl_ddim.to(dtype=torch.float16)
    n_steps = 30

    prompt = "best quality, high quality"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    ip_adapter = SDXLIPAdapter(target=sdxl.unet, weights=load_from_safetensors(sdxl_ip_adapter_weights))
    ip_adapter.clip_image_encoder.load_from_safetensors(image_encoder_weights)
    ip_adapter.inject()

    with torch.no_grad():
        clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
            text=prompt, negative_text=negative_prompt
        )
        clip_image_embedding = ip_adapter.compute_clip_image_embedding(ip_adapter.preprocess_image(woman_image))

        negative_text_embedding, conditional_text_embedding = clip_text_embedding.chunk(2)
        negative_image_embedding, conditional_image_embedding = clip_image_embedding.chunk(2)

        clip_text_embedding = torch.cat(
            (
                torch.cat([negative_text_embedding, negative_image_embedding], dim=1),
                torch.cat([conditional_text_embedding, conditional_image_embedding], dim=1),
            )
        )
    time_ids = sdxl.default_time_ids
    sdxl.set_num_inference_steps(n_steps)

    manual_seed(2)
    x = torch.randn(1, 4, 128, 128, device=test_device, dtype=torch.float16)

    with torch.no_grad():
        for step in sdxl.steps:
            x = sdxl(
                x,
                step=step,
                clip_text_embedding=clip_text_embedding,
                pooled_text_embedding=pooled_text_embedding,
                time_ids=time_ids,
                condition_scale=5,
            )
        # See https://huggingface.co/madebyollin/sdxl-vae-fp16-fix: "SDXL-VAE generates NaNs in fp16 because the
        # internal activation values are too big"
        sdxl.lda.to(dtype=torch.float32)
        predicted_image = sdxl.lda.decode_latents(x.to(dtype=torch.float32))

    ensure_similar_images(predicted_image, expected_image_sdxl_ip_adapter_woman)


@torch.no_grad()
def test_diffusion_ip_adapter_controlnet(
    sd15_ddim: StableDiffusion_1,
    ip_adapter_weights: Path,
    image_encoder_weights: Path,
    lora_data_pokemon: tuple[Image.Image, Path],
    controlnet_data_depth: tuple[str, Image.Image, Image.Image, Path],
    expected_image_ip_adapter_controlnet: Image.Image,
    test_device: torch.device,
):
    sd15 = sd15_ddim.to(dtype=torch.float16)
    n_steps = 50
    input_image, _ = lora_data_pokemon  # use the Pokemon LoRA output as input
    _, depth_condition_image, _, depth_cn_weights_path = controlnet_data_depth

    prompt = "best quality, high quality"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    ip_adapter = SD1IPAdapter(target=sd15.unet, weights=load_from_safetensors(ip_adapter_weights))
    ip_adapter.clip_image_encoder.load_from_safetensors(image_encoder_weights)
    ip_adapter.inject()

    depth_controlnet = SD1ControlnetAdapter(
        sd15.unet,
        name="depth",
        scale=1.0,
        weights=load_from_safetensors(depth_cn_weights_path),
    ).inject()

    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)
    clip_image_embedding = ip_adapter.compute_clip_image_embedding(ip_adapter.preprocess_image(input_image))

    negative_text_embedding, conditional_text_embedding = clip_text_embedding.chunk(2)
    negative_image_embedding, conditional_image_embedding = clip_image_embedding.chunk(2)

    clip_text_embedding = torch.cat(
        (
            torch.cat([negative_text_embedding, negative_image_embedding], dim=1),
            torch.cat([conditional_text_embedding, conditional_image_embedding], dim=1),
        )
    )

    depth_cn_condition = image_to_tensor(
        depth_condition_image.convert("RGB"),
        device=test_device,
        dtype=torch.float16,
    )

    sd15.set_num_inference_steps(n_steps)

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device, dtype=torch.float16)

    for step in sd15.steps:
        depth_controlnet.set_controlnet_condition(depth_cn_condition)
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.decode_latents(x)

    ensure_similar_images(predicted_image, expected_image_ip_adapter_controlnet)


@torch.no_grad()
def test_diffusion_ip_adapter_plus(
    sd15_ddim_lda_ft_mse: StableDiffusion_1,
    ip_adapter_plus_weights: Path,
    image_encoder_weights: Path,
    statue_image: Image.Image,
    expected_image_ip_adapter_plus_statue: Image.Image,
    test_device: torch.device,
):
    sd15 = sd15_ddim_lda_ft_mse.to(dtype=torch.float16)
    n_steps = 50

    prompt = "best quality, high quality"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    ip_adapter = SD1IPAdapter(
        target=sd15.unet, weights=load_from_safetensors(ip_adapter_plus_weights), fine_grained=True
    )
    ip_adapter.clip_image_encoder.load_from_safetensors(image_encoder_weights)
    ip_adapter.inject()

    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)
    clip_image_embedding = ip_adapter.compute_clip_image_embedding(ip_adapter.preprocess_image(statue_image))

    negative_text_embedding, conditional_text_embedding = clip_text_embedding.chunk(2)
    negative_image_embedding, conditional_image_embedding = clip_image_embedding.chunk(2)

    clip_text_embedding = torch.cat(
        (
            torch.cat([negative_text_embedding, negative_image_embedding], dim=1),
            torch.cat([conditional_text_embedding, conditional_image_embedding], dim=1),
        )
    )

    sd15.set_num_inference_steps(n_steps)

    manual_seed(42)  # seed=42 is used in the official IP-Adapter demo
    x = torch.randn(1, 4, 64, 64, device=test_device, dtype=torch.float16)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.decode_latents(x)

    ensure_similar_images(predicted_image, expected_image_ip_adapter_plus_statue, min_psnr=35, min_ssim=0.98)


@torch.no_grad()
def test_diffusion_sdxl_ip_adapter_plus(
    sdxl_ddim: StableDiffusion_XL,
    sdxl_ip_adapter_plus_weights: Path,
    image_encoder_weights: Path,
    woman_image: Image.Image,
    expected_image_sdxl_ip_adapter_plus_woman: Image.Image,
    test_device: torch.device,
):
    sdxl = sdxl_ddim.to(dtype=torch.float16)
    n_steps = 30

    prompt = "best quality, high quality"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    ip_adapter = SDXLIPAdapter(
        target=sdxl.unet, weights=load_from_safetensors(sdxl_ip_adapter_plus_weights), fine_grained=True
    )
    ip_adapter.clip_image_encoder.load_from_safetensors(image_encoder_weights)
    ip_adapter.inject()

    clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
        text=prompt, negative_text=negative_prompt
    )
    clip_image_embedding = ip_adapter.compute_clip_image_embedding(ip_adapter.preprocess_image(woman_image))

    negative_text_embedding, conditional_text_embedding = clip_text_embedding.chunk(2)
    negative_image_embedding, conditional_image_embedding = clip_image_embedding.chunk(2)

    clip_text_embedding = torch.cat(
        (
            torch.cat([negative_text_embedding, negative_image_embedding], dim=1),
            torch.cat([conditional_text_embedding, conditional_image_embedding], dim=1),
        )
    )
    time_ids = sdxl.default_time_ids
    sdxl.set_num_inference_steps(n_steps)

    manual_seed(2)
    x = torch.randn(1, 4, 128, 128, device=test_device, dtype=torch.float16)

    for step in sdxl.steps:
        x = sdxl(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            time_ids=time_ids,
            condition_scale=5,
        )
    sdxl.lda.to(dtype=torch.float32)
    predicted_image = sdxl.lda.decode_latents(x.to(dtype=torch.float32))

    ensure_similar_images(predicted_image, expected_image_sdxl_ip_adapter_plus_woman)


@torch.no_grad()
def test_sdxl_random_init(
    sdxl_ddim: StableDiffusion_XL, expected_sdxl_ddim_random_init: Image.Image, test_device: torch.device
) -> None:
    sdxl = sdxl_ddim
    expected_image = expected_sdxl_ddim_random_init
    n_steps = 30

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"

    clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
        text=prompt, negative_text=negative_prompt
    )
    time_ids = sdxl.default_time_ids

    sdxl.set_num_inference_steps(num_inference_steps=n_steps)

    manual_seed(seed=2)
    x = torch.randn(1, 4, 128, 128, device=test_device)

    for step in sdxl.steps:
        x = sdxl(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            time_ids=time_ids,
            condition_scale=5,
        )
    predicted_image = sdxl.lda.decode_latents(x=x)

    ensure_similar_images(img_1=predicted_image, img_2=expected_image, min_psnr=35, min_ssim=0.98)


@torch.no_grad()
def test_sdxl_random_init_sag(
    sdxl_ddim: StableDiffusion_XL, expected_sdxl_ddim_random_init_sag: Image.Image, test_device: torch.device
) -> None:
    sdxl = sdxl_ddim
    expected_image = expected_sdxl_ddim_random_init_sag
    n_steps = 30

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"

    clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
        text=prompt, negative_text=negative_prompt
    )
    time_ids = sdxl.default_time_ids

    sdxl.set_num_inference_steps(num_inference_steps=n_steps)
    sdxl.set_self_attention_guidance(enable=True, scale=0.75)

    manual_seed(seed=2)
    x = torch.randn(1, 4, 128, 128, device=test_device)

    for step in sdxl.steps:
        x = sdxl(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            time_ids=time_ids,
            condition_scale=5,
        )
    predicted_image = sdxl.lda.decode_latents(x=x)

    ensure_similar_images(img_1=predicted_image, img_2=expected_image)


@torch.no_grad()
def test_multi_diffusion(sd15_ddim: StableDiffusion_1, expected_multi_diffusion: Image.Image) -> None:
    manual_seed(seed=2)
    sd = sd15_ddim
    multi_diffusion = SD1MultiDiffusion(sd)
    clip_text_embedding = sd.compute_clip_text_embedding(text="a panorama of a mountain")
    target_1 = DiffusionTarget(
        size=(64, 64),
        offset=(0, 0),
        clip_text_embedding=clip_text_embedding,
        start_step=0,
    )
    target_2 = DiffusionTarget(
        size=(64, 64),
        offset=(0, 16),
        clip_text_embedding=clip_text_embedding,
        start_step=0,
    )
    noise = torch.randn(1, 4, 64, 80, device=sd.device, dtype=sd.dtype)
    x = noise
    for step in sd.steps:
        x = multi_diffusion(
            x,
            noise=noise,
            step=step,
            targets=[target_1, target_2],
        )
    result = sd.lda.decode_latents(x=x)
    ensure_similar_images(img_1=result, img_2=expected_multi_diffusion, min_psnr=35, min_ssim=0.98)


@torch.no_grad()
def test_t2i_adapter_depth(
    sd15_std: StableDiffusion_1,
    t2i_adapter_data_depth: tuple[str, Image.Image, Image.Image, Path],
    test_device: torch.device,
):
    sd15 = sd15_std
    n_steps = 30

    name, condition_image, expected_image, weights_path = t2i_adapter_data_depth

    if not weights_path.is_file():
        warn(f"could not find weights at {weights_path}, skipping")
        pytest.skip(allow_module_level=True)

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

    sd15.set_num_inference_steps(n_steps)

    t2i_adapter = SD1T2IAdapter(target=sd15.unet, name=name, weights=load_from_safetensors(weights_path)).inject()

    condition = image_to_tensor(condition_image.convert("RGB"), device=test_device)
    t2i_adapter.set_condition_features(features=t2i_adapter.compute_condition_features(condition))

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.decode_latents(x)

    ensure_similar_images(predicted_image, expected_image)


@torch.no_grad()
def test_t2i_adapter_xl_canny(
    sdxl_ddim: StableDiffusion_XL,
    t2i_adapter_xl_data_canny: tuple[str, Image.Image, Image.Image, Path],
    test_device: torch.device,
):
    sdxl = sdxl_ddim
    n_steps = 30

    name, condition_image, expected_image, weights_path = t2i_adapter_xl_data_canny

    if not weights_path.is_file():
        warn(f"could not find weights at {weights_path}, skipping")
        pytest.skip(allow_module_level=True)

    prompt = "Mystical fairy in real, magic, 4k picture, high quality"
    negative_prompt = (
        "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"
    )
    clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
        text=prompt, negative_text=negative_prompt
    )
    time_ids = sdxl.default_time_ids

    sdxl.set_num_inference_steps(n_steps)

    t2i_adapter = SDXLT2IAdapter(target=sdxl.unet, name=name, weights=load_from_safetensors(weights_path)).inject()
    t2i_adapter.set_scale(0.8)

    condition = image_to_tensor(condition_image.convert("RGB"), device=test_device)
    t2i_adapter.set_condition_features(features=t2i_adapter.compute_condition_features(condition))

    manual_seed(2)
    x = torch.randn(1, 4, condition_image.height // 8, condition_image.width // 8, device=test_device)

    for step in sdxl.steps:
        x = sdxl(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            time_ids=time_ids,
            condition_scale=7.5,
        )
    predicted_image = sdxl.lda.decode_latents(x)

    ensure_similar_images(predicted_image, expected_image)
