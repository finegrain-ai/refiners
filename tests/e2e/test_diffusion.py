import gc
from pathlib import Path
from typing import Iterator
from warnings import warn

import pytest
import torch
from PIL import Image

from refiners.fluxion.layers.attentions import ScaledDotProductAttention
from refiners.fluxion.utils import image_to_tensor, load_from_safetensors, load_tensors, manual_seed, no_grad
from refiners.foundationals.clip.concepts import ConceptExtender
from refiners.foundationals.latent_diffusion import (
    SD1ControlnetAdapter,
    SD1IPAdapter,
    SD1T2IAdapter,
    SD1UNet,
    SDFreeUAdapter,
    SDXLIPAdapter,
    SDXLT2IAdapter,
    StableDiffusion_1,
    StableDiffusion_1_Inpainting,
)
from refiners.foundationals.latent_diffusion.lora import SDLoraManager
from refiners.foundationals.latent_diffusion.multi_diffusion import DiffusionTarget
from refiners.foundationals.latent_diffusion.reference_only_control import ReferenceOnlyControlAdapter
from refiners.foundationals.latent_diffusion.restart import Restart
from refiners.foundationals.latent_diffusion.solvers import DDIM, Euler, NoiseSchedule
from refiners.foundationals.latent_diffusion.stable_diffusion_1.multi_diffusion import SD1MultiDiffusion
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.model import StableDiffusion_XL
from tests.utils import ensure_similar_images


@pytest.fixture(autouse=True)
def ensure_gc():
    # Avoid GPU OOMs
    # See https://github.com/pytest-dev/pytest/discussions/8153#discussioncomment-214812
    gc.collect()


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
def expected_image_std_random_init_euler(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_std_random_init_euler.png").convert("RGB")


@pytest.fixture
def expected_karras_random_init(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_karras_random_init.png").convert("RGB")


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
def expected_image_ip_adapter_multi(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_image_ip_adapter_multi.png").convert("RGB")


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
    return Image.open(ref_path / "expected_cutecat_sdxl_ddim_random_init.png").convert("RGB")


@pytest.fixture
def expected_sdxl_ddim_random_init_sag(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_cutecat_sdxl_ddim_random_init_sag.png").convert("RGB")


@pytest.fixture
def expected_sdxl_euler_random_init(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_cutecat_sdxl_euler_random_init.png").convert("RGB")


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

    if not weights_path.is_file():
        warn(f"could not find weights at {weights_path}, skipping")
        pytest.skip(allow_module_level=True)

    return name, condition_image, expected_image, weights_path


@pytest.fixture(scope="module")
def lora_data_pokemon(ref_path: Path, test_weights_path: Path) -> tuple[Image.Image, dict[str, torch.Tensor]]:
    expected_image = Image.open(ref_path / "expected_lora_pokemon.png").convert("RGB")
    weights_path = test_weights_path / "loras" / "pokemon-lora" / "pytorch_lora_weights.bin"

    if not weights_path.is_file():
        warn(f"could not find weights at {weights_path}, skipping")
        pytest.skip(allow_module_level=True)

    tensors = load_tensors(weights_path)
    return expected_image, tensors


@pytest.fixture(scope="module")
def lora_data_dpo(ref_path: Path, test_weights_path: Path) -> tuple[Image.Image, dict[str, torch.Tensor]]:
    expected_image = Image.open(ref_path / "expected_sdxl_dpo_lora.png").convert("RGB")
    weights_path = test_weights_path / "loras" / "dpo-lora" / "pytorch_lora_weights.safetensors"

    if not weights_path.is_file():
        warn(f"could not find weights at {weights_path}, skipping")
        pytest.skip(allow_module_level=True)

    tensors = load_from_safetensors(weights_path)
    return expected_image, tensors


@pytest.fixture(scope="module")
def lora_sliders(test_weights_path: Path) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, float]]:
    weights_path = test_weights_path / "loras" / "sliders"

    if not weights_path.is_dir():
        warn(f"could not find weights at {weights_path}, skipping")
        pytest.skip(allow_module_level=True)

    return {
        "age": load_tensors(weights_path / "age.pt"),  # type: ignore
        "cartoon_style": load_tensors(weights_path / "cartoon_style.pt"),  # type: ignore
        "eyesize": load_tensors(weights_path / "eyesize.pt"),  # type: ignore
    }, {
        "age": 0.3,
        "cartoon_style": -0.2,
        "dpo": 1.4,
        "eyesize": -0.2,
    }


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
def expected_restart(ref_path: Path) -> Image.Image:
    return Image.open(fp=ref_path / "expected_restart.png").convert(mode="RGB")


@pytest.fixture
def expected_freeu(ref_path: Path) -> Image.Image:
    return Image.open(fp=ref_path / "expected_freeu.png").convert(mode="RGB")


@pytest.fixture
def expected_sdxl_multi_loras(ref_path: Path) -> Image.Image:
    return Image.open(fp=ref_path / "expected_sdxl_multi_loras.png").convert(mode="RGB")


@pytest.fixture
def hello_world_assets(ref_path: Path) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    assets = Path(__file__).parent.parent.parent / "assets"
    dropy = assets / "dropy_logo.png"
    image_prompt = assets / "dragon_quest_slime.jpg"
    condition_image = assets / "dropy_canny.png"
    return (
        Image.open(fp=dropy).convert(mode="RGB"),
        Image.open(fp=image_prompt).convert(mode="RGB"),
        Image.open(fp=condition_image).convert(mode="RGB"),
        Image.open(fp=ref_path / "expected_dropy_slime_9752.png").convert(mode="RGB"),
    )


@pytest.fixture
def text_embedding_textual_inversion(test_textual_inversion_path: Path) -> torch.Tensor:
    return load_tensors(test_textual_inversion_path / "gta5-artwork" / "learned_embeds.bin")["<gta5-artwork>"]


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

    ddim_solver = DDIM(num_inference_steps=20)
    sd15 = StableDiffusion_1(solver=ddim_solver, device=test_device)

    sd15.clip_text_encoder.load_from_safetensors(text_encoder_weights)
    sd15.lda.load_from_safetensors(lda_weights)
    sd15.unet.load_from_safetensors(unet_weights_std)

    return sd15


@pytest.fixture
def sd15_ddim_karras(
    text_encoder_weights: Path, lda_weights: Path, unet_weights_std: Path, test_device: torch.device
) -> StableDiffusion_1:
    if test_device.type == "cpu":
        warn("not running on CPU, skipping")
        pytest.skip()

    ddim_solver = DDIM(num_inference_steps=20, noise_schedule=NoiseSchedule.KARRAS)
    sd15 = StableDiffusion_1(solver=ddim_solver, device=test_device)

    sd15.clip_text_encoder.load_from_safetensors(text_encoder_weights)
    sd15.lda.load_from_safetensors(lda_weights)
    sd15.unet.load_from_safetensors(unet_weights_std)

    return sd15


@pytest.fixture
def sd15_euler(
    text_encoder_weights: Path, lda_weights: Path, unet_weights_std: Path, test_device: torch.device
) -> StableDiffusion_1:
    if test_device.type == "cpu":
        warn("not running on CPU, skipping")
        pytest.skip()

    euler_solver = Euler(num_inference_steps=30)
    sd15 = StableDiffusion_1(solver=euler_solver, device=test_device)

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

    ddim_solver = DDIM(num_inference_steps=20)
    sd15 = StableDiffusion_1(solver=ddim_solver, device=test_device)

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
def sdxl_lda_fp16_fix_weights(test_weights_path: Path) -> Path:
    sdxl_lda_weights = test_weights_path / "sdxl-lda-fp16-fix.safetensors"
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

    solver = DDIM(num_inference_steps=30)
    sdxl = StableDiffusion_XL(solver=solver, device=test_device)

    sdxl.clip_text_encoder.load_from_safetensors(tensors_path=sdxl_text_encoder_weights)
    sdxl.lda.load_from_safetensors(tensors_path=sdxl_lda_weights)
    sdxl.unet.load_from_safetensors(tensors_path=sdxl_unet_weights)

    return sdxl


@pytest.fixture
def sdxl_ddim_lda_fp16_fix(
    sdxl_text_encoder_weights: Path, sdxl_lda_fp16_fix_weights: Path, sdxl_unet_weights: Path, test_device: torch.device
) -> StableDiffusion_XL:
    if test_device.type == "cpu":
        warn(message="not running on CPU, skipping")
        pytest.skip()

    solver = DDIM(num_inference_steps=30)
    sdxl = StableDiffusion_XL(solver=solver, device=test_device)

    sdxl.clip_text_encoder.load_from_safetensors(tensors_path=sdxl_text_encoder_weights)
    sdxl.lda.load_from_safetensors(tensors_path=sdxl_lda_fp16_fix_weights)
    sdxl.unet.load_from_safetensors(tensors_path=sdxl_unet_weights)

    return sdxl


@pytest.fixture
def sdxl_euler_deterministic(sdxl_ddim: StableDiffusion_XL) -> StableDiffusion_XL:
    return StableDiffusion_XL(
        unet=sdxl_ddim.unet,
        lda=sdxl_ddim.lda,
        clip_text_encoder=sdxl_ddim.clip_text_encoder,
        solver=Euler(num_inference_steps=30),
        device=sdxl_ddim.device,
        dtype=sdxl_ddim.dtype,
    )


@no_grad()
def test_diffusion_std_random_init(
    sd15_std: StableDiffusion_1, expected_image_std_random_init: Image.Image, test_device: torch.device
):
    sd15 = sd15_std

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

    sd15.set_inference_steps(30)

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_image_std_random_init)


@no_grad()
def test_diffusion_std_random_init_euler(
    sd15_euler: StableDiffusion_1, expected_image_std_random_init_euler: Image.Image, test_device: torch.device
):
    sd15 = sd15_euler
    euler_solver = sd15_euler.solver
    assert isinstance(euler_solver, Euler)

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

    sd15.set_inference_steps(30)

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device)
    x = x * euler_solver.init_noise_sigma

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_image_std_random_init_euler)


@no_grad()
def test_diffusion_karras_random_init(
    sd15_ddim_karras: StableDiffusion_1, expected_karras_random_init: Image.Image, test_device: torch.device
):
    sd15 = sd15_ddim_karras

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_karras_random_init, min_psnr=35, min_ssim=0.98)


@no_grad()
def test_diffusion_std_random_init_float16(
    sd15_std_float16: StableDiffusion_1, expected_image_std_random_init: Image.Image, test_device: torch.device
):
    sd15 = sd15_std_float16

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)
    assert clip_text_embedding.dtype == torch.float16

    sd15.set_inference_steps(30)

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device, dtype=torch.float16)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_image_std_random_init, min_psnr=35, min_ssim=0.98)


@no_grad()
def test_diffusion_std_random_init_sag(
    sd15_std: StableDiffusion_1, expected_image_std_random_init_sag: Image.Image, test_device: torch.device
):
    sd15 = sd15_std

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

    sd15.set_inference_steps(30)
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
    predicted_image = sd15.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_image_std_random_init_sag)


@no_grad()
def test_diffusion_std_init_image(
    sd15_std: StableDiffusion_1,
    cutecat_init: Image.Image,
    expected_image_std_init_image: Image.Image,
):
    sd15 = sd15_std

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

    sd15.set_inference_steps(35, first_step=5)

    manual_seed(2)
    x = sd15.init_latents((512, 512), cutecat_init)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_image_std_init_image)


@no_grad()
def test_rectangular_init_latents(
    sd15_std: StableDiffusion_1,
    cutecat_init: Image.Image,
):
    sd15 = sd15_std

    # Just check latents initialization with a non-square image (and not the entire diffusion)
    width, height = 512, 504
    rect_init_image = cutecat_init.crop((0, 0, width, height))
    x = sd15.init_latents((height, width), rect_init_image)

    assert sd15.lda.latents_to_image(x).size == (width, height)


@no_grad()
def test_diffusion_inpainting(
    sd15_inpainting: StableDiffusion_1_Inpainting,
    kitchen_dog: Image.Image,
    kitchen_dog_mask: Image.Image,
    expected_image_std_inpainting: Image.Image,
    test_device: torch.device,
):
    sd15 = sd15_inpainting

    prompt = "a large white cat, detailed high-quality professional image, sitting on a chair, in a kitchen"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

    sd15.set_inference_steps(30)
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
    predicted_image = sd15.lda.latents_to_image(x)

    # PSNR and SSIM values are large because with float32 we get large differences even v.s. ourselves.
    ensure_similar_images(predicted_image, expected_image_std_inpainting, min_psnr=25, min_ssim=0.95)


@no_grad()
def test_diffusion_inpainting_float16(
    sd15_inpainting_float16: StableDiffusion_1_Inpainting,
    kitchen_dog: Image.Image,
    kitchen_dog_mask: Image.Image,
    expected_image_std_inpainting: Image.Image,
    test_device: torch.device,
):
    sd15 = sd15_inpainting_float16

    prompt = "a large white cat, detailed high-quality professional image, sitting on a chair, in a kitchen"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)
    assert clip_text_embedding.dtype == torch.float16

    sd15.set_inference_steps(30)
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
    predicted_image = sd15.lda.latents_to_image(x)

    # PSNR and SSIM values are large because float16 is even worse than float32.
    ensure_similar_images(predicted_image, expected_image_std_inpainting, min_psnr=20, min_ssim=0.92)


@no_grad()
def test_diffusion_controlnet(
    sd15_std: StableDiffusion_1,
    controlnet_data: tuple[str, Image.Image, Image.Image, Path],
    test_device: torch.device,
):
    sd15 = sd15_std

    cn_name, condition_image, expected_image, cn_weights_path = controlnet_data

    if not cn_weights_path.is_file():
        warn(f"could not find weights at {cn_weights_path}, skipping")
        pytest.skip(allow_module_level=True)

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

    sd15.set_inference_steps(30)

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
    predicted_image = sd15.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_image, min_psnr=35, min_ssim=0.98)


@no_grad()
def test_diffusion_controlnet_structural_copy(
    sd15_std: StableDiffusion_1,
    controlnet_data_canny: tuple[str, Image.Image, Image.Image, Path],
    test_device: torch.device,
):
    sd15_base = sd15_std
    sd15 = sd15_base.structural_copy()

    cn_name, condition_image, expected_image, cn_weights_path = controlnet_data_canny

    if not cn_weights_path.is_file():
        warn(f"could not find weights at {cn_weights_path}, skipping")
        pytest.skip(allow_module_level=True)

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

    sd15.set_inference_steps(30)

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
    predicted_image = sd15.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_image, min_psnr=35, min_ssim=0.98)


@no_grad()
def test_diffusion_controlnet_float16(
    sd15_std_float16: StableDiffusion_1,
    controlnet_data_canny: tuple[str, Image.Image, Image.Image, Path],
    test_device: torch.device,
):
    sd15 = sd15_std_float16

    cn_name, condition_image, expected_image, cn_weights_path = controlnet_data_canny

    if not cn_weights_path.is_file():
        warn(f"could not find weights at {cn_weights_path}, skipping")
        pytest.skip(allow_module_level=True)

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

    sd15.set_inference_steps(30)

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
    predicted_image = sd15.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_image, min_psnr=35, min_ssim=0.98)


@no_grad()
def test_diffusion_controlnet_stack(
    sd15_std: StableDiffusion_1,
    controlnet_data_depth: tuple[str, Image.Image, Image.Image, Path],
    controlnet_data_canny: tuple[str, Image.Image, Image.Image, Path],
    expected_image_controlnet_stack: Image.Image,
    test_device: torch.device,
):
    sd15 = sd15_std

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

    sd15.set_inference_steps(30)

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
    predicted_image = sd15.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_image_controlnet_stack, min_psnr=35, min_ssim=0.98)


@no_grad()
def test_diffusion_lora(
    sd15_std: StableDiffusion_1,
    lora_data_pokemon: tuple[Image.Image, dict[str, torch.Tensor]],
    test_device: torch.device,
) -> None:
    sd15 = sd15_std

    expected_image, lora_weights = lora_data_pokemon

    prompt = "a cute cat"
    clip_text_embedding = sd15.compute_clip_text_embedding(prompt)

    sd15.set_inference_steps(30)

    SDLoraManager(sd15).add_loras("pokemon", lora_weights, scale=1)

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_image, min_psnr=35, min_ssim=0.98)


@no_grad()
def test_diffusion_sdxl_lora(
    sdxl_ddim: StableDiffusion_XL,
    lora_data_dpo: tuple[Image.Image, dict[str, torch.Tensor]],
) -> None:
    sdxl = sdxl_ddim
    expected_image, lora_weights = lora_data_dpo

    # parameters are the same as https://huggingface.co/radames/sdxl-DPO-LoRA
    # except that we are using DDIM instead of sde-dpmsolver++
    seed = 12341234123
    guidance_scale = 7.5
    lora_scale = 1.4
    prompt = "professional portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography"
    negative_prompt = "3d render, cartoon, drawing, art, low light, blur, pixelated, low resolution, black and white"

    SDLoraManager(sdxl).add_loras("dpo", lora_weights, scale=lora_scale)

    clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
        text=prompt, negative_text=negative_prompt
    )

    time_ids = sdxl.default_time_ids
    sdxl.set_inference_steps(40)

    manual_seed(seed=seed)
    x = torch.randn(1, 4, 128, 128, device=sdxl.device, dtype=sdxl.dtype)

    for step in sdxl.steps:
        x = sdxl(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            time_ids=time_ids,
            condition_scale=guidance_scale,
        )

    predicted_image = sdxl.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_image, min_psnr=35, min_ssim=0.98)


@no_grad()
def test_diffusion_sdxl_multiple_loras(
    sdxl_ddim: StableDiffusion_XL,
    lora_data_dpo: tuple[Image.Image, dict[str, torch.Tensor]],
    lora_sliders: tuple[dict[str, dict[str, torch.Tensor]], dict[str, float]],
    expected_sdxl_multi_loras: Image.Image,
) -> None:
    sdxl = sdxl_ddim
    expected_image = expected_sdxl_multi_loras
    _, dpo = lora_data_dpo
    loras, scales = lora_sliders
    loras["dpo"] = dpo

    SDLoraManager(sdxl).add_multiple_loras(loras, scales)

    # parameters are the same as https://huggingface.co/radames/sdxl-DPO-LoRA
    # except that we are using DDIM instead of sde-dpmsolver++
    n_steps = 40
    seed = 12341234123
    guidance_scale = 4
    prompt = "professional portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography"
    negative_prompt = "3d render, cartoon, drawing, art, low light, blur, pixelated, low resolution, black and white"

    clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
        text=prompt, negative_text=negative_prompt
    )

    time_ids = sdxl.default_time_ids
    sdxl.set_inference_steps(n_steps)

    manual_seed(seed=seed)
    x = torch.randn(1, 4, 128, 128, device=sdxl.device, dtype=sdxl.dtype)

    for step in sdxl.steps:
        x = sdxl(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            time_ids=time_ids,
            condition_scale=guidance_scale,
        )

    predicted_image = sdxl.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_image, min_psnr=35, min_ssim=0.98)


@no_grad()
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

    guide = sd15.lda.image_to_latents(condition_image_refonly)
    guide = torch.cat((guide, guide))

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device)

    for step in sd15.steps:
        noise = torch.randn(2, 4, 64, 64, device=test_device)
        noised_guide = sd15.solver.add_noise(guide, noise, step)
        refonly_adapter.set_controlnet_condition(noised_guide)
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
        torch.randn(2, 4, 64, 64, device=test_device)  # for SD Web UI reproductibility only
    predicted_image = sd15.lda.latents_to_image(x)

    # min_psnr lowered to 33 because this reference image was generated without noise removal (see #192)
    ensure_similar_images(predicted_image, expected_image_refonly, min_psnr=33, min_ssim=0.99)


@no_grad()
def test_diffusion_inpainting_refonly(
    sd15_inpainting: StableDiffusion_1_Inpainting,
    scene_image_inpainting_refonly: Image.Image,
    target_image_inpainting_refonly: Image.Image,
    mask_image_inpainting_refonly: Image.Image,
    expected_image_inpainting_refonly: Image.Image,
    test_device: torch.device,
):
    sd15 = sd15_inpainting

    prompt = ""  # unconditional
    clip_text_embedding = sd15.compute_clip_text_embedding(prompt)

    refonly_adapter = ReferenceOnlyControlAdapter(sd15.unet).inject()

    sd15.set_inference_steps(30)
    sd15.set_inpainting_conditions(target_image_inpainting_refonly, mask_image_inpainting_refonly)

    guide = sd15.lda.image_to_latents(scene_image_inpainting_refonly)
    guide = torch.cat((guide, guide))

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device)

    for step in sd15.steps:
        noise = torch.randn_like(guide)
        noised_guide = sd15.solver.add_noise(guide, noise, step)
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
    predicted_image = sd15.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_image_inpainting_refonly, min_psnr=35, min_ssim=0.99)


@no_grad()
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

    prompt = "a cute cat on a <gta5-artwork>"
    clip_text_embedding = sd15.compute_clip_text_embedding(prompt)

    sd15.set_inference_steps(30)

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_image_textual_inversion_random_init, min_psnr=35, min_ssim=0.98)


@no_grad()
def test_diffusion_ip_adapter(
    sd15_ddim_lda_ft_mse: StableDiffusion_1,
    ip_adapter_weights: Path,
    image_encoder_weights: Path,
    woman_image: Image.Image,
    expected_image_ip_adapter_woman: Image.Image,
    test_device: torch.device,
):
    sd15 = sd15_ddim_lda_ft_mse.to(dtype=torch.float16)

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
    ip_adapter.set_clip_image_embedding(clip_image_embedding)

    sd15.set_inference_steps(50)

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device, dtype=torch.float16)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_image_ip_adapter_woman)


@no_grad()
def test_diffusion_ip_adapter_multi(
    sd15_ddim_lda_ft_mse: StableDiffusion_1,
    ip_adapter_weights: Path,
    image_encoder_weights: Path,
    woman_image: Image.Image,
    statue_image: Image.Image,
    expected_image_ip_adapter_multi: Image.Image,
    test_device: torch.device,
):
    sd15 = sd15_ddim_lda_ft_mse.to(dtype=torch.float16)

    prompt = "best quality, high quality"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    ip_adapter = SD1IPAdapter(target=sd15.unet, weights=load_from_safetensors(ip_adapter_weights))
    ip_adapter.clip_image_encoder.load_from_safetensors(image_encoder_weights)
    ip_adapter.inject()

    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)
    clip_image_embedding = ip_adapter.compute_clip_image_embedding([woman_image, statue_image], weights=[1.0, 1.4])
    ip_adapter.set_clip_image_embedding(clip_image_embedding)

    sd15.set_inference_steps(50)

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

    ensure_similar_images(predicted_image, expected_image_ip_adapter_multi)


@no_grad()
def test_diffusion_sdxl_ip_adapter(
    sdxl_ddim: StableDiffusion_XL,
    sdxl_ip_adapter_weights: Path,
    image_encoder_weights: Path,
    woman_image: Image.Image,
    expected_image_sdxl_ip_adapter_woman: Image.Image,
    test_device: torch.device,
):
    sdxl = sdxl_ddim.to(dtype=torch.float16)

    prompt = "best quality, high quality"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    ip_adapter = SDXLIPAdapter(target=sdxl.unet, weights=load_from_safetensors(sdxl_ip_adapter_weights))
    ip_adapter.clip_image_encoder.load_from_safetensors(image_encoder_weights)
    ip_adapter.inject()

    with no_grad():
        clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
            text=prompt, negative_text=negative_prompt
        )
        clip_image_embedding = ip_adapter.compute_clip_image_embedding(ip_adapter.preprocess_image(woman_image))
        ip_adapter.set_clip_image_embedding(clip_image_embedding)

    time_ids = sdxl.default_time_ids
    sdxl.set_inference_steps(30)

    manual_seed(2)
    x = torch.randn(1, 4, 128, 128, device=test_device, dtype=torch.float16)

    with no_grad():
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
        predicted_image = sdxl.lda.latents_to_image(x.to(dtype=torch.float32))

    ensure_similar_images(predicted_image, expected_image_sdxl_ip_adapter_woman)


@no_grad()
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
    ip_adapter.set_clip_image_embedding(clip_image_embedding)

    depth_cn_condition = image_to_tensor(
        depth_condition_image.convert("RGB"),
        device=test_device,
        dtype=torch.float16,
    )

    sd15.set_inference_steps(50)

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
    predicted_image = sd15.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_image_ip_adapter_controlnet)


@no_grad()
def test_diffusion_ip_adapter_plus(
    sd15_ddim_lda_ft_mse: StableDiffusion_1,
    ip_adapter_plus_weights: Path,
    image_encoder_weights: Path,
    statue_image: Image.Image,
    expected_image_ip_adapter_plus_statue: Image.Image,
    test_device: torch.device,
):
    sd15 = sd15_ddim_lda_ft_mse.to(dtype=torch.float16)

    prompt = "best quality, high quality"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    ip_adapter = SD1IPAdapter(
        target=sd15.unet, weights=load_from_safetensors(ip_adapter_plus_weights), fine_grained=True
    )
    ip_adapter.clip_image_encoder.load_from_safetensors(image_encoder_weights)
    ip_adapter.inject()

    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)
    clip_image_embedding = ip_adapter.compute_clip_image_embedding(ip_adapter.preprocess_image(statue_image))
    ip_adapter.set_clip_image_embedding(clip_image_embedding)

    sd15.set_inference_steps(50)

    manual_seed(42)  # seed=42 is used in the official IP-Adapter demo
    x = torch.randn(1, 4, 64, 64, device=test_device, dtype=torch.float16)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_image_ip_adapter_plus_statue, min_psnr=35, min_ssim=0.98)


@no_grad()
def test_diffusion_sdxl_ip_adapter_plus(
    sdxl_ddim: StableDiffusion_XL,
    sdxl_ip_adapter_plus_weights: Path,
    image_encoder_weights: Path,
    woman_image: Image.Image,
    expected_image_sdxl_ip_adapter_plus_woman: Image.Image,
    test_device: torch.device,
):
    sdxl = sdxl_ddim.to(dtype=torch.float16)

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
    ip_adapter.set_clip_image_embedding(clip_image_embedding)

    time_ids = sdxl.default_time_ids
    sdxl.set_inference_steps(30)

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
    predicted_image = sdxl.lda.latents_to_image(x.to(dtype=torch.float32))

    ensure_similar_images(predicted_image, expected_image_sdxl_ip_adapter_plus_woman)


@no_grad()
def test_sdxl_random_init(
    sdxl_ddim: StableDiffusion_XL, expected_sdxl_ddim_random_init: Image.Image, test_device: torch.device
) -> None:
    sdxl = sdxl_ddim
    expected_image = expected_sdxl_ddim_random_init

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"

    clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
        text=prompt, negative_text=negative_prompt
    )
    time_ids = sdxl.default_time_ids

    sdxl.set_inference_steps(30)

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
    predicted_image = sdxl.lda.latents_to_image(x=x)

    ensure_similar_images(img_1=predicted_image, img_2=expected_image, min_psnr=35, min_ssim=0.98)


@no_grad()
def test_sdxl_random_init_sag(
    sdxl_ddim: StableDiffusion_XL, expected_sdxl_ddim_random_init_sag: Image.Image, test_device: torch.device
) -> None:
    sdxl = sdxl_ddim
    expected_image = expected_sdxl_ddim_random_init_sag

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"

    clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
        text=prompt, negative_text=negative_prompt
    )
    time_ids = sdxl.default_time_ids

    sdxl.set_inference_steps(30)
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
    predicted_image = sdxl.lda.latents_to_image(x=x)

    ensure_similar_images(img_1=predicted_image, img_2=expected_image)


@no_grad()
def test_diffusion_sdxl_sliced_attention(
    sdxl_ddim: StableDiffusion_XL, expected_sdxl_ddim_random_init: Image.Image
) -> None:
    unet = sdxl_ddim.unet.structural_copy()
    for layer in unet.layers(ScaledDotProductAttention):
        layer.slice_size = 2048

    sdxl = StableDiffusion_XL(
        unet=unet,
        lda=sdxl_ddim.lda,
        clip_text_encoder=sdxl_ddim.clip_text_encoder,
        solver=sdxl_ddim.solver,
        device=sdxl_ddim.device,
        dtype=sdxl_ddim.dtype,
    )

    expected_image = expected_sdxl_ddim_random_init

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"

    clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
        text=prompt, negative_text=negative_prompt
    )
    time_ids = sdxl.default_time_ids
    sdxl.set_inference_steps(30)
    manual_seed(2)
    x = torch.randn(1, 4, 128, 128, device=sdxl.device, dtype=sdxl.dtype)
    for step in sdxl.steps:
        x = sdxl(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            time_ids=time_ids,
            condition_scale=5,
        )

    predicted_image = sdxl.lda.decode_latents(x)
    ensure_similar_images(predicted_image, expected_image, min_psnr=35, min_ssim=0.98)


@no_grad()
def test_diffusion_sdxl_euler_deterministic(
    sdxl_euler_deterministic: StableDiffusion_XL, expected_sdxl_euler_random_init: Image.Image
) -> None:
    sdxl = sdxl_euler_deterministic
    assert isinstance(sdxl.solver, Euler)

    expected_image = expected_sdxl_euler_random_init

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"

    clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
        text=prompt, negative_text=negative_prompt
    )
    time_ids = sdxl.default_time_ids
    sdxl.set_inference_steps(30)
    manual_seed(2)
    x = torch.randn(1, 4, 128, 128, device=sdxl.device, dtype=sdxl.dtype)

    # init latents must be scaled for Euler
    # TODO make init_latents work
    x = x * sdxl.solver.init_noise_sigma

    for step in sdxl.steps:
        x = sdxl(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            time_ids=time_ids,
            condition_scale=5,
        )

    predicted_image = sdxl.lda.decode_latents(x)
    ensure_similar_images(predicted_image, expected_image)


@no_grad()
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
    result = sd.lda.latents_to_image(x=x)
    ensure_similar_images(img_1=result, img_2=expected_multi_diffusion, min_psnr=35, min_ssim=0.98)


@no_grad()
def test_t2i_adapter_depth(
    sd15_std: StableDiffusion_1,
    t2i_adapter_data_depth: tuple[str, Image.Image, Image.Image, Path],
    test_device: torch.device,
):
    sd15 = sd15_std

    name, condition_image, expected_image, weights_path = t2i_adapter_data_depth

    if not weights_path.is_file():
        warn(f"could not find weights at {weights_path}, skipping")
        pytest.skip(allow_module_level=True)

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

    sd15.set_inference_steps(30)

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
    predicted_image = sd15.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_image)


@no_grad()
def test_t2i_adapter_xl_canny(
    sdxl_ddim: StableDiffusion_XL,
    t2i_adapter_xl_data_canny: tuple[str, Image.Image, Image.Image, Path],
    test_device: torch.device,
):
    sdxl = sdxl_ddim

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

    sdxl.set_inference_steps(30)

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
    predicted_image = sdxl.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_image)


@no_grad()
def test_restart(
    sd15_ddim: StableDiffusion_1,
    expected_restart: Image.Image,
    test_device: torch.device,
):
    sd15 = sd15_ddim

    prompt = "a cute cat, detailed high-quality professional image"
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"

    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

    sd15.set_inference_steps(30)
    restart = Restart(ldm=sd15)

    manual_seed(2)
    x = torch.randn(1, 4, 64, 64, device=test_device)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=8,
        )

        if step == restart.start_step:
            x = restart(
                x,
                clip_text_embedding=clip_text_embedding,
                condition_scale=8,
            )

    predicted_image = sd15.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_restart, min_psnr=35, min_ssim=0.98)


@no_grad()
def test_freeu(
    sd15_std: StableDiffusion_1,
    expected_freeu: Image.Image,
):
    sd15 = sd15_std

    prompt = "best quality, high quality cute cat"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
    clip_text_embedding = sd15.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

    sd15.set_inference_steps(50, first_step=1)

    SDFreeUAdapter(
        sd15.unet, backbone_scales=[1.2, 1.2, 1.2, 1.4, 1.4, 1.4], skip_scales=[0.9, 0.9, 0.9, 0.2, 0.2, 0.2]
    ).inject()

    manual_seed(9752)
    x = sd15.init_latents((512, 512)).to(device=sd15.device, dtype=sd15.dtype)

    for step in sd15.steps:
        x = sd15(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=7.5,
        )
    predicted_image = sd15.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_freeu)


@no_grad()
def test_hello_world(
    sdxl_ddim_lda_fp16_fix: StableDiffusion_XL,
    t2i_adapter_xl_data_canny: tuple[str, Image.Image, Image.Image, Path],
    sdxl_ip_adapter_weights: Path,
    image_encoder_weights: Path,
    hello_world_assets: tuple[Image.Image, Image.Image, Image.Image, Image.Image],
) -> None:
    sdxl = sdxl_ddim_lda_fp16_fix.to(dtype=torch.float16)
    sdxl.dtype = torch.float16  # FIXME: should not be necessary

    name, _, _, weights_path = t2i_adapter_xl_data_canny
    init_image, image_prompt, condition_image, expected_image = hello_world_assets

    if not weights_path.is_file():
        warn(f"could not find weights at {weights_path}, skipping")
        pytest.skip(allow_module_level=True)

    ip_adapter = SDXLIPAdapter(target=sdxl.unet, weights=load_from_safetensors(sdxl_ip_adapter_weights))
    ip_adapter.clip_image_encoder.load_from_safetensors(image_encoder_weights)
    ip_adapter.inject()

    image_embedding = ip_adapter.compute_clip_image_embedding(ip_adapter.preprocess_image(image_prompt))
    ip_adapter.set_clip_image_embedding(image_embedding)

    # Note: default text prompts for IP-Adapter
    clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
        text="best quality, high quality", negative_text="monochrome, lowres, bad anatomy, worst quality, low quality"
    )
    time_ids = sdxl.default_time_ids

    t2i_adapter = SDXLT2IAdapter(target=sdxl.unet, name=name, weights=load_from_safetensors(weights_path)).inject()

    condition = image_to_tensor(condition_image.convert("RGB"), device=sdxl.device, dtype=sdxl.dtype)
    t2i_adapter.set_condition_features(features=t2i_adapter.compute_condition_features(condition))

    ip_adapter.set_scale(0.85)
    t2i_adapter.set_scale(0.8)
    sdxl.set_inference_steps(50, first_step=1)
    sdxl.set_self_attention_guidance(enable=True, scale=0.75)

    manual_seed(9752)
    x = sdxl.init_latents(size=(1024, 1024), init_image=init_image).to(device=sdxl.device, dtype=sdxl.dtype)
    for step in sdxl.steps:
        x = sdxl(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            time_ids=time_ids,
        )
    predicted_image = sdxl.lda.latents_to_image(x)

    ensure_similar_images(predicted_image, expected_image)
