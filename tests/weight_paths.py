from pathlib import Path

import pytest
import requests

from refiners.conversion import (
    autoencoder_sd15,
    autoencoder_sdxl,
    clip_image_sd21,
    clip_text_sd15,
    clip_text_sdxl,
    controlnet_sd15,
    dinov2,
    ella,
    hq_sam,
    ipadapter_sd15,
    ipadapter_sdxl,
    loras,
    mvanet,
    preprocessors,
    sam,
    t2iadapter_sd15,
    t2iadapter_sdxl,
    unet_sd15,
    unet_sdxl,
)
from refiners.conversion.utils import Hub


def get_path(hub: Hub, use_local_weights: bool) -> Path:
    if use_local_weights:
        path = hub.local_path
    else:
        if hub.download_url is not None:
            pytest.skip(f"{hub.filename} is not available on Hugging Face Hub")

        try:
            path = hub.hf_cache_path
        except requests.exceptions.HTTPError:
            pytest.skip(f"Could not download weights from {hub.hf_url}")

    if not path.is_file():
        pytest.skip(f"File not found: {path}")

    return path


######################################## CLIP ########################################


@pytest.fixture(scope="session")
def unclip21_transformers_stabilityai_path() -> str:
    return "stabilityai/stable-diffusion-2-1-unclip"


@pytest.fixture(scope="session")
def clip_image_encoder_huge_weights_path(use_local_weights: bool) -> Path:
    return get_path(clip_image_sd21.unclip_21.converted, use_local_weights)


######################################## SD1.5 ########################################


@pytest.fixture(scope="session")
def sd15_diffusers_runwayml_path() -> str:
    return "stable-diffusion-v1-5/stable-diffusion-v1-5"


@pytest.fixture(scope="session")
def sd15_text_encoder_weights_path(use_local_weights: bool) -> Path:
    return get_path(clip_text_sd15.runwayml.converted, use_local_weights)


@pytest.fixture(scope="session")
def sd15_autoencoder_weights_path(use_local_weights: bool) -> Path:
    return get_path(autoencoder_sd15.runwayml.converted, use_local_weights)


@pytest.fixture(scope="session")
def sd15_autoencoder_mse_weights_path(use_local_weights: bool) -> Path:
    return get_path(autoencoder_sd15.stability_mse.converted, use_local_weights)


@pytest.fixture(scope="session")
def sd15_unet_weights_path(use_local_weights: bool) -> Path:
    return get_path(unet_sd15.runwayml.converted, use_local_weights)


@pytest.fixture(scope="session")
def sd15_unet_inpainting_weights_path(use_local_weights: bool) -> Path:
    return get_path(unet_sd15.runwayml_inpainting.converted, use_local_weights)


######################################## SDXL ########################################


@pytest.fixture(scope="session")
def sdxl_diffusers_stabilityai_path() -> str:
    return "stabilityai/stable-diffusion-xl-base-1.0"


@pytest.fixture(scope="session")
def sdxl_autoencoder_weights_path(use_local_weights: bool) -> Path:
    return get_path(autoencoder_sdxl.stability.converted, use_local_weights)


@pytest.fixture(scope="session")
def sdxl_autoencoder_fp16fix_weights_path(use_local_weights: bool) -> Path:
    return get_path(autoencoder_sdxl.madebyollin_fp16fix.converted, use_local_weights)


@pytest.fixture(scope="session")
def sdxl_unet_weights_path(use_local_weights: bool) -> Path:
    return get_path(unet_sdxl.stability.converted, use_local_weights)


@pytest.fixture(scope="session")
def sdxl_unet_lcm_weights_path(use_local_weights: bool) -> Path:
    return get_path(unet_sdxl.lcm.converted, use_local_weights)


@pytest.fixture(scope="session")
def sdxl_unet_lightning_4step_weights_path(use_local_weights: bool) -> Path:
    return get_path(unet_sdxl.lightning_4step.converted, use_local_weights)


@pytest.fixture(scope="session")
def sdxl_unet_lightning_1step_weights_path(use_local_weights: bool) -> Path:
    return get_path(unet_sdxl.lightning_1step.converted, use_local_weights)


@pytest.fixture(scope="session")
def sdxl_text_encoder_weights_path(use_local_weights: bool) -> Path:
    return get_path(clip_text_sdxl.stability.converted, use_local_weights)


######################################## ControlNet ########################################


@pytest.fixture(scope="session")
def controlnet_canny_weights_path(use_local_weights: bool) -> Path:
    return get_path(controlnet_sd15.canny.converted, use_local_weights)


@pytest.fixture(scope="session")
def controlnet_depth_weights_path(use_local_weights: bool) -> Path:
    return get_path(controlnet_sd15.depth.converted, use_local_weights)


@pytest.fixture(scope="session")
def controlnet_lineart_weights_path(use_local_weights: bool) -> Path:
    return get_path(controlnet_sd15.lineart.converted, use_local_weights)


@pytest.fixture(scope="session")
def controlnet_normals_weights_path(use_local_weights: bool) -> Path:
    return get_path(controlnet_sd15.normalbae.converted, use_local_weights)


@pytest.fixture(scope="session")
def controlnet_sam_weights_path(use_local_weights: bool) -> Path:
    return get_path(controlnet_sd15.sam.converted, use_local_weights)


@pytest.fixture(scope="session")
def controlnet_tiles_weights_path(use_local_weights: bool) -> Path:
    return get_path(controlnet_sd15.tile.converted, use_local_weights)


@pytest.fixture(scope="session")
def controlnet_preprocessor_info_drawings_weights_path(use_local_weights: bool) -> Path:
    return get_path(preprocessors.informative_drawings.converted, use_local_weights)


######################################## IP Adapter ########################################


@pytest.fixture(scope="session")
def ip_adapter_sd15_weights_path(use_local_weights: bool) -> Path:
    return get_path(ipadapter_sd15.base.converted, use_local_weights)


@pytest.fixture(scope="session")
def ip_adapter_sd15_plus_weights_path(use_local_weights: bool) -> Path:
    return get_path(ipadapter_sd15.plus.converted, use_local_weights)


@pytest.fixture(scope="session")
def ip_adapter_sdxl_weights_path(use_local_weights: bool) -> Path:
    return get_path(ipadapter_sdxl.base.converted, use_local_weights)


@pytest.fixture(scope="session")
def ip_adapter_sdxl_plus_weights_path(use_local_weights: bool) -> Path:
    return get_path(ipadapter_sdxl.plus.converted, use_local_weights)


######################################## T2I ########################################


@pytest.fixture(scope="session")
def t2i_depth_weights_path(use_local_weights: bool) -> Path:
    return get_path(t2iadapter_sd15.depth.converted, use_local_weights)


@pytest.fixture(scope="session")
def t2i_sdxl_canny_weights_path(use_local_weights: bool) -> Path:
    return get_path(t2iadapter_sdxl.canny.converted, use_local_weights)


######################################## LoRA ########################################


@pytest.fixture(scope="session")
def lora_pokemon_weights_path(use_local_weights: bool) -> Path:
    return get_path(loras.sd15_pokemon, use_local_weights)


@pytest.fixture(scope="session")
def lora_dpo_weights_path(use_local_weights: bool) -> Path:
    return get_path(loras.sdxl_dpo, use_local_weights)


@pytest.fixture(scope="session")
def lora_slider_age_weights_path(use_local_weights: bool) -> Path:
    return get_path(loras.sdxl_age_slider, use_local_weights)


@pytest.fixture(scope="session")
def lora_slider_cartoon_style_weights_path(use_local_weights: bool) -> Path:
    return get_path(loras.sdxl_cartoon_slider, use_local_weights)


@pytest.fixture(scope="session")
def lora_slider_eyesize_weights_path(use_local_weights: bool) -> Path:
    return get_path(loras.sdxl_eyesize_slider, use_local_weights)


@pytest.fixture(scope="session")
def lora_sdxl_lcm_weights_path(use_local_weights: bool) -> Path:
    return get_path(loras.sdxl_lcm, use_local_weights)


@pytest.fixture(scope="session")
def lora_sdxl_lightning_4step_weights_path(use_local_weights: bool) -> Path:
    return get_path(loras.sdxl_lightning_4steps, use_local_weights)


@pytest.fixture(scope="session")
def lora_scifi_weights_path(use_local_weights: bool) -> Path:
    return get_path(loras.sdxl_scifi, use_local_weights)


@pytest.fixture(scope="session")
def lora_pixelart_weights_path(use_local_weights: bool) -> Path:
    return get_path(loras.sdxl_pixelart, use_local_weights)


######################################## IC Light ########################################


@pytest.fixture(scope="session")
def ic_light_sd15_fc_weights_path(use_local_weights: bool) -> Path:
    return get_path(unet_sd15.ic_light_fc.converted, use_local_weights)


@pytest.fixture(scope="session")
def ic_light_sd15_fcon_weights_path(use_local_weights: bool) -> Path:
    return get_path(unet_sd15.ic_light_fcon.converted, use_local_weights)


@pytest.fixture(scope="session")
def ic_light_sd15_fbc_weights_path(use_local_weights: bool) -> Path:
    return get_path(unet_sd15.ic_light_fbc.converted, use_local_weights)


######################################## ELLA ########################################


@pytest.fixture(scope="session")
def t5xl_transformers_path() -> str:
    return "google/flan-t5-xl"


@pytest.fixture(scope="session")
def ella_sd15_tsc_t5xl_weights_path(use_local_weights: bool) -> Path:
    return get_path(ella.sd15_t5xl.converted, use_local_weights)


######################################## MVANet ########################################


@pytest.fixture(scope="session")
def mvanet_weights_path(use_local_weights: bool) -> Path:
    return get_path(mvanet.mvanet.converted, use_local_weights)


@pytest.fixture(scope="session")
def box_segmenter_weights_path(use_local_weights: bool) -> Path:
    return get_path(mvanet.finegrain_v01, use_local_weights)


######################################## Segment Anything ########################################


@pytest.fixture(scope="session")
def sam_h_weights_path(use_local_weights: bool) -> Path:
    return get_path(sam.vit_h.converted, use_local_weights)


@pytest.fixture(scope="session")
def sam_h_unconverted_weights_path(use_local_weights: bool) -> Path:
    return get_path(sam.vit_h.original, use_local_weights)


@pytest.fixture(scope="session")
def sam_h_hq_adapter_weights_path(use_local_weights: bool) -> Path:
    return get_path(hq_sam.vit_h.converted, use_local_weights)


@pytest.fixture(scope="session")
def sam_h_hq_adapter_unconverted_weights_path(use_local_weights: bool) -> Path:
    return get_path(hq_sam.vit_h.original, use_local_weights)


######################################## DINOv2 ########################################


@pytest.fixture(scope="session")
def dinov2_small_weights_path(use_local_weights: bool) -> Path:
    return get_path(dinov2.small.converted, use_local_weights)


@pytest.fixture(scope="session")
def dinov2_small_unconverted_weights_path(use_local_weights: bool) -> Path:
    return get_path(dinov2.small.original, use_local_weights)


@pytest.fixture(scope="session")
def dinov2_small_reg4_weights_path(use_local_weights: bool) -> Path:
    return get_path(dinov2.small_reg.converted, use_local_weights)


@pytest.fixture(scope="session")
def dinov2_small_reg4_unconverted_weights_path(use_local_weights: bool) -> Path:
    return get_path(dinov2.small_reg.original, use_local_weights)


@pytest.fixture(scope="session")
def dinov2_base_weights_path(use_local_weights: bool) -> Path:
    return get_path(dinov2.base.converted, use_local_weights)


@pytest.fixture(scope="session")
def dinov2_base_unconverted_weights_path(use_local_weights: bool) -> Path:
    return get_path(dinov2.base.original, use_local_weights)


@pytest.fixture(scope="session")
def dinov2_base_reg4_weights_path(use_local_weights: bool) -> Path:
    return get_path(dinov2.base_reg.converted, use_local_weights)


@pytest.fixture(scope="session")
def dinov2_base_reg4_unconverted_weights_path(use_local_weights: bool) -> Path:
    return get_path(dinov2.base_reg.original, use_local_weights)


@pytest.fixture(scope="session")
def dinov2_large_weights_path(use_local_weights: bool) -> Path:
    return get_path(dinov2.large.converted, use_local_weights)


@pytest.fixture(scope="session")
def dinov2_large_unconverted_weights_path(use_local_weights: bool) -> Path:
    return get_path(dinov2.large.original, use_local_weights)


@pytest.fixture(scope="session")
def dinov2_large_reg4_weights_path(use_local_weights: bool) -> Path:
    return get_path(dinov2.large_reg.converted, use_local_weights)


@pytest.fixture(scope="session")
def dinov2_large_reg4_unconverted_weights_path(use_local_weights: bool) -> Path:
    return get_path(dinov2.large_reg.original, use_local_weights)


@pytest.fixture(scope="session")
def dinov2_giant_weights_path(use_local_weights: bool) -> Path:
    return get_path(dinov2.giant.converted, use_local_weights)


@pytest.fixture(scope="session")
def dinov2_giant_unconverted_weights_path(use_local_weights: bool) -> Path:
    return get_path(dinov2.giant.original, use_local_weights)


@pytest.fixture(scope="session")
def dinov2_giant_reg4_weights_path(use_local_weights: bool) -> Path:
    return get_path(dinov2.giant_reg.converted, use_local_weights)


@pytest.fixture(scope="session")
def dinov2_giant_reg4_unconverted_weights_path(use_local_weights: bool) -> Path:
    return get_path(dinov2.giant_reg.original, use_local_weights)
