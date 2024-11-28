from pathlib import Path

import pytest
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

from refiners.fluxion.utils import load_from_safetensors
from refiners.foundationals.latent_diffusion import (
    CLIPTextEncoderL,
    DoubleTextEncoder,
    SD1Autoencoder,
    SD1UNet,
    SDXLAutoencoder,
    SDXLUNet,
    StableDiffusion_1,
    StableDiffusion_XL,
)


@pytest.fixture(scope="module")
def refiners_sd15_autoencoder(sd15_autoencoder_weights_path: Path) -> SD1Autoencoder:
    autoencoder = SD1Autoencoder()
    tensors = load_from_safetensors(sd15_autoencoder_weights_path)
    autoencoder.load_state_dict(tensors)
    return autoencoder


@pytest.fixture(scope="module")
def refiners_sd15_unet(sd15_unet_weights_path: Path) -> SD1UNet:
    unet = SD1UNet(in_channels=4)
    tensors = load_from_safetensors(sd15_unet_weights_path)
    unet.load_state_dict(tensors)
    return unet


@pytest.fixture(scope="module")
def refiners_sd15_text_encoder(sd15_text_encoder_weights_path: Path) -> CLIPTextEncoderL:
    text_encoder = CLIPTextEncoderL()
    tensors = load_from_safetensors(sd15_text_encoder_weights_path)
    text_encoder.load_state_dict(tensors)
    return text_encoder


@pytest.fixture(scope="module")
def refiners_sd15(
    refiners_sd15_autoencoder: SD1Autoencoder,
    refiners_sd15_unet: SD1UNet,
    refiners_sd15_text_encoder: CLIPTextEncoderL,
) -> StableDiffusion_1:
    return StableDiffusion_1(
        lda=refiners_sd15_autoencoder,
        unet=refiners_sd15_unet,
        clip_text_encoder=refiners_sd15_text_encoder,
    )


@pytest.fixture(scope="module")
def refiners_sdxl_autoencoder(sdxl_autoencoder_weights_path: Path) -> SDXLAutoencoder:
    autoencoder = SDXLAutoencoder()
    tensors = load_from_safetensors(sdxl_autoencoder_weights_path)
    autoencoder.load_state_dict(tensors)
    return autoencoder


@pytest.fixture(scope="module")
def refiners_sdxl_unet(sdxl_unet_weights_path: Path) -> SDXLUNet:
    unet = SDXLUNet(in_channels=4)
    tensors = load_from_safetensors(sdxl_unet_weights_path)
    unet.load_state_dict(tensors)
    return unet


@pytest.fixture(scope="module")
def refiners_sdxl_text_encoder(sdxl_text_encoder_weights_path: Path) -> DoubleTextEncoder:
    text_encoder = DoubleTextEncoder()
    tensors = load_from_safetensors(sdxl_text_encoder_weights_path)
    text_encoder.load_state_dict(tensors)
    return text_encoder


@pytest.fixture(scope="module")
def refiners_sdxl(
    refiners_sdxl_autoencoder: SDXLAutoencoder,
    refiners_sdxl_unet: SDXLUNet,
    refiners_sd15_text_encoder: DoubleTextEncoder,
) -> StableDiffusion_XL:
    return StableDiffusion_XL(
        lda=refiners_sdxl_autoencoder,
        unet=refiners_sdxl_unet,
        clip_text_encoder=refiners_sd15_text_encoder,
    )


@pytest.fixture(scope="module")
def diffusers_sd15_pipeline(
    sd15_diffusers_runwayml_path: str,
    use_local_weights: bool,
) -> StableDiffusionPipeline:
    return StableDiffusionPipeline.from_pretrained(  # type: ignore
        sd15_diffusers_runwayml_path,
        local_files_only=use_local_weights,
    )


@pytest.fixture(scope="module")
def diffusers_sdxl_pipeline(
    sdxl_diffusers_stabilityai_path: str,
    use_local_weights: bool,
) -> StableDiffusionXLPipeline:
    return StableDiffusionXLPipeline.from_pretrained(  # type: ignore
        sdxl_diffusers_stabilityai_path,
        local_files_only=use_local_weights,
    )


@pytest.fixture(scope="module")
def diffusers_sdxl_unet(diffusers_sdxl_pipeline: StableDiffusionXLPipeline) -> UNet2DConditionModel:
    return diffusers_sdxl_pipeline.unet  # type: ignore
