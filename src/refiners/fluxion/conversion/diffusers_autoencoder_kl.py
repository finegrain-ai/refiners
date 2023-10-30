import torch
from torch import nn

from refiners.fluxion.model_converter import ModelConverter
from refiners.foundationals.latent_diffusion.auto_encoder import LatentDiffusionAutoencoder


def setup_autoencoder_kl_converter(source_path: str, subfolder="vae", verbose: bool = False) -> ModelConverter:
    from diffusers import AutoencoderKL  # type: ignore

    target = LatentDiffusionAutoencoder()
    source: nn.Module = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path=source_path, subfolder=subfolder
    )  # type: ignore
    x = torch.randn(1, 3, 512, 512)
    converter = ModelConverter(source_model=source, target_model=target, skip_output_check=True, verbose=verbose)
    if not converter.run(source_args=(x,)):
        raise RuntimeError("Model conversion failed")
    return converter
