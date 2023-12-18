import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKL  # type: ignore
from torch import nn

from refiners.fluxion.model_converter import ModelConverter
from refiners.foundationals.latent_diffusion.auto_encoder import LatentDiffusionAutoencoder


class Args(argparse.Namespace):
    source_path: str
    output_path: str | None
    use_half: bool
    verbose: bool


def setup_converter(args: Args) -> ModelConverter:
    target = LatentDiffusionAutoencoder()
    # low_cpu_mem_usage=False stops some annoying console messages us to `pip install accelerate`
    source: nn.Module = AutoencoderKL.from_pretrained(  # type: ignore
        pretrained_model_name_or_path=args.source_path,
        subfolder=args.subfolder,
        low_cpu_mem_usage=False,
    )  # type: ignore
    x = torch.randn(1, 3, 512, 512)
    converter = ModelConverter(source_model=source, target_model=target, skip_output_check=True, verbose=args.verbose)
    if not converter.run(source_args=(x,)):
        raise RuntimeError("Model conversion failed")
    return converter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a pretrained diffusers AutoencoderKL model to a refiners Latent Diffusion Autoencoder"
    )
    parser.add_argument(
        "--from",
        type=str,
        dest="source_path",
        default="runwayml/stable-diffusion-v1-5",
        help="Path to the source pretrained model (default: 'runwayml/stable-diffusion-v1-5').",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        dest="subfolder",
        default="vae",
        help="Subfolder in the source path where the model is located inside the Hub (default: 'vae')",
    )
    parser.add_argument(
        "--to",
        type=str,
        dest="output_path",
        default=None,
        help=(
            "Path to save the converted model (extension will be .safetensors). If not specified, the output path will"
            " be the source path with the extension changed to .safetensors."
        ),
    )
    parser.add_argument(
        "--half",
        action="store_true",
        dest="use_half",
        default=False,
        help="Use this flag to save the output file as half precision (default: full precision).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        dest="verbose",
        default=False,
        help="Use this flag to print verbose output during conversion.",
    )
    args = parser.parse_args(namespace=Args())
    if args.output_path is None:
        args.output_path = f"{Path(args.source_path).stem}-autoencoder.safetensors"
    assert args.output_path is not None
    converter = setup_converter(args=args)
    converter.save_to_safetensors(path=args.output_path, half=args.use_half)
