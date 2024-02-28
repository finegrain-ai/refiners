import argparse
from pathlib import Path

import torch
from diffusers.models.autoencoders.consistency_decoder_vae import ConsistencyDecoderVAE
from torch import nn

from refiners.fluxion.model_converter import ModelConverter
from refiners.fluxion.utils import save_to_safetensors
from refiners.foundationals.latent_diffusion.consistency_decoder import ConsistencyDecoder


class Args(argparse.Namespace):
    source_path: str
    subfolder: str
    output_path: str | None
    half: bool
    verbose: bool
    threshold: float


def setup_converter(args: Args) -> ModelConverter:
    # low_cpu_mem_usage=False stops some annoying console messages us to `pip install accelerate`
    source: nn.Module = ConsistencyDecoderVAE.from_pretrained(  # type: ignore
        pretrained_model_name_or_path=args.source_path,
        subfolder=args.subfolder,
        low_cpu_mem_usage=False,
    ).decoder_unet

    assert isinstance(source, nn.Module), "Source model is not a nn.Module"
    image_size: int = 32
    num_latent_channels: int = 7

    target = ConsistencyDecoder()

    x = torch.randn(1, num_latent_channels, image_size, image_size)

    source_to_ignore = [f"down_blocks.{i}.downsamplers.0.downsample.conv" for i in [0, 1, 2]]
    target_to_ignore = [f"DownBlocks.DownsamplingBlock_{i}.Chain.AvgPool2d" for i in [1, 2, 3]]

    converter = ModelConverter(
        source_model=source,
        target_model=target,
        verbose=True,
        source_keys_to_skip=source_to_ignore,
        target_keys_to_skip=target_to_ignore,
        skip_output_check=True,
    )

    timestep = torch.tensor(data=[0])
    target.set_timestep(timestep=timestep)
    converter.run(source_args=(x, timestep), target_args=(x,))
    return converter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converts a ConsistencyDecoderVAE from the library transformers from the HuggingFace Hub to refiners."
    )
    parser.add_argument(
        "--from",
        type=str,
        dest="source_path",
        default="openai/consistency-decoder",
        help=(
            "Can be a path to a .bin file, a .safetensors file or a model name from the HuggingFace Hub. Default:"
            "openai/consistency-decoder"
        ),
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        dest="subfolder",
        default=".",
        help="Subfolder in the source path where the model is located inside the Hub. Default: image_encoder",
    )
    parser.add_argument(
        "--to",
        type=str,
        dest="output_path",
        default=None,
        help=(
            "Output path (.safetensors) for converted model. If not provided, the output path will be the same as the"
            " source path."
        ),
    )
    parser.add_argument("--half", action="store_true", help="Convert to half precision.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Prints additional information during conversion. Default: False",
    )
    parser.add_argument("--threshold", type=float, default=1e-2, help="Threshold for model comparison. Default: 1e-2")
    args = parser.parse_args(namespace=Args())
    if args.output_path is None:
        args.output_path = f"{Path(args.source_path).stem}-{args.subfolder}.safetensors"
    converter = setup_converter(args=args)
    # Do not use converter.save_to_safetensors since it is not in a valid state due to the ad hoc conversion
    state_dict = converter.target_model.state_dict()
    if args.half:
        state_dict = {key: value.half() for key, value in state_dict.items()}
    save_to_safetensors(path=args.output_path, tensors=state_dict)


if __name__ == "__main__":
    main()
