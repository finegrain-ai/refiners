import argparse
from pathlib import Path
from typing import NamedTuple, cast

import torch
from torch import nn
from transformers import CLIPVisionModelWithProjection  # type: ignore

import refiners.fluxion.layers as fl
from refiners.fluxion.model_converter import ModelConverter
from refiners.fluxion.utils import save_to_safetensors
from refiners.foundationals.clip.image_encoder import CLIPImageEncoder


class Args(argparse.Namespace):
    source_path: str
    subfolder: str
    output_path: str | None
    half: bool
    verbose: bool
    threshold: float


class CLIPImageEncoderConfig(NamedTuple):
    architectures: list[str]
    num_channels: int
    hidden_size: int
    hidden_act: str
    image_size: int
    projection_dim: int
    patch_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    layer_norm_eps: float


def setup_converter(args: Args) -> ModelConverter:
    # low_cpu_mem_usage=False stops some annoying console messages us to `pip install accelerate`
    source: nn.Module = CLIPVisionModelWithProjection.from_pretrained(  # type: ignore
        pretrained_model_name_or_path=args.source_path,
        subfolder=args.subfolder,
        low_cpu_mem_usage=False,
    )
    assert isinstance(source, nn.Module), "Source model is not a nn.Module"
    config = cast(CLIPImageEncoderConfig, source.config)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]

    assert (
        config.architectures[0] == "CLIPVisionModelWithProjection"
    ), f"Unsupported architecture: {config.architectures[0]}"
    assert config.num_channels == 3, f"Expected 3 input channels, got {config.num_channels}"
    assert config.hidden_act == "gelu", f"Unsupported activation: {config.hidden_act}"

    target = CLIPImageEncoder(
        image_size=config.image_size,
        embedding_dim=config.hidden_size,
        output_dim=config.projection_dim,
        patch_size=config.patch_size,
        num_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        feedforward_dim=config.intermediate_size,
        layer_norm_eps=config.layer_norm_eps,
    )

    x = torch.randn(1, 3, config.image_size, config.image_size)

    converter = ModelConverter(source_model=source, target_model=target, verbose=True)

    # Custom conversion logic since the class embedding (fl.Parameter layer) is not supported out-of-the-box by the
    # converter
    mapping = converter.map_state_dicts((x,))
    assert mapping is not None

    source_state_dict = source.state_dict()
    target_state_dict = target.state_dict()

    # Remove the class embedding from state dict since it was not mapped by the model converter
    class_embedding = target.ensure_find(fl.Parameter)
    class_embedding_key = next((n for n, p in target.named_parameters() if id(p) == id(class_embedding.weight)), None)
    assert class_embedding_key is not None
    assert class_embedding_key in target_state_dict
    del target_state_dict[class_embedding_key]

    converted_state_dict = converter._convert_state_dict(  # type: ignore[reportPrivateUsage]
        source_state_dict=source_state_dict, target_state_dict=target_state_dict, state_dict_mapping=mapping
    )
    target.load_state_dict(state_dict=converted_state_dict, strict=False)

    # Ad hoc post-conversion steps
    embed = source.vision_model.embeddings.class_embedding
    class_embedding.weight = torch.nn.Parameter(embed.clone().reshape_as(class_embedding.weight))  # type: ignore

    assert converter.compare_models((x,), threshold=args.threshold)

    return converter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converts a CLIPImageEncoder from the library transformers from the HuggingFace Hub to refiners."
    )
    parser.add_argument(
        "--from",
        type=str,
        dest="source_path",
        default="stabilityai/stable-diffusion-2-1-unclip",
        help=(
            "Can be a path to a .bin file, a .safetensors file or a model name from the HuggingFace Hub. Default:"
            " stabilityai/stable-diffusion-2-1-unclip"
        ),
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        dest="subfolder",
        default="image_encoder",
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
