import argparse
from pathlib import Path
from typing import cast

from torch import nn
from transformers import CLIPTextModel, CLIPTextModelWithProjection  # type: ignore

import refiners.fluxion.layers as fl
from refiners.fluxion.model_converter import ModelConverter
from refiners.fluxion.utils import save_to_safetensors
from refiners.foundationals.clip.text_encoder import CLIPTextEncoder, CLIPTextEncoderG, CLIPTextEncoderL
from refiners.foundationals.clip.tokenizer import CLIPTokenizer
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.text_encoder import DoubleTextEncoder


class Args(argparse.Namespace):
    source_path: str
    subfolder: str
    output_path: str | None
    half: bool
    verbose: bool


def setup_converter(args: Args, with_projection: bool = False) -> ModelConverter:
    # low_cpu_mem_usage=False stops some annoying console messages us to `pip install accelerate`
    cls = CLIPTextModelWithProjection if with_projection else CLIPTextModel
    source: nn.Module = cls.from_pretrained(  # type: ignore
        pretrained_model_name_or_path=args.source_path,
        subfolder=args.subfolder,
        low_cpu_mem_usage=False,
    )
    assert isinstance(source, nn.Module), "Source model is not a nn.Module"
    architecture: str = source.config.architectures[0]  # type: ignore
    embedding_dim: int = source.config.hidden_size  # type: ignore
    projection_dim: int = source.config.projection_dim  # type: ignore
    num_layers: int = source.config.num_hidden_layers  # type: ignore
    num_attention_heads: int = source.config.num_attention_heads  # type: ignore
    feed_forward_dim: int = source.config.intermediate_size  # type: ignore
    use_quick_gelu: bool = source.config.hidden_act == "quick_gelu"  # type: ignore
    assert architecture in ("CLIPTextModel", "CLIPTextModelWithProjection"), f"Unsupported architecture: {architecture}"
    target = CLIPTextEncoder(
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        feedforward_dim=feed_forward_dim,
        use_quick_gelu=use_quick_gelu,
    )
    if architecture == "CLIPTextModelWithProjection":
        target.append(module=fl.Linear(in_features=embedding_dim, out_features=projection_dim, bias=False))
    text = "What a nice cat you have there!"
    tokenizer = target.ensure_find(CLIPTokenizer)
    tokens = tokenizer(text)
    converter = ModelConverter(source_model=source, target_model=target, skip_output_check=True, verbose=args.verbose)
    if not converter.run(source_args=(tokens,), target_args=(text,)):
        raise RuntimeError("Model conversion failed")
    return converter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converts a CLIPTextEncoder from the library transformers from the HuggingFace Hub to refiners."
    )
    parser.add_argument(
        "--from",
        type=str,
        dest="source_path",
        default="runwayml/stable-diffusion-v1-5",
        help=(
            "Can be a path to a .bin file, a .safetensors file or a model name from the HuggingFace Hub. Default:"
            " runwayml/stable-diffusion-v1-5"
        ),
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        dest="subfolder",
        default="text_encoder",
        help=(
            "Subfolder in the source path where the model is located inside the Hub. Default: text_encoder (for"
            " CLIPTextModel)"
        ),
    )
    parser.add_argument(
        "--subfolder2",
        type=str,
        dest="subfolder2",
        default=None,
        help="Additional subfolder for the 2nd text encoder (useful for SDXL). Default: None",
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
    args = parser.parse_args(namespace=Args())
    if args.output_path is None:
        args.output_path = f"{Path(args.source_path).stem}-{args.subfolder}.safetensors"
    converter = setup_converter(args=args)
    if args.subfolder2 is not None:
        # Assume this is the second text encoder of Stable Diffusion XL
        args.subfolder = args.subfolder2
        converter2 = setup_converter(args=args, with_projection=True)

        text_encoder_l = CLIPTextEncoderL()
        text_encoder_l.load_state_dict(state_dict=converter.get_state_dict())

        projection = cast(CLIPTextEncoder, converter2.target_model)[-1]
        assert isinstance(projection, fl.Linear)
        text_encoder_g_with_projection = CLIPTextEncoderG()
        text_encoder_g_with_projection.append(module=projection)
        text_encoder_g_with_projection.load_state_dict(state_dict=converter2.get_state_dict())

        projection = text_encoder_g_with_projection.pop(index=-1)
        assert isinstance(projection, fl.Linear)
        double_text_encoder = DoubleTextEncoder(
            text_encoder_l=text_encoder_l, text_encoder_g=text_encoder_g_with_projection, projection=projection
        )

        state_dict = double_text_encoder.state_dict()
        if args.half:
            state_dict = {key: value.half() for key, value in state_dict.items()}
        save_to_safetensors(path=args.output_path, tensors=state_dict)
    else:
        converter.save_to_safetensors(path=args.output_path, half=args.half)


if __name__ == "__main__":
    main()
