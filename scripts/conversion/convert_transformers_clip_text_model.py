import argparse
from pathlib import Path
from torch import nn
from refiners.fluxion.model_converter import ModelConverter
from transformers import CLIPTextModelWithProjection  # type: ignore
from refiners.foundationals.clip.text_encoder import CLIPTextEncoder
from refiners.foundationals.clip.tokenizer import CLIPTokenizer
import refiners.fluxion.layers as fl


class Args(argparse.Namespace):
    source_path: str
    subfolder: str
    output_path: str | None
    use_half: bool
    verbose: bool


def setup_converter(args: Args) -> ModelConverter:
    source: nn.Module = CLIPTextModelWithProjection.from_pretrained(  # type: ignore
        pretrained_model_name_or_path=args.source_path, subfolder=args.subfolder
    )
    assert isinstance(source, nn.Module), "Source model is not a nn.Module"
    architecture: str = source.config.architectures[0]  # type: ignore
    embedding_dim: int = source.config.hidden_size  # type: ignore
    projection_dim: int = source.config.projection_dim  # type: ignore
    num_layers: int = source.config.num_hidden_layers  # type: ignore
    num_attention_heads: int = source.config.num_attention_heads  # type: ignore
    feed_forward_dim: int = source.config.intermediate_size  # type: ignore
    use_quick_gelu: bool = source.config.hidden_act == "quick_gelu"  # type: ignore
    target = CLIPTextEncoder(
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        feedforward_dim=feed_forward_dim,
        use_quick_gelu=use_quick_gelu,
    )
    match architecture:
        case "CLIPTextModel":
            source.text_projection = fl.Identity()
        case "CLIPTextModelWithProjection":
            target.append(module=fl.Linear(in_features=embedding_dim, out_features=projection_dim, bias=False))
        case _:
            raise RuntimeError(f"Unsupported architecture: {architecture}")
    text = "What a nice cat you have there!"
    tokenizer = target.find(layer_type=CLIPTokenizer)
    assert tokenizer is not None, "Could not find tokenizer"
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
        "--to",
        type=str,
        dest="output_path",
        default=None,
        help=(
            "Output path (.safetensors) for converted model. If not provided, the output path will be the same as the"
            " source path."
        ),
    )
    parser.add_argument("--half", action="store_true", default=True, help="Convert to half precision. Default: True")
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
    converter.save_to_safetensors(path=args.output_path, half=args.half)


if __name__ == "__main__":
    main()
