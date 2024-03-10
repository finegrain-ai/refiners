import argparse
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from refiners.fluxion.model_converter import ModelConverter
from refiners.fluxion.utils import load_tensors
from refiners.foundationals.latent_diffusion.preprocessors.informative_drawings import InformativeDrawings

try:
    from model import Generator  # type: ignore
except ImportError:
    raise ImportError(
        "Please download the model.py file from https://github.com/carolineec/informative-drawings and add it to your"
        " PYTHONPATH"
    )
if TYPE_CHECKING:
    Generator = cast(nn.Module, Generator)


class Args(argparse.Namespace):
    source_path: str
    output_path: str
    verbose: bool
    half: bool


def setup_converter(args: Args) -> ModelConverter:
    source = Generator(3, 1, 3)
    source.load_state_dict(state_dict=load_tensors(args.source_path))
    source.eval()
    target = InformativeDrawings()
    x = torch.randn(1, 3, 512, 512)
    converter = ModelConverter(source_model=source, target_model=target, skip_output_check=True, verbose=args.verbose)
    if not converter.run(source_args=(x,)):
        raise RuntimeError("Model conversion failed")
    return converter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converts a pretrained Informative Drawings model to a refiners Informative Drawings model"
    )
    parser.add_argument(
        "--from",
        type=str,
        dest="source_path",
        default="model2.pth",
        help="Path to the source model. (default: 'model2.pth').",
    )
    parser.add_argument(
        "--to",
        type=str,
        dest="output_path",
        default="informative-drawings.safetensors",
        help="Path to save the converted model. (default: 'informative-drawings.safetensors').",
    )
    parser.add_argument("--verbose", action="store_true", dest="verbose")
    parser.add_argument("--half", action="store_true", dest="half")
    args = parser.parse_args(namespace=Args())
    converter = setup_converter(args=args)
    converter.save_to_safetensors(path=args.output_path, half=args.half)


if __name__ == "__main__":
    main()
