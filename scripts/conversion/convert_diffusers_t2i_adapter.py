import argparse
from pathlib import Path
import torch
from torch import nn
from diffusers import T2IAdapter  # type: ignore
from refiners.foundationals.latent_diffusion.t2i_adapter import ConditionEncoder, ConditionEncoderXL
from refiners.fluxion.model_converter import ModelConverter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a pretrained diffusers T2I-Adapter model to refiners")
    parser.add_argument(
        "--from",
        type=str,
        dest="source_path",
        required=True,
        help="Path or repository name of the source model. (e.g.: 'ip-adapter_sd15.bin').",
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
    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = f"{Path(args.source_path).name}.safetensors"
    assert args.output_path is not None

    sdxl = "xl" in args.source_path
    target = ConditionEncoderXL() if sdxl else ConditionEncoder()
    source: nn.Module = T2IAdapter.from_pretrained(pretrained_model_name_or_path=args.source_path)  # type: ignore
    assert isinstance(source, nn.Module), "Source model is not a nn.Module"

    x = torch.randn(1, 3, 1024, 1024) if sdxl else torch.randn(1, 3, 512, 512)
    converter = ModelConverter(source_model=source, target_model=target, verbose=args.verbose)
    if not converter.run(source_args=(x,)):
        raise RuntimeError("Model conversion failed")

    converter.save_to_safetensors(path=args.output_path, half=args.use_half)
