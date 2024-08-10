import argparse
from pathlib import Path

from refiners.fluxion.utils import load_tensors, save_to_safetensors
from refiners.foundationals.swin.mvanet.converter import convert_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from",
        type=str,
        required=True,
        dest="source_path",
        help="A MVANet checkpoint. One can be found at https://github.com/qianyu-dlut/MVANet",
    )
    parser.add_argument(
        "--to",
        type=str,
        dest="output_path",
        default=None,
        help=(
            "Path to save the converted model. If not specified, the output path will be the source path with the"
            " extension changed to .safetensors."
        ),
    )
    parser.add_argument("--half", action="store_true", dest="half")
    args = parser.parse_args()

    src_weights = load_tensors(args.source_path)
    weights = convert_weights(src_weights)
    if args.half:
        weights = {key: value.half() for key, value in weights.items()}
    if args.output_path is None:
        args.output_path = f"{Path(args.source_path).stem}.safetensors"
    save_to_safetensors(path=args.output_path, tensors=weights)


if __name__ == "__main__":
    main()
