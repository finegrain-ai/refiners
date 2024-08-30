import argparse
from pathlib import Path

from convert_diffusers_unet import Args as UNetArgs, setup_converter as setup_unet_converter
from huggingface_hub import hf_hub_download  # type: ignore

from refiners.fluxion.utils import load_from_safetensors, save_to_safetensors


class Args(argparse.Namespace):
    source_path: str
    output_path: str | None
    subfolder: str
    half: bool
    verbose: bool
    reference_unet_path: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Converts IC-Light patch weights to work with Refiners")
    parser.add_argument(
        "--from",
        type=str,
        dest="source_path",
        default="lllyasviel/ic-light",
        help=(
            "Can be a path to a .bin file, a .safetensors file or a model name from the Hugging Face Hub. Default:"
            " lllyasviel/ic-light"
        ),
    )
    parser.add_argument("--filename", type=str, default="iclight_sd15_fc.safetensors", help="Filename inside the hub.")
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Prints additional information during conversion. Default: False",
    )
    parser.add_argument(
        "--reference-unet-path",
        type=str,
        dest="reference_unet_path",
        default="sd-legacy/stable-diffusion-v1-5",
        help="Path to the reference UNet weights.",
    )
    args = parser.parse_args(namespace=Args())
    if args.output_path is None:
        args.output_path = f"{Path(args.filename).stem}-refiners.safetensors"

    patch_file = (
        Path(args.source_path)
        if args.source_path.endswith(".safetensors")
        else Path(
            hf_hub_download(
                repo_id=args.source_path,
                filename=args.filename,
            )
        )
    )
    patch_weights = load_from_safetensors(patch_file)

    unet_args = UNetArgs(
        source_path=args.reference_unet_path,
        subfolder="unet",
        half=False,
        verbose=False,
        skip_init_check=True,
        override_weights=None,
    )
    converter = setup_unet_converter(args=unet_args)
    result = converter._convert_state_dict(  # pyright: ignore[reportPrivateUsage]
        source_state_dict=patch_weights,
        target_state_dict=converter.target_model.state_dict(),
        state_dict_mapping=converter.get_mapping(),
    )
    save_to_safetensors(path=args.output_path, tensors=result)


if __name__ == "__main__":
    main()
