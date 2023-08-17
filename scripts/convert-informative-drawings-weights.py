# Original weights can be found here: https://huggingface.co/spaces/carolineec/informativedrawings
# Code is at https://github.com/carolineec/informative-drawings
# Copy `model.py` in your `PYTHONPATH`. You can edit it to remove un-necessary code
# and imports if you want, we only need `Generator`.

import torch

from refiners.fluxion.utils import create_state_dict_mapping, convert_state_dict, save_to_safetensors

from refiners.foundationals.latent_diffusion.preprocessors.informative_drawings import InformativeDrawings
from model import Generator  # type: ignore


@torch.no_grad()
def convert(checkpoint: str, device: torch.device | str) -> dict[str, torch.Tensor]:
    src_model = Generator(3, 1, 3)  # type: ignore
    src_model.load_state_dict(torch.load(checkpoint, map_location=device))  # type: ignore
    src_model.eval()  # type: ignore

    dst_model = InformativeDrawings()

    x = torch.randn(1, 3, 512, 512)

    mapping = create_state_dict_mapping(source_model=src_model, target_model=dst_model, source_args=[x])  # type: ignore
    assert mapping is not None, "Model conversion failed"
    state_dict = convert_state_dict(source_state_dict=src_model.state_dict(), target_state_dict=dst_model.state_dict(), state_dict_mapping=mapping)  # type: ignore
    return {k: v.half() for k, v in state_dict.items()}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from",
        type=str,
        dest="source",
        required=False,
        default="model2.pth",
        help="Source model",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        default="informative-drawings.safetensors",
        help="Path for the output file",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tensors = convert(checkpoint=args.source, device=device)
    save_to_safetensors(path=args.output_file, tensors=tensors)


if __name__ == "__main__":
    main()
