# Original weights can be found here: https://huggingface.co/spaces/carolineec/informativedrawings
# Code is at https://github.com/carolineec/informative-drawings
# Copy `model.py` in your `PYTHONPATH`. You can edit it to remove un-necessary code
# and imports if you want, we only need `Generator`.

import torch

from safetensors.torch import save_file
from refiners.fluxion.utils import (
    create_state_dict_mapping,
    convert_state_dict,
)

from refiners.foundationals.latent_diffusion.preprocessors.informative_drawings import InformativeDrawings
from model import Generator


@torch.no_grad()
def convert(checkpoint: str, device: torch.device) -> dict[str, torch.Tensor]:
    src_model = Generator(3, 1, 3)
    src_model.load_state_dict(torch.load(checkpoint, map_location=device))
    src_model.eval()

    dst_model = InformativeDrawings()

    x = torch.randn(1, 3, 512, 512)

    mapping = create_state_dict_mapping(src_model, dst_model, [x])
    state_dict = convert_state_dict(src_model.state_dict(), dst_model.state_dict(), mapping)
    return {k: v.half() for k, v in state_dict.items()}


def main():
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

    tensors = convert(args.source, device)
    save_file(tensors, args.output_file)


if __name__ == "__main__":
    main()
