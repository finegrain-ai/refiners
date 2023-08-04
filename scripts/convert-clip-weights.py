import torch

from safetensors.torch import save_file
from refiners.fluxion.utils import (
    create_state_dict_mapping,
    convert_state_dict,
)

from diffusers import DiffusionPipeline
from transformers.models.clip.modeling_clip import CLIPTextModel

from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL


@torch.no_grad()
def convert(src_model: CLIPTextModel) -> dict[str, torch.Tensor]:
    dst_model = CLIPTextEncoderL()
    x = dst_model.tokenizer("Nice cat", sequence_length=77)
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
        default="runwayml/stable-diffusion-v1-5",
        help="Source model",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        default="CLIPTextEncoderL.safetensors",
        help="Path for the output file",
    )
    args = parser.parse_args()
    src_model = DiffusionPipeline.from_pretrained(args.source).text_encoder
    tensors = convert(src_model)
    save_file(tensors, args.output_file)


if __name__ == "__main__":
    main()
