import torch

from refiners.fluxion.utils import create_state_dict_mapping, convert_state_dict, save_to_safetensors

from diffusers import DiffusionPipeline  # type: ignore
from transformers.models.clip.modeling_clip import CLIPTextModel  # type: ignore

from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL


@torch.no_grad()
def convert(src_model: CLIPTextModel) -> dict[str, torch.Tensor]:
    dst_model = CLIPTextEncoderL()
    x = dst_model.tokenizer("Nice cat", sequence_length=77)
    mapping = create_state_dict_mapping(source_model=src_model, target_model=dst_model, source_args=[x])  # type: ignore
    assert mapping is not None, "Model conversion failed"
    state_dict = convert_state_dict(
        source_state_dict=src_model.state_dict(), target_state_dict=dst_model.state_dict(), state_dict_mapping=mapping
    )
    return {k: v.half() for k, v in state_dict.items()}


def main() -> None:
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
    src_model = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path=args.source).text_encoder  # type: ignore
    tensors = convert(src_model=src_model)  # type: ignore
    save_to_safetensors(path=args.output_file, tensors=tensors)


if __name__ == "__main__":
    main()
