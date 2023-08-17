import torch

from safetensors.torch import save_file  # type: ignore
from refiners.fluxion.utils import create_state_dict_mapping, convert_state_dict

from diffusers import DiffusionPipeline  # type: ignore
from transformers.models.clip.modeling_clip import CLIPTextModel  # type: ignore

from refiners.foundationals.clip.tokenizer import CLIPTokenizer
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderG
import refiners.fluxion.layers as fl


@torch.no_grad()
def convert(src_model: CLIPTextModel) -> dict[str, torch.Tensor]:
    dst_model = CLIPTextEncoderG()
    # Extra projection layer (see CLIPTextModelWithProjection in transformers)
    dst_model.append(module=fl.Linear(in_features=1280, out_features=1280, bias=False))
    tokenizer = dst_model.find(layer_type=CLIPTokenizer)
    assert tokenizer is not None, "Could not find tokenizer"
    tokens = tokenizer("Nice cat")
    mapping = create_state_dict_mapping(source_model=src_model, target_model=dst_model, source_args=[tokens], target_args=["Nice cat"])  # type: ignore
    if mapping is None:
        raise RuntimeError("Could not create state dict mapping")
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
        default="stabilityai/stable-diffusion-xl-base-0.9",
        help="Source model",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        default="CLIPTextEncoderG.safetensors",
        help="Path for the output file",
    )
    args = parser.parse_args()
    src_model = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path=args.source).text_encoder_2  # type: ignore
    tensors = convert(src_model=src_model)  # type: ignore
    save_file(tensors=tensors, filename=args.output_file)


if __name__ == "__main__":
    main()
