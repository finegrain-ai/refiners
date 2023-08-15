import torch

from refiners.fluxion.utils import (
    create_state_dict_mapping,
    convert_state_dict,
    save_to_safetensors,
)

from diffusers import DiffusionPipeline  # type: ignore
from diffusers.models.autoencoder_kl import AutoencoderKL  # type: ignore

from refiners.foundationals.latent_diffusion.auto_encoder import LatentDiffusionAutoencoder


@torch.no_grad()
def convert(src_model: AutoencoderKL) -> dict[str, torch.Tensor]:
    dst_model = LatentDiffusionAutoencoder()
    x = torch.randn(1, 3, 512, 512)
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
        default="lda.safetensors",
        help="Path for the output file",
    )
    args = parser.parse_args()
    src_model = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path=args.source).vae  # type: ignore
    tensors = convert(src_model=src_model)  # type: ignore
    save_to_safetensors(path=args.output_file, tensors=tensors)


if __name__ == "__main__":
    main()
