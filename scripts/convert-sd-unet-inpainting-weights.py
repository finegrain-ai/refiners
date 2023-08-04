import torch

from safetensors.torch import save_file
from refiners.fluxion.utils import (
    create_state_dict_mapping,
    convert_state_dict,
)

from diffusers import StableDiffusionInpaintPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionModel

from refiners.foundationals.latent_diffusion.unet import UNet


@torch.no_grad()
def convert(src_model: UNet2DConditionModel) -> dict[str, torch.Tensor]:
    dst_model = UNet(in_channels=9, clip_embedding_dim=768)

    x = torch.randn(1, 9, 32, 32)
    timestep = torch.tensor([0])
    clip_text_embeddings = torch.randn(1, 77, 768)

    src_args = (x, timestep, clip_text_embeddings)
    dst_model.set_timestep(timestep)
    dst_model.set_clip_text_embedding(clip_text_embeddings)
    dst_args = (x,)

    mapping = create_state_dict_mapping(src_model, dst_model, src_args, dst_args)
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
        default="runwayml/stable-diffusion-inpainting",
        help="Source model",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        default="stable_diffusion_1_5_inpainting_unet.safetensors",
        help="Path for the output file",
    )
    args = parser.parse_args()
    src_model = StableDiffusionInpaintPipeline.from_pretrained(args.source).unet
    tensors = convert(src_model)
    save_file(tensors, args.output_file)


if __name__ == "__main__":
    main()
