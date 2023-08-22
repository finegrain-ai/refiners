import torch

from refiners.fluxion.utils import create_state_dict_mapping, convert_state_dict, save_to_safetensors

from diffusers import StableDiffusionInpaintPipeline  # type: ignore
from diffusers.models.unet_2d_condition import UNet2DConditionModel  # type: ignore

from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet


@torch.no_grad()
def convert(src_model: UNet2DConditionModel) -> dict[str, torch.Tensor]:
    dst_model = SD1UNet(in_channels=9, clip_embedding_dim=768)

    x = torch.randn(1, 9, 32, 32)
    timestep = torch.tensor(data=[0])
    clip_text_embeddings = torch.randn(1, 77, 768)

    src_args = (x, timestep, clip_text_embeddings)
    dst_model.set_timestep(timestep=timestep)
    dst_model.set_clip_text_embedding(clip_text_embedding=clip_text_embeddings)
    dst_args = (x,)

    mapping = create_state_dict_mapping(source_model=src_model, target_model=dst_model, source_args=src_args, target_args=dst_args)  # type: ignore
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
    src_model = StableDiffusionInpaintPipeline.from_pretrained(pretrained_model_name_or_path=args.source).unet  # type: ignore
    tensors = convert(src_model=src_model)  # type: ignore
    save_to_safetensors(path=args.output_file, tensors=tensors)


if __name__ == "__main__":
    main()
