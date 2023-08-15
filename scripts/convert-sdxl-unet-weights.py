import torch

from safetensors.torch import save_file  # type: ignore
from refiners.fluxion.utils import create_state_dict_mapping, convert_state_dict

from diffusers import DiffusionPipeline  # type: ignore
from diffusers.models.unet_2d_condition import UNet2DConditionModel  # type: ignore

from refiners.foundationals.latent_diffusion.sdxl_unet import SDXLUNet


@torch.no_grad()
def convert(src_model: UNet2DConditionModel) -> dict[str, torch.Tensor]:
    dst_model = SDXLUNet(in_channels=4)

    x = torch.randn(1, 4, 32, 32)
    timestep = torch.tensor(data=[0])
    clip_text_embeddings = torch.randn(1, 77, 2048)

    added_cond_kwargs = {"text_embeds": torch.randn(1, 1280), "time_ids": torch.randn(1, 6)}
    src_args = (x, timestep, clip_text_embeddings, None, None, None, None, added_cond_kwargs)
    dst_model.set_timestep(timestep=timestep)
    dst_model.set_clip_text_embedding(clip_text_embedding=clip_text_embeddings)
    dst_model.set_time_ids(time_ids=added_cond_kwargs["time_ids"])
    dst_model.set_pooled_text_embedding(pooled_text_embedding=added_cond_kwargs["text_embeds"])
    dst_args = (x,)

    mapping = create_state_dict_mapping(
        source_model=src_model, target_model=dst_model, source_args=src_args, target_args=dst_args  # type: ignore
    )
    if mapping is None:
        raise RuntimeError("Could not create state dict mapping")
    state_dict = convert_state_dict(
        source_state_dict=src_model.state_dict(), target_state_dict=dst_model.state_dict(), state_dict_mapping=mapping
    )
    return {k: v for k, v in state_dict.items()}


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
        default="stable_diffusion_xl_unet.safetensors",
        help="Path for the output file",
    )
    args = parser.parse_args()
    src_model = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path=args.source).unet  # type: ignore
    tensors = convert(src_model=src_model)  # type: ignore
    save_file(tensors=tensors, filename=args.output_file)


if __name__ == "__main__":
    main()
