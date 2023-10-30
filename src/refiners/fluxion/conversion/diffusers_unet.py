import torch
from torch import nn

from refiners.fluxion.model_converter import ModelConverter
from refiners.foundationals.latent_diffusion import SD1UNet, SDXLUNet


def setup_diffusers_unet_converter(source_path, subfolder="unet", verbose=False) -> ModelConverter:
    from diffusers import UNet2DConditionModel  # type: ignore

    source: nn.Module = UNet2DConditionModel.from_pretrained(  # type: ignore
        pretrained_model_name_or_path=source_path, subfolder=subfolder
    )
    source_in_channels: int = source.config.in_channels  # type: ignore
    source_clip_embedding_dim: int = source.config.cross_attention_dim  # type: ignore
    source_has_time_ids: bool = source.config.addition_embed_type == "text_time"  # type: ignore
    target = (
        SDXLUNet(in_channels=source_in_channels) if source_has_time_ids else SD1UNet(in_channels=source_in_channels)
    )

    x = torch.randn(1, source_in_channels, 32, 32)
    timestep = torch.tensor(data=[0])
    clip_text_embeddings = torch.randn(1, 77, source_clip_embedding_dim)

    target.set_timestep(timestep=timestep)
    target.set_clip_text_embedding(clip_text_embedding=clip_text_embeddings)
    added_cond_kwargs = {}
    if source_has_time_ids:
        added_cond_kwargs = {"text_embeds": torch.randn(1, 1280), "time_ids": torch.randn(1, 6)}
        target.set_time_ids(time_ids=added_cond_kwargs["time_ids"])
        target.set_pooled_text_embedding(pooled_text_embedding=added_cond_kwargs["text_embeds"])

    target_args = (x,)
    source_args = {
        "positional": (x, timestep, clip_text_embeddings),
        "keyword": {"added_cond_kwargs": added_cond_kwargs} if source_has_time_ids else {},
    }

    converter = ModelConverter(source_model=source, target_model=target, skip_output_check=True, verbose=verbose)
    if not converter.run(
        source_args=source_args,
        target_args=target_args,
    ):
        raise RuntimeError("Model conversion failed")
    return converter
