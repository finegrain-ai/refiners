import torch

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.color_palette import ColorPaletteEncoder, SD1ColorPaletteAdapter
from refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock2d
from refiners.foundationals.latent_diffusion.stable_diffusion_1 import SD1UNet

print("yoyoyoyo")


def test_color_palette_encoder() -> None:
    in_channels = 22
    max_colors = 10
    unet = SD1UNet(in_channels)
    cross_attn_2d = unet.ensure_find(CrossAttentionBlock2d)

    color_palette_encoder = ColorPaletteEncoder(
        model_dim=in_channels, max_colors=max_colors, embedding_dim=cross_attn_2d.context_embedding_dim
    ).to(device=0)

    batch_size = 5
    color_size = 4

    palettes = torch.zeros(batch_size, color_size, 3)
    print("palettes.get_device()", palettes.get_device())

    encoded = color_palette_encoder(palettes)

    assert isinstance(encoded.shape, torch.Size)
    assert encoded.shape == torch.Size([batch_size, max_colors, cross_attn_2d.context_embedding_dim])
