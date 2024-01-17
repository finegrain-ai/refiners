import torch

from refiners.fluxion.adapters.color_palette import ColorPaletteEncoder
from refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock2d
from refiners.foundationals.latent_diffusion.stable_diffusion_1 import SD1UNet


def test_color_palette_encoder() -> None:
    in_channels = 22
    max_colors = 10
    unet = SD1UNet(in_channels)
    cross_attn_2d = unet.ensure_find(CrossAttentionBlock2d)

    color_palette_encoder = ColorPaletteEncoder(
        model_dim=in_channels, max_colors=max_colors, embedding_dim=cross_attn_2d.context_embedding_dim
    ).to(device="cuda:0")

    batch_size = 5
    color_size = 4

    palettes = torch.zeros(batch_size, color_size, 3)

    encoded = color_palette_encoder(palettes)

    assert isinstance(encoded.shape, torch.Size)
    assert encoded.shape == torch.Size([batch_size, max_colors, cross_attn_2d.context_embedding_dim])

    # test with 0-colors palette
    encodeded_empty = color_palette_encoder(torch.zeros(batch_size, 0, 3))
    assert isinstance(encodeded_empty.shape, torch.Size)
    assert encodeded_empty.shape == torch.Size([batch_size, max_colors, cross_attn_2d.context_embedding_dim])
