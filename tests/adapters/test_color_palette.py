
import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.color_palette import SD1ColorPaletteAdapter
from refiners.foundationals.latent_diffusion.stable_diffusion_1 import SD1UNet
import torch
from refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock2d

def test_single_color_palette_adapter() -> None:
    in_channels = 22
    max_colors = 10
    unet = SD1UNet(in_channels)
    cross_attn_2d = unet.ensure_find(CrossAttentionBlock2d)
    color_palette_adapter = SD1ColorPaletteAdapter(target = unet, model_dim = in_channels, max_colors= max_colors).inject()
    
    batch_size = 5
    color_size = 4
    
    palettes = torch.zeros(batch_size, color_size, 3)
    
    encoded = color_palette_adapter.color_palette_encoder(palettes)

    assert isinstance(encoded.shape, torch.Size)
    assert encoded.shape == torch.Size([batch_size, max_colors, cross_attn_2d.context_embedding_dim])
