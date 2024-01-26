import torch

from refiners.fluxion.adapters.color_palette import ColorPaletteEncoder, ColorsTokenizer
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder

def test_colors_tokenizer() -> None:
    max_colors = 10
    tokenizer = ColorsTokenizer(max_colors=max_colors)

    batch_size = 5

    colors = torch.zeros(batch_size, 0, 3)

    color_tokens = tokenizer(colors)
    assert isinstance(color_tokens.shape, torch.Size)
    assert color_tokens.shape == torch.Size([batch_size, max_colors, 4])


def test_color_palette_encoder() -> None:
    device = "cuda:0"
    in_channels = 22
    max_colors = 10

    batch_size = 5
    color_size = 4
    # embedding_dim = cross_attn_2d.context_embedding_dim
    embedding_dim = 5

    color_palette_encoder = ColorPaletteEncoder(
        feedforward_dim=in_channels, max_colors=max_colors, embedding_dim=embedding_dim
    ).to(device=device)

    palettes = torch.zeros(batch_size, color_size, 3, device=device)

    encoded = color_palette_encoder(palettes)

    assert isinstance(encoded.shape, torch.Size)
    assert encoded.shape == torch.Size([batch_size, max_colors, embedding_dim])

    # test with 0-colors palette
    encoded_empty = color_palette_encoder(torch.zeros(batch_size, 0, 3, device=device))
    assert isinstance(encoded_empty.shape, torch.Size)
    assert encoded_empty.shape == torch.Size([batch_size, max_colors, embedding_dim])

    # test with 10-colors palette
    encoded_full = color_palette_encoder(torch.zeros(batch_size, max_colors, 3, device=device))
    assert isinstance(encoded_full.shape, torch.Size)
    assert encoded_full.shape == torch.Size([batch_size, max_colors, embedding_dim])

    color_palette_encoder.to(dtype=torch.float16)
    palette = torch.zeros(batch_size, max_colors, 3, dtype=torch.float16, device=device)
    encoded_half = color_palette_encoder(palette)
    assert encoded_half.dtype == torch.float16


def test_lda_color_palette_encoder() -> None:
    device = "cuda:0"
    in_channels = 22
    max_colors = 10

    batch_size = 5
    color_size = 4
    # embedding_dim = cross_attn_2d.context_embedding_dim
    embedding_dim = 5

    color_palette_encoder = ColorPaletteEncoder(
        feedforward_dim=in_channels, 
        max_colors=max_colors, 
        embedding_dim=embedding_dim,
        use_lda=True,
        lda=SD1Autoencoder()
    ).to(device=device)

    palettes = torch.zeros(batch_size, color_size, 3, device=device)

    encoded = color_palette_encoder(palettes)

    assert isinstance(encoded.shape, torch.Size)
    assert encoded.shape == torch.Size([batch_size, max_colors, embedding_dim])

    # test with 0-colors palette
    encoded_empty = color_palette_encoder(torch.zeros(batch_size, 0, 3, device=device))
    assert isinstance(encoded_empty.shape, torch.Size)
    assert encoded_empty.shape == torch.Size([batch_size, max_colors, embedding_dim])

    # test with 10-colors palette
    encoded_full = color_palette_encoder(torch.zeros(batch_size, max_colors, 3, device=device))
    assert isinstance(encoded_full.shape, torch.Size)
    assert encoded_full.shape == torch.Size([batch_size, max_colors, embedding_dim])

    color_palette_encoder.to(dtype=torch.float16)
    palette = torch.zeros(batch_size, max_colors, 3, dtype=torch.float16, device=device)
    encoded_half = color_palette_encoder(palette)
    assert encoded_half.dtype == torch.float16

def test_2_layer_color_palette_encoder() -> None:
    device = "cuda:0"
    in_channels = 22
    max_colors = 10

    batch_size = 5
    color_size = 4
    # embedding_dim = cross_attn_2d.context_embedding_dim
    embedding_dim = 5

    color_palette_encoder = ColorPaletteEncoder(
        feedforward_dim=in_channels, 
        max_colors=max_colors, 
        num_layers=2,
        embedding_dim=embedding_dim,
        mode='mlp'
    ).to(device=device)

    palettes = torch.zeros(batch_size, color_size, 3, device=device)

    encoded = color_palette_encoder(palettes)

    assert isinstance(encoded.shape, torch.Size)
    assert encoded.shape == torch.Size([batch_size, max_colors, embedding_dim])

    # test with 0-colors palette
    encoded_empty = color_palette_encoder(torch.zeros(batch_size, 0, 3, device=device))
    assert isinstance(encoded_empty.shape, torch.Size)
    assert encoded_empty.shape == torch.Size([batch_size, max_colors, embedding_dim])

    # test with 10-colors palette
    encoded_full = color_palette_encoder(torch.zeros(batch_size, max_colors, 3, device=device))
    assert isinstance(encoded_full.shape, torch.Size)
    assert encoded_full.shape == torch.Size([batch_size, max_colors, embedding_dim])

    color_palette_encoder.to(dtype=torch.float16)
    palette = torch.zeros(batch_size, max_colors, 3, dtype=torch.float16, device=device)
    encoded_half = color_palette_encoder(palette)
    assert encoded_half.dtype == torch.float16


def test_0_layer_color_palette_encoder() -> None:
    device = "cuda:0"
    in_channels = 22
    max_colors = 10

    batch_size = 5
    color_size = 4
    # embedding_dim = cross_attn_2d.context_embedding_dim
    embedding_dim = 5

    color_palette_encoder = ColorPaletteEncoder(
        feedforward_dim=in_channels, 
        max_colors=max_colors, 
        num_layers=0,
        embedding_dim=embedding_dim,
        mode='mlp'
    ).to(device=device)

    palettes = torch.zeros(batch_size, color_size, 3, device=device)

    encoded = color_palette_encoder(palettes)

    assert isinstance(encoded.shape, torch.Size)
    assert encoded.shape == torch.Size([batch_size, max_colors, embedding_dim])

    # test with 0-colors palette
    encoded_empty = color_palette_encoder(torch.zeros(batch_size, 0, 3, device=device))
    assert isinstance(encoded_empty.shape, torch.Size)
    assert encoded_empty.shape == torch.Size([batch_size, max_colors, embedding_dim])

    # test with 10-colors palette
    encoded_full = color_palette_encoder(torch.zeros(batch_size, max_colors, 3, device=device))
    assert isinstance(encoded_full.shape, torch.Size)
    assert encoded_full.shape == torch.Size([batch_size, max_colors, embedding_dim])

    color_palette_encoder.to(dtype=torch.float16)
    palette = torch.zeros(batch_size, max_colors, 3, dtype=torch.float16, device=device)
    encoded_half = color_palette_encoder(palette)
    assert encoded_half.dtype == torch.float16