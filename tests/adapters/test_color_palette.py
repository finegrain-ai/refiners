import torch

from refiners.fluxion.adapters.color_palette import ColorPaletteEncoder, PalettesTokenizer


def test_colors_tokenizer() -> None:
    max_colors = 10
    tokenizer = PalettesTokenizer(max_colors=max_colors)

    batch_size = 5

    empty_palette = [[]]
    empty_palettes = empty_palette * batch_size

    color_tokens = tokenizer(empty_palettes)
    assert isinstance(color_tokens.shape, torch.Size)
    assert color_tokens.shape == torch.Size([batch_size, max_colors, 4])


def test_color_palette_encoder() -> None:
    device = "cuda:0"
    in_channels = 22
    max_colors = 10

    batch_size = 5
    color_size = 4
    # embedding_dim = cross_attn_2d.context_embedding_dim
    embedding_dim = 6

    color_palette_encoder = ColorPaletteEncoder(
        feedforward_dim=in_channels, max_colors=max_colors, embedding_dim=embedding_dim
    ).to(device=device)

    palette = [[0.0, 0.0, 0.0] for _ in range(color_size)]
    palettes = [palette] * batch_size
    
    encoded = color_palette_encoder(palettes)

    assert isinstance(encoded.shape, torch.Size)
    assert encoded.shape == torch.Size([batch_size, max_colors, embedding_dim])

    # test with 0-colors palette
    empty_palette = []
    empty_palettes = [empty_palette] * batch_size

    encoded_empty = color_palette_encoder(empty_palettes)
    assert isinstance(encoded_empty.shape, torch.Size)
    assert encoded_empty.shape == torch.Size([batch_size, max_colors, embedding_dim])

    # test with 10-colors palette
    palette = [[0.0, 0.0, 0.0] for _ in range(max_colors)]
    palettes = [palette] * batch_size

    encoded_full = color_palette_encoder(palettes)
    assert isinstance(encoded_full.shape, torch.Size)
    assert encoded_full.shape == torch.Size([batch_size, max_colors, embedding_dim])

    color_palette_encoder.to(dtype=torch.float16)
    encoded_half = color_palette_encoder(palettes)
    assert encoded_half.dtype == torch.float16
    
    encoded_mix = color_palette_encoder([palette, empty_palette])
    assert encoded_mix.shape == torch.Size([2, max_colors, embedding_dim])
