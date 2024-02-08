import torch

from refiners.fluxion.adapters.color_palette import ColorPaletteEncoder, PalettesTokenizer, ColorPaletteExtractor, ColorPalette
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder

def test_colors_tokenizer() -> None:
    max_colors = 10
    tokenizer = PalettesTokenizer(max_colors=max_colors, lda=SD1Autoencoder(), weighted_palette=True)

    batch_size = 5

    empty_palette : list[ColorPalette] = [[]]
    empty_palettes = empty_palette * batch_size

    color_tokens = tokenizer(empty_palettes)
    assert isinstance(color_tokens.shape, torch.Size)
    assert color_tokens.shape == torch.Size([batch_size, max_colors, 5])

    tokenizer2 = PalettesTokenizer(max_colors=max_colors, lda=SD1Autoencoder(), weighted_palette=False)

    batch_size = 5

    color_tokens2 = tokenizer2(empty_palettes)
    assert isinstance(color_tokens2.shape, torch.Size)
    assert color_tokens2.shape == torch.Size([batch_size, max_colors, 4])

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

    palette : ColorPalette = [((0, 0, 0), 30) for _ in range(color_size)]
    palettes : list[ColorPalette] = [palette] * batch_size
    
    encoded = color_palette_encoder(palettes)

    assert isinstance(encoded.shape, torch.Size)
    assert encoded.shape == torch.Size([batch_size, max_colors, embedding_dim])

    # test with 0-colors palette
    empty_palette: list[ColorPalette] = []
    empty_palettes = [empty_palette] * batch_size

    encoded_empty = color_palette_encoder(empty_palettes)
    assert isinstance(encoded_empty.shape, torch.Size)
    assert encoded_empty.shape == torch.Size([batch_size, max_colors, embedding_dim])

    # test with 10-colors palette
    palette = [((0, 0, 0), 30) for _ in range(max_colors)]
    palettes = [palette] * batch_size

    encoded_full = color_palette_encoder(palettes)
    assert isinstance(encoded_full.shape, torch.Size)
    assert encoded_full.shape == torch.Size([batch_size, max_colors, embedding_dim])

    color_palette_encoder.to(dtype=torch.float16)
    encoded_half = color_palette_encoder(palettes)
    assert encoded_half.dtype == torch.float16
    
    encoded_mix = color_palette_encoder([palette, empty_palette])
    assert encoded_mix.shape == torch.Size([2, max_colors, embedding_dim])


def test_lda_color_palette_encoder() -> None:
    device = "cuda:0"
    in_channels = 22
    max_colors = 10

    batch_size = 5
    color_size = 4
    # embedding_dim = cross_attn_2d.context_embedding_dim
    embedding_dim = 6

    color_palette_encoder = ColorPaletteEncoder(
        feedforward_dim=in_channels, 
        max_colors=max_colors, 
        embedding_dim=embedding_dim,
        use_lda=True,
        lda=SD1Autoencoder()
    ).to(device=device)

    palette : ColorPalette = [((0, 0, 0), 30) for _ in range(color_size)]
    palettes = [palette] * batch_size
    
    encoded = color_palette_encoder(palettes)

    assert isinstance(encoded.shape, torch.Size)
    assert encoded.shape == torch.Size([batch_size, max_colors, embedding_dim])

    # test with 0-colors palette
    empty_palette = []
    empty_palettes: list[ColorPalette] = [empty_palette] * batch_size

    encoded_empty = color_palette_encoder(empty_palettes)
    assert isinstance(encoded_empty.shape, torch.Size)
    assert encoded_empty.shape == torch.Size([batch_size, max_colors, embedding_dim])

    # test with 10-colors palette
    palette : ColorPalette = [((0, 0, 0), 30) for _ in range(max_colors)]
    palettes = [palette] * batch_size

    encoded_full = color_palette_encoder(palettes)
    assert isinstance(encoded_full.shape, torch.Size)
    assert encoded_full.shape == torch.Size([batch_size, max_colors, embedding_dim])

    color_palette_encoder.to(dtype=torch.float16)
    encoded_half = color_palette_encoder(palettes)
    assert encoded_half.dtype == torch.float16
    
    encoded_mix = color_palette_encoder([palette, empty_palette])
    assert encoded_mix.shape == torch.Size([2, max_colors, embedding_dim])

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

   
    palette : ColorPalette = [((0, 0, 0), 30) for _ in range(color_size)]
    palettes = [palette] * batch_size
    
    encoded = color_palette_encoder(palettes)

    assert isinstance(encoded.shape, torch.Size)
    assert encoded.shape == torch.Size([batch_size, max_colors, embedding_dim])

    # test with 0-colors palette
    empty_palette = []
    empty_palettes: list[ColorPalette] = [empty_palette] * batch_size

    encoded_empty = color_palette_encoder(empty_palettes)
    assert isinstance(encoded_empty.shape, torch.Size)
    assert encoded_empty.shape == torch.Size([batch_size, max_colors, embedding_dim])

    # test with 10-colors palette
    palette : ColorPalette = [((0, 0, 0), 30) for _ in range(max_colors)]
    palettes = [palette] * batch_size

    encoded_full = color_palette_encoder(palettes)
    assert isinstance(encoded_full.shape, torch.Size)
    assert encoded_full.shape == torch.Size([batch_size, max_colors, embedding_dim])

    color_palette_encoder.to(dtype=torch.float16)
    encoded_half = color_palette_encoder(palettes)
    assert encoded_half.dtype == torch.float16
    
    encoded_mix = color_palette_encoder([palette, empty_palette])
    assert encoded_mix.shape == torch.Size([2, max_colors, embedding_dim])


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

    
    palette : ColorPalette = [((0, 0, 0), 30) for _ in range(max_colors)]
    palettes = [palette] * batch_size
    
    encoded = color_palette_encoder(palettes)

    assert isinstance(encoded.shape, torch.Size)
    assert encoded.shape == torch.Size([batch_size, max_colors, embedding_dim])

    # test with 0-colors palette
    empty_palette = []
    empty_palettes: list[ColorPalette] = [empty_palette] * batch_size

    encoded_empty = color_palette_encoder(empty_palettes)
    assert isinstance(encoded_empty.shape, torch.Size)
    assert encoded_empty.shape == torch.Size([batch_size, max_colors, embedding_dim])

    # test with 10-colors palette
    palette : ColorPalette = [((0, 0, 0), 30) for _ in range(max_colors)]
    palettes = [palette] * batch_size

    encoded_full = color_palette_encoder(palettes)
    assert isinstance(encoded_full.shape, torch.Size)
    assert encoded_full.shape == torch.Size([batch_size, max_colors, embedding_dim])

    color_palette_encoder.to(dtype=torch.float16)
    encoded_half = color_palette_encoder(palettes)
    assert encoded_half.dtype == torch.float16
    
    encoded_mix = color_palette_encoder([palette, empty_palette])
    assert encoded_mix.shape == torch.Size([2, max_colors, embedding_dim])


from PIL import Image

def test_palette_extractor() -> None:

    white_image = Image.open("tests/fixtures/photo-1439246854758-f686a415d9da.jpeg").resize((512, 512))
    palette_size = 8
    color_palette_extractor = ColorPaletteExtractor(
        size = palette_size
    )
    
    palette = color_palette_extractor(white_image)
    
    assert len(palette) == palette_size
    assert isinstance(palette[0], tuple)
    assert isinstance(palette[0][1], float)
    assert len(palette[0][0]) == 3
#    assert isinstance(palette[0][0][0], int)