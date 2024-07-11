import pytest

from refiners.foundationals.latent_diffusion.multi_diffusion import MultiDiffusion, Size


def test_generate_latent_tiles() -> None:
    size = Size(height=128, width=128)
    tile_size = Size(height=32, width=32)
    tiles = MultiDiffusion.generate_latent_tiles(size=size, tile_size=tile_size)
    assert len(tiles) == 25

    tiles = MultiDiffusion.generate_latent_tiles(size=size, tile_size=tile_size, min_overlap=0)
    assert len(tiles) == 16

    size = Size(height=100, width=200)
    tile_size = Size(height=32, width=32)
    tiles = MultiDiffusion.generate_latent_tiles(size=size, tile_size=tile_size, min_overlap=2)
    assert len(tiles) == 28


def test_generate_latent_tiles_small_size() -> None:
    # Test when the size is smaller than the tile size
    size = Size(height=32, width=32)
    tile_size = Size(height=64, width=64)
    tiles = MultiDiffusion.generate_latent_tiles(size=size, tile_size=tile_size)
    assert len(tiles) == 1
    assert Size(tiles[0].bottom - tiles[0].top, tiles[0].right - tiles[0].left) == size


def test_overlap_larger_tile_size() -> None:
    with pytest.raises(AssertionError):
        size = Size(height=128, width=128)
        tile_size = Size(height=32, width=32)
        MultiDiffusion.generate_latent_tiles(size=size, tile_size=tile_size, min_overlap=32)
