from typing import Annotated, NamedTuple, cast

import torch

from refiners.fluxion import layers as fl


class FluxParams(NamedTuple):
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class DoubleStreamBlocks(fl.Chain):
    pass


class SingleStreamBlocks(fl.Chain):
    pass


class PositionalEncoding(fl.Passthrough):
    pass


class TimestepEmbedding(fl.Passthrough):
    pass


class TextEmbedding(fl.Passthrough):
    pass


class DoubleStreamAttention(fl.Chain):
    pass


class SingleStreamAttention(fl.Chain):
    pass


class LastLayer(fl.Chain):
    def __init__(self, params: FluxParams) -> None:
        super().__init__(fl.LayerNorm(params.hidden_size), fl.Linear(params.hidden_size, params.in_channels))


class Flux(fl.Sum):
    def __init__(
        self,
        params: FluxParams,
    ) -> None:
        super().__init__(
            TimestepEmbedding(),
            PositionalEncoding(),
            TextEmbedding(),
            DoubleStreamBlocks(DoubleStreamAttention() for _ in range(params.depth)),
            SingleStreamBlocks(SingleStreamAttention() for _ in range(params.depth_single_blocks)),
        )


if __name__ == "__main__":
    flux = Flux(
        params=FluxParams(
            in_channels=3,
            vec_in_dim=32,
            context_in_dim=64,
            hidden_size=256,
            mlp_ratio=4.0,
            num_heads=8,
            depth=2,
            depth_single_blocks=1,
            axes_dim=[32, 32],
            theta=1024,
            qkv_bias=True,
            guidance_embed=True,
        ),
    )
    from typing import get_type_hints

    print(repr(flux))
    image_in = flux.layer(0)
    print(image_in)
    print(get_type_hints(image_in))
