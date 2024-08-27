# Multi-View Complementary Refinement

import torch
from torch import Tensor, device as Device

import refiners.fluxion.layers as fl

from .utils import FeedForward, Interpolate, MultiPool, PatchMerge, PatchSplit, PatchwiseCrossAttention, Unflatten


class Multiply(fl.Chain):
    def __init__(self, o1: fl.Module, o2: fl.Module) -> None:
        super().__init__(o1, o2)

    def forward(self, *args: Tensor) -> Tensor:
        return torch.mul(self[0](*args), self[1](*args))


class TiledCrossAttention(fl.Chain):
    def __init__(
        self,
        emb_dim: int,
        dim: int,
        num_heads: int = 1,
        pool_ratios: list[int] | None = None,
        device: Device | None = None,
    ) -> None:
        # Input must be a 4-tuple: (local, global)

        if pool_ratios is None:
            pool_ratios = [1, 2, 4]

        super().__init__(
            fl.Distribute(
                fl.Chain(  # local
                    fl.Flatten(-2, -1),
                    fl.Permute(1, 3, 0, 2),
                ),
                fl.Chain(  # global
                    PatchSplit(),
                    fl.Squeeze(0),
                    MultiPool(pool_ratios),
                ),
            ),
            fl.Sum(
                fl.Chain(
                    fl.GetArg(0),
                    fl.Permute(2, 1, 0, 3),
                ),
                fl.Chain(
                    PatchwiseCrossAttention(emb_dim, num_heads, device=device),
                    fl.Permute(2, 1, 0, 3),
                ),
            ),
            fl.LayerNorm(emb_dim, device=device),
            FeedForward(emb_dim, device=device),
            fl.LayerNorm(emb_dim, device=device),
            fl.Permute(0, 2, 3, 1),
            Unflatten(-1, (dim, dim)),
        )


class MCRM(fl.Chain):
    """Multi-View Complementary Refinement"""

    def __init__(
        self,
        emb_dim: int,
        size: int,
        num_heads: int = 1,
        pool_ratios: list[int] | None = None,
        device: Device | None = None,
    ) -> None:
        if pool_ratios is None:
            pool_ratios = [1, 2, 4]

        super().__init__(
            fl.Parallel(
                fl.Chain(  # local
                    fl.Slicing(dim=1, end=4),
                ),
                fl.Chain(  # global
                    fl.Slicing(dim=1, start=4),
                    fl.Squeeze(1),
                ),
            ),
            fl.Parallel(
                Multiply(
                    fl.GetArg(0),
                    fl.Chain(
                        fl.GetArg(1),
                        fl.Conv2d(emb_dim, 1, 1, device=device),
                        fl.Sigmoid(),
                        Interpolate((size * 2, size * 2), "nearest"),
                        PatchSplit(),
                    ),
                ),
                fl.GetArg(1),
            ),
            fl.Parallel(
                TiledCrossAttention(emb_dim, size, num_heads, pool_ratios, device=device),
                fl.GetArg(1),
            ),
            fl.Concatenate(
                fl.GetArg(0),
                fl.Chain(
                    fl.Sum(
                        fl.GetArg(1),
                        fl.Chain(
                            fl.GetArg(0),
                            PatchMerge(),
                            Interpolate((size, size), "nearest"),
                        ),
                    ),
                    fl.Unsqueeze(1),
                ),
                dim=1,
            ),
        )
