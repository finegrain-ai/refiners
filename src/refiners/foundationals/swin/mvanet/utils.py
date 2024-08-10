import torch
from torch import Size, Tensor
from torch.nn.functional import (
    adaptive_avg_pool2d,
    interpolate,  # type: ignore
)

import refiners.fluxion.layers as fl


class Unflatten(fl.Module):
    def __init__(self, dim: int, sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.dim = dim
        self.sizes = Size(sizes)

    def forward(self, x: Tensor) -> Tensor:
        return torch.unflatten(input=x, dim=self.dim, sizes=self.sizes)


class Interpolate(fl.Module):
    def __init__(self, size: tuple[int, ...], mode: str = "bilinear"):
        super().__init__()
        self.size = Size(size)
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        return interpolate(x, size=self.size, mode=self.mode)  # type: ignore


class Rescale(fl.Module):
    def __init__(self, scale_factor: float, mode: str = "nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        return interpolate(x, scale_factor=self.scale_factor, mode=self.mode)  # type: ignore


class BatchNorm2d(torch.nn.BatchNorm2d, fl.WeightedModule):
    def __init__(self, num_features: int, device: torch.device | None = None):
        super().__init__(num_features=num_features, device=device)  # type: ignore


class PReLU(torch.nn.PReLU, fl.WeightedModule, fl.Activation):
    def __init__(self, device: torch.device | None = None):
        super().__init__(device=device)  # type: ignore


class PatchSplit(fl.Chain):
    """(B, N, H, W) -> B, 4, N, H/2, W/2"""

    def __init__(self):
        super().__init__(
            Unflatten(-2, (2, -1)),
            Unflatten(-1, (2, -1)),
            fl.Permute(0, 2, 4, 1, 3, 5),
            fl.Flatten(1, 2),
        )


class PatchMerge(fl.Chain):
    """B, 4, N, H, W -> (B, N, 2*H, 2*W)"""

    def __init__(self):
        super().__init__(
            Unflatten(1, (2, 2)),
            fl.Permute(0, 3, 1, 4, 2, 5),
            fl.Flatten(-2, -1),
            fl.Flatten(-3, -2),
        )


class FeedForward(fl.Residual):
    def __init__(self, emb_dim: int, device: torch.device | None = None) -> None:
        super().__init__(
            fl.Linear(in_features=emb_dim, out_features=2 * emb_dim, device=device),
            fl.ReLU(),
            fl.Linear(in_features=2 * emb_dim, out_features=emb_dim, device=device),
        )


class _GetArgs(fl.Parallel):
    def __init__(self, n: int):
        super().__init__(
            fl.Chain(
                fl.GetArg(0),
                fl.Slicing(dim=0, start=n, end=n + 1),
                fl.Squeeze(0),
            ),
            fl.Chain(
                fl.GetArg(1),
                fl.Slicing(dim=0, start=n, end=n + 1),
                fl.Squeeze(0),
            ),
            fl.Chain(
                fl.GetArg(1),
                fl.Slicing(dim=0, start=n, end=n + 1),
                fl.Squeeze(0),
            ),
        )


class MultiheadAttention(torch.nn.MultiheadAttention, fl.WeightedModule):
    def __init__(self, embedding_dim: int, num_heads: int, device: torch.device | None = None):
        super().__init__(embed_dim=embedding_dim, num_heads=num_heads, device=device)  # type: ignore

    @property
    def weight(self) -> Tensor:  # type: ignore
        return self.in_proj_weight

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:  # type: ignore
        return super().forward(q, k, v)[0]


class PatchwiseCrossAttention(fl.Chain):
    # Input is 2 tensors of sizes (4, HW, B, C) and (4, HW', B, C),
    # output is size (4, HW, B, C).
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
    ):
        super().__init__(
            fl.Concatenate(
                fl.Chain(
                    _GetArgs(0),
                    MultiheadAttention(d_model, num_heads, device=device),
                ),
                fl.Chain(
                    _GetArgs(1),
                    MultiheadAttention(d_model, num_heads, device=device),
                ),
                fl.Chain(
                    _GetArgs(2),
                    MultiheadAttention(d_model, num_heads, device=device),
                ),
                fl.Chain(
                    _GetArgs(3),
                    MultiheadAttention(d_model, num_heads, device=device),
                ),
            ),
            Unflatten(0, (4, -1)),
        )


class Pool(fl.Module):
    def __init__(self, ratio: int) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape
        assert h % self.ratio == 0 and w % self.ratio == 0
        r = adaptive_avg_pool2d(x, (h // self.ratio, w // self.ratio))
        return torch.unflatten(r, 0, (b, -1))


class MultiPool(fl.Concatenate):
    def __init__(self, pool_ratios: list[int]) -> None:
        super().__init__(
            *(
                fl.Chain(
                    Pool(pool_ratio),
                    fl.Flatten(-2, -1),
                    fl.Permute(0, 3, 1, 2),
                )
                for pool_ratio in pool_ratios
            ),
            dim=1,
        )
