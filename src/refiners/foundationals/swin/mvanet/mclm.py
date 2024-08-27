# Multi-View Complementary Localization

import math

import torch
from torch import Tensor, device as Device

import refiners.fluxion.layers as fl
from refiners.fluxion.context import Contexts

from .utils import FeedForward, MultiheadAttention, MultiPool, PatchMerge, PatchwiseCrossAttention, Unflatten


class PerPixel(fl.Chain):
    """(B, C, H, W) -> H*W, B, C"""

    def __init__(self) -> None:
        super().__init__(
            fl.Permute(2, 3, 0, 1),
            fl.Flatten(0, 1),
        )


class PositionEmbeddingSine(fl.Module):
    """
    Non-trainable position embedding, originally from https://github.com/facebookresearch/detr
    """

    def __init__(self, num_pos_feats: int) -> None:
        super().__init__()
        temperature = 10000
        self.dim_t = torch.arange(0, num_pos_feats, dtype=torch.float32)
        self.dim_t = temperature ** (2 * (self.dim_t // 2) / num_pos_feats)

    def __call__(self, h: int, w: int) -> Tensor:
        mask = torch.ones([1, h, w, 1], dtype=torch.bool)
        y_embed = mask.cumsum(dim=1, dtype=torch.float32)
        x_embed = mask.cumsum(dim=2, dtype=torch.float32)

        eps, scale = 1e-6, 2 * math.pi
        y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * scale
        x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * scale

        pos_x = x_embed / self.dim_t
        pos_y = y_embed / self.dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(1, 2, 0, 3).flatten(0, 1)


class MultiPoolPos(fl.Module):
    def __init__(self, pool_ratios: list[int], positional_embedding: PositionEmbeddingSine) -> None:
        super().__init__()
        self.pool_ratios = pool_ratios
        self.positional_embedding = positional_embedding

    def forward(self, *args: int) -> Tensor:
        h, w = args
        return torch.cat([self.positional_embedding(h // ratio, w // ratio) for ratio in self.pool_ratios])


class Repeat(fl.Module):
    def __init__(self, dim: int = 0) -> None:
        self.dim = dim
        super().__init__()

    def forward(self, x: Tensor, n: int) -> Tensor:
        return torch.repeat_interleave(x, n, dim=self.dim)


class _MHA_Arg(fl.Sum):
    def __init__(self, offset: int) -> None:
        self.offset = offset
        super().__init__(
            fl.GetArg(offset),  # value
            fl.Chain(
                fl.Parallel(
                    fl.GetArg(self.offset + 1),  # position embedding
                    fl.Lambda(self._batch_size),
                ),
                Repeat(1),
            ),
        )

    def _batch_size(self, *args: Tensor) -> int:
        return args[self.offset].size(1)


class GlobalAttention(fl.Chain):
    # Input must be a 4-tuple: (global, global pos. emb, pools, pools pos. emb.)
    def __init__(
        self,
        emb_dim: int,
        num_heads: int = 1,
        device: Device | None = None,
    ) -> None:
        super().__init__(
            fl.Sum(
                fl.GetArg(0),  # global
                fl.Chain(
                    fl.Parallel(
                        _MHA_Arg(0),  # Q: global + pos. emb
                        _MHA_Arg(2),  # K: pools + pos. emb
                        fl.GetArg(2),  # V: pools
                    ),
                    MultiheadAttention(emb_dim, num_heads, device=device),
                ),
            ),
        )


class MCLM(fl.Chain):
    """Multi-View Complementary Localization Module
    Inputs:
        tensor: (b, 5, e, h, h)
    Outputs:
        tensor: (b, 5, e, h, h)
    """

    def __init__(
        self,
        emb_dim: int,
        num_heads: int = 1,
        pool_ratios: list[int] | None = None,
        device: Device | None = None,
    ) -> None:
        if pool_ratios is None:
            pool_ratios = [2, 8, 16]

        positional_embedding = PositionEmbeddingSine(num_pos_feats=emb_dim // 2)

        # LayerNorms in MCLM share their weights.
        # We use the `proxy` trick below so they can be present only
        # once in the tree but called in two different places.

        ln1 = fl.LayerNorm(emb_dim, device=device)
        ln2 = fl.LayerNorm(emb_dim, device=device)

        def proxy(m: fl.Module) -> fl.Module:
            def f(x: Tensor) -> Tensor:
                return m(x)

            return fl.Lambda(f)

        super().__init__(
            fl.Parallel(
                fl.Chain(  # global
                    fl.Slicing(dim=1, start=4),
                    fl.Squeeze(1),
                    fl.Parallel(
                        PerPixel(),  # glb
                        fl.Chain(  # g_pos
                            fl.Lambda(lambda x: x.shape[-2:]),  # type: ignore
                            positional_embedding,
                        ),
                    ),
                ),
                fl.Chain(  # local
                    fl.Slicing(dim=1, end=4),
                    fl.SetContext("mclm", "local"),
                    PatchMerge(),
                    fl.Parallel(
                        fl.Chain(  # pool
                            MultiPool(pool_ratios),
                            fl.Squeeze(0),
                        ),
                        fl.Chain(  # pool_pos
                            fl.Lambda(lambda x: x.shape[-2:]),  # type: ignore
                            MultiPoolPos(pool_ratios, positional_embedding),
                        ),
                    ),
                ),
            ),
            fl.Lambda(lambda t1, t2: (*t1, *t2)),  # type: ignore
            fl.Converter(set_dtype=False),
            GlobalAttention(emb_dim, num_heads, device=device),
            ln1,
            FeedForward(emb_dim, device=device),
            ln2,
            fl.SetContext("mclm", "global"),
            fl.UseContext("mclm", "local"),
            fl.Flatten(-2, -1),
            fl.Permute(1, 3, 0, 2),
            fl.Residual(
                fl.Parallel(
                    fl.Identity(),
                    fl.Chain(
                        fl.UseContext("mclm", "global"),
                        Unflatten(0, (2, 8, 2, 8)),  # 2, h/2, 2, h/2
                        fl.Permute(0, 2, 1, 3, 4, 5),
                        fl.Flatten(0, 1),
                        fl.Flatten(1, 2),
                    ),
                ),
                PatchwiseCrossAttention(emb_dim, num_heads, device=device),
            ),
            proxy(ln1),
            FeedForward(emb_dim, device=device),
            proxy(ln2),
            fl.Concatenate(
                fl.Identity(),
                fl.Chain(
                    fl.UseContext("mclm", "global"),
                    fl.Unsqueeze(0),
                ),
            ),
            Unflatten(1, (16, 16)),  # h, h
            fl.Permute(3, 0, 4, 1, 2),
        )

    def init_context(self) -> Contexts:
        return {"mclm": {"global": None, "local": None}}
