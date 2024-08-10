# Swin Transformer (arXiv:2103.14030)
#
# Specific to MVANet, only supports square inputs.
# Originally adapted from the version in MVANet and InSPyReNet (https://github.com/plemeri/InSPyReNet)
# Original implementation by Microsoft at https://github.com/microsoft/Swin-Transformer

import functools
from math import isqrt

import torch
from torch import Tensor, device as Device

import refiners.fluxion.layers as fl
from refiners.fluxion.context import Contexts


def to_windows(x: Tensor, window_size: int) -> Tensor:
    B, H, W, C = x.shape
    assert W == H and H % window_size == 0
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, -1, window_size * window_size, C)


class ToWindows(fl.Module):
    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:
        return to_windows(x, self.window_size)


class FromWindows(fl.Module):
    def forward(self, x: Tensor) -> Tensor:
        B, num_windows, window_size_2, C = x.shape
        window_size = isqrt(window_size_2)
        H = isqrt(num_windows * window_size_2)
        x = x.reshape(B, H // window_size, H // window_size, window_size, window_size, C)
        return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, H, C)


@functools.cache
def get_attn_mask(H: int, window_size: int, device: Device | None = None) -> Tensor:
    assert H % window_size == 0
    shift_size = window_size // 2
    img_mask = torch.zeros((1, H, H, 1), device=device)
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = to_windows(img_mask, window_size).squeeze()  # B, nW, window_size * window_size, [1]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask.masked_fill_(attn_mask != 0, -100.0).masked_fill_(attn_mask == 0, 0.0)
    return attn_mask


class Pad(fl.Module):
    def __init__(self, step: int):
        super().__init__()
        self.step = step

    def forward(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        assert W == H
        if H % self.step == 0:
            return x
        p = self.step * ((H + self.step - 1) // self.step)
        padded = torch.zeros(B, p, p, C, device=x.device, dtype=x.dtype)
        padded[:, :H, :H, :] = x
        return padded


class StatefulPad(fl.Chain):
    def __init__(self, context: str, key: str, step: int) -> None:
        super().__init__(
            fl.SetContext(context=context, key=key, callback=self._push),
            Pad(step=step),
        )

    def _push(self, sizes: list[int], x: Tensor) -> None:
        sizes.append(x.size(1))


class StatefulUnpad(fl.Chain):
    def __init__(self, context: str, key: str) -> None:
        super().__init__(
            fl.Parallel(
                fl.Identity(),
                fl.UseContext(context=context, key=key).compose(lambda x: x.pop()),
            ),
            fl.Lambda(self._unpad),
        )

    @staticmethod
    def _unpad(x: Tensor, size: int) -> Tensor:
        return x[:, :size, :size, :]


class SquareUnflatten(fl.Module):
    # ..., L^2, ... -> ..., L, L, ...

    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        d = isqrt(x.shape[self.dim])
        return torch.unflatten(x, self.dim, (d, d))


class WindowUnflatten(fl.Module):
    # ..., H, ... -> ..., H // ws, ws, ...

    def __init__(self, window_size: int, dim: int = 0) -> None:
        super().__init__()
        self.window_size = window_size
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[self.dim] % self.window_size == 0
        H = x.shape[self.dim]
        return torch.unflatten(x, self.dim, (H // self.window_size, self.window_size))


class Roll(fl.Module):
    def __init__(self, *shifts: tuple[int, int]):
        super().__init__()
        self.shifts = shifts
        self._dims = tuple(s[0] for s in shifts)
        self._shifts = tuple(s[1] for s in shifts)

    def forward(self, x: Tensor) -> Tensor:
        return torch.roll(x, self._shifts, self._dims)


class RelativePositionBias(fl.Module):
    relative_position_index: Tensor

    def __init__(self, window_size: int, num_heads: int, device: Device | None = None):
        super().__init__()
        self.relative_position_bias_table = torch.nn.Parameter(
            torch.empty(
                (2 * window_size - 1) * (2 * window_size - 1),
                num_heads,
                device=device,
            )
        )
        relative_position_index = torch.empty(
            window_size**2,
            window_size**2,
            device=device,
            dtype=torch.int64,
        )
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self) -> Tensor:
        # Yes, this is a (trainable) constant.
        return self.relative_position_bias_table[self.relative_position_index].permute(2, 0, 1).unsqueeze(0)


class WindowSDPA(fl.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        shift: bool = False,
        device: Device | None = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift = shift
        self.rpb = RelativePositionBias(window_size, num_heads, device=device)

    def forward(self, x: Tensor):
        B, num_windows, N, _C = x.shape
        assert _C % (3 * self.num_heads) == 0
        C = _C // 3
        x = torch.reshape(x, (B * num_windows, N, 3, self.num_heads, C // self.num_heads))
        q, k, v = x.permute(2, 0, 3, 1, 4)

        attn_mask = self.rpb()
        if self.shift:
            mask = get_attn_mask(isqrt(num_windows * (self.window_size**2)), self.window_size, x.device)
            mask = mask.reshape(1, num_windows, 1, N, N)
            mask = mask.expand(B, -1, self.num_heads, -1, -1)
            attn_mask = attn_mask + mask.reshape(-1, self.num_heads, N, N)

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask)
        x = x.transpose(1, 2).reshape(B, num_windows, N, C)
        return x


class WindowAttention(fl.Chain):
    """
    Window-based Multi-head Self-Attenion (W-MSA), optionally shifted (SW-MSA).

    It has a trainable relative position bias (RelativePositionBias).

    The input projection is stored as a single Linear for q, k and v.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        shift: bool = False,
        device: Device | None = None,
    ):
        super().__init__(
            fl.Linear(dim, dim * 3, bias=True, device=device),
            WindowSDPA(dim, window_size, num_heads, shift, device=device),
            fl.Linear(dim, dim, device=device),
        )


class SwinTransformerBlock(fl.Chain):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        device: Device | None = None,
    ):
        assert 0 <= shift_size < window_size, "shift_size must in [0, window_size["

        super().__init__(
            fl.Residual(
                fl.LayerNorm(dim, device=device),
                SquareUnflatten(1),
                StatefulPad(context="padding", key="sizes", step=window_size),
                Roll((1, -shift_size), (2, -shift_size)),
                ToWindows(window_size),
                WindowAttention(
                    dim,
                    window_size=window_size,
                    num_heads=num_heads,
                    shift=shift_size > 0,
                    device=device,
                ),
                FromWindows(),
                Roll((1, shift_size), (2, shift_size)),
                StatefulUnpad(context="padding", key="sizes"),
                fl.Flatten(1, 2),
            ),
            fl.Residual(
                fl.LayerNorm(dim, device=device),
                fl.Linear(dim, int(dim * mlp_ratio), device=device),
                fl.GeLU(),
                fl.Linear(int(dim * mlp_ratio), dim, device=device),
            ),
        )

    def init_context(self) -> Contexts:
        return {"padding": {"sizes": []}}


class PatchMerging(fl.Chain):
    def __init__(self, dim: int, device: Device | None = None):
        super().__init__(
            SquareUnflatten(1),
            Pad(2),
            WindowUnflatten(2, 2),
            WindowUnflatten(2, 1),
            fl.Permute(0, 1, 3, 4, 2, 5),
            fl.Flatten(3),
            fl.Flatten(1, 2),
            fl.LayerNorm(4 * dim, device=device),
            fl.Linear(4 * dim, 2 * dim, bias=False, device=device),
        )


class BasicLayer(fl.Chain):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        device: Device | None = None,
    ):
        super().__init__(
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                device=device,
            )
            for i in range(depth)
        )


class PatchEmbedding(fl.Chain):
    def __init__(
        self,
        patch_size: tuple[int, int] = (4, 4),
        in_chans: int = 3,
        embedding_dim: int = 96,
        device: Device | None = None,
    ):
        super().__init__(
            fl.Conv2d(in_chans, embedding_dim, kernel_size=patch_size, stride=patch_size, device=device),
            fl.Flatten(2),
            fl.Transpose(1, 2),
            fl.LayerNorm(embedding_dim, device=device),
        )


class SwinTransformer(fl.Chain):
    """Swin Transformer (arXiv:2103.14030)

    Currently specific to MVANet, only supports square inputs.
    """

    def __init__(
        self,
        patch_size: tuple[int, int] = (4, 4),
        in_chans: int = 3,
        embedding_dim: int = 96,
        depths: list[int] | None = None,
        num_heads: list[int] | None = None,
        window_size: int = 7,  # image size is 32 * this
        mlp_ratio: float = 4.0,
        device: Device | None = None,
    ):
        if depths is None:
            depths = [2, 2, 6, 2]

        if num_heads is None:
            num_heads = [3, 6, 12, 24]

        self.num_layers = len(depths)
        assert len(num_heads) == self.num_layers

        super().__init__(
            PatchEmbedding(
                patch_size=patch_size,
                in_chans=in_chans,
                embedding_dim=embedding_dim,
                device=device,
            ),
            fl.Passthrough(
                fl.Transpose(1, 2),
                SquareUnflatten(2),
                fl.SetContext("swin", "outputs", callback=lambda t, x: t.append(x)),
            ),
            *(
                fl.Chain(
                    BasicLayer(
                        dim=int(embedding_dim * 2**i),
                        depth=depths[i],
                        num_heads=num_heads[i],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        device=device,
                    ),
                    fl.Passthrough(
                        fl.LayerNorm(int(embedding_dim * 2**i), device=device),
                        fl.Transpose(1, 2),
                        SquareUnflatten(2),
                        fl.SetContext("swin", "outputs", callback=lambda t, x: t.insert(0, x)),
                    ),
                    PatchMerging(dim=int(embedding_dim * 2**i), device=device)
                    if i < self.num_layers - 1
                    else fl.UseContext("swin", "outputs").compose(lambda t: tuple(t)),
                )
                for i in range(self.num_layers)
            ),
        )

    def init_context(self) -> Contexts:
        return {"swin": {"outputs": []}}
