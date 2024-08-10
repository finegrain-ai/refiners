# Multi-View Aggregation Network (arXiv:2404.07445)

from torch import device as Device

import refiners.fluxion.layers as fl
from refiners.fluxion.context import Contexts
from refiners.foundationals.swin.swin_transformer import SwinTransformer

from .mclm import MCLM  # Multi-View Complementary Localization
from .mcrm import MCRM  # Multi-View Complementary Refinement
from .utils import BatchNorm2d, Interpolate, PatchMerge, PatchSplit, PReLU, Rescale, Unflatten


class CBG(fl.Chain):
    """(C)onvolution + (B)atchNorm + (G)eLU"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int | None = None,
        device: Device | None = None,
    ):
        out_dim = out_dim or in_dim
        super().__init__(
            fl.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, device=device),
            BatchNorm2d(out_dim, device=device),
            fl.GeLU(),
        )


class CBR(fl.Chain):
    """(C)onvolution + (B)atchNorm + Parametric (R)eLU"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int | None = None,
        device: Device | None = None,
    ):
        out_dim = out_dim or in_dim
        super().__init__(
            fl.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, device=device),
            BatchNorm2d(out_dim, device=device),
            PReLU(device=device),
        )


class SplitMultiView(fl.Chain):
    """
    Split a hd tensor into 5 ld views, (5 = 1 global + 4 tiles)
    See also the reverse Module [`RearrangeMultiView`][refiners.foundationals.swin.mvanet.RearrangeMultiView]

    Inputs:
        single_view (b, c, H, W)

    Outputs:
        multi_view (b, 5, c, H/2, W/2)
    """

    def __init__(self):
        super().__init__(
            fl.Concatenate(
                PatchSplit(),  # global features
                fl.Chain(  # local features
                    Rescale(scale_factor=0.5, mode="bilinear"),
                    fl.Unsqueeze(1),
                ),
                dim=1,
            )
        )


class ShallowUpscaler(fl.Chain):
    """4x Upscaler reusing the image as input to upscale the feature
    See [[arXiv:2108.10257] SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257)

    Args:
        embedding_dim (int): the embedding dimension

    Inputs:
        feature (b, E, image_size/4, image_size/4)

    Output:
        upscaled tensor (b, E, image_size, image_size)
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        device: Device | None = None,
    ):
        super().__init__(
            fl.Sum(
                fl.Identity(),
                fl.Chain(
                    fl.UseContext("mvanet", "shallow"),
                    Interpolate((256, 256)),
                ),
            ),
            fl.Sum(
                fl.Chain(
                    Rescale(2),
                    CBG(embedding_dim, device=device),
                ),
                fl.Chain(
                    fl.UseContext("mvanet", "shallow"),
                    Interpolate((512, 512)),
                ),
            ),
            Rescale(2),
            CBG(embedding_dim, device=device),
        )


class PyramidL5(fl.Chain):
    def __init__(
        self,
        embedding_dim: int = 128,
        device: Device | None = None,
    ):
        super().__init__(
            fl.GetArg(0),  # output5
            fl.Flatten(0, 1),
            CBR(1024, embedding_dim, device=device),
            Unflatten(0, (-1, 5)),
            MCLM(embedding_dim, device=device),
            fl.Flatten(0, 1),
            Interpolate((32, 32)),
        )


class PyramidL4(fl.Chain):
    def __init__(
        self,
        embedding_dim: int = 128,
        device: Device | None = None,
    ):
        super().__init__(
            fl.Sum(
                PyramidL5(embedding_dim=embedding_dim, device=device),
                fl.Chain(
                    fl.GetArg(1),
                    fl.Flatten(0, 1),
                    CBR(512, embedding_dim, device=device),  # output4
                    Unflatten(0, (-1, 5)),
                ),
            ),
            MCRM(embedding_dim, 32, device=device),  # dec_blk4
            fl.Flatten(0, 1),
            CBR(embedding_dim, device=device),  # conv4
            Interpolate((64, 64)),
        )


class PyramidL3(fl.Chain):
    def __init__(
        self,
        embedding_dim: int = 128,
        device: Device | None = None,
    ):
        super().__init__(
            fl.Sum(
                PyramidL4(embedding_dim=embedding_dim, device=device),
                fl.Chain(
                    fl.GetArg(2),
                    fl.Flatten(0, 1),
                    CBR(256, embedding_dim, device=device),  # output3
                    Unflatten(0, (-1, 5)),
                ),
            ),
            MCRM(embedding_dim, 64, device=device),  # dec_blk3
            fl.Flatten(0, 1),
            CBR(embedding_dim, device=device),  # conv3
            Interpolate((128, 128)),
        )


class PyramidL2(fl.Chain):
    def __init__(
        self,
        embedding_dim: int = 128,
        device: Device | None = None,
    ):
        embedding_dim = 128
        super().__init__(
            fl.Sum(
                PyramidL3(embedding_dim=embedding_dim, device=device),
                fl.Chain(
                    fl.GetArg(3),
                    fl.Flatten(0, 1),
                    CBR(128, embedding_dim, device=device),  # output2
                    Unflatten(0, (-1, 5)),
                ),
            ),
            MCRM(embedding_dim, 128, device=device),  # dec_blk2
            fl.Flatten(0, 1),
            CBR(embedding_dim, device=device),  # conv2
            Interpolate((128, 128)),
        )


class Pyramid(fl.Chain):
    """
    Recursive Pyramidal Network calling MCLM and MCRM blocks

    It acts as a FPN (Feature Pyramid Network) Neck for MVANet
    see [[arXiv:1612.03144] Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)

    Inputs:
        features: a pyramid of N = 5 tensors
            shapes are (b, 5, E_{0}, S_{0}, S_{0}), ..., (b, 5, E_{1}, S_{i}, S_{i}), ..., (b, 5, E_{N-1}, S_{N-1}, S_{N-1})
            with S_{i} = S_{i-1} or S_{i} = 2*S_{i-1} for 0 < i < N

    Outputs:
        output (b, 5, E, S_{N-1}, S_{N-1})
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        device: Device | None = None,
    ):
        super().__init__(
            fl.Sum(
                PyramidL2(embedding_dim=embedding_dim, device=device),
                fl.Chain(
                    fl.GetArg(4),
                    fl.Flatten(0, 1),
                    CBR(128, embedding_dim, device=device),  # output1
                    Unflatten(0, (-1, 5)),
                ),
            ),
            MCRM(embedding_dim, 128, device=device),  # dec_blk1
            fl.Flatten(0, 1),
            CBR(embedding_dim, device=device),  # conv1
            Unflatten(0, (-1, 5)),
        )


class RearrangeMultiView(fl.Chain):
    """
    Inputs:
        multi_view (b, 5, E, H, W)

    Outputs:
        single_view (b, E, H*2, W*2)

    Fusion a multi view tensor into a single view tensor, using convolutions
    See also the reverse Module [`SplitMultiView`][refiners.foundationals.swin.mvanet.SplitMultiView]
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        device: Device | None = None,
    ):
        super().__init__(
            fl.Sum(
                fl.Chain(  # local features
                    fl.Slicing(dim=1, end=4),
                    PatchMerge(),
                ),
                fl.Chain(  # global feature
                    fl.Slicing(dim=1, start=4),
                    fl.Squeeze(1),
                    Interpolate((256, 256)),
                ),
            ),
            fl.Chain(  # conv head
                CBR(embedding_dim, 384, device=device),
                CBR(384, device=device),
                fl.Conv2d(384, embedding_dim, kernel_size=3, padding=1, device=device),
            ),
        )


class ComputeShallow(fl.Passthrough):
    def __init__(
        self,
        embedding_dim: int = 128,
        device: Device | None = None,
    ):
        super().__init__(
            fl.Conv2d(3, embedding_dim, kernel_size=3, padding=1, device=device),
            fl.SetContext("mvanet", "shallow"),
        )


class MVANet(fl.Chain):
    """Multi-view Aggregation Network for Dichotomous Image Segmentation

    See [[arXiv:2404.07445] Multi-view Aggregation Network for Dichotomous Image Segmentation](https://arxiv.org/abs/2404.07445) for more details.

    Args:
        embedding_dim (int): embedding dimension
        n_logits (int): the number of output logits (default to 1)
            1 logit is used for alpha matting/foreground-background segmentation/sod segmentation
        depths (list[int]): see [`SwinTransformer`][refiners.foundationals.swin.swin_transformer.SwinTransformer]
        num_heads (list[int]): see [`SwinTransformer`][refiners.foundationals.swin.swin_transformer.SwinTransformer]
        window_size (int): default to 12, see [`SwinTransformer`][refiners.foundationals.swin.swin_transformer.SwinTransformer]
        device (Device | None): the device to use
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        n_logits: int = 1,
        depths: list[int] | None = None,
        num_heads: list[int] | None = None,
        window_size: int = 12,
        device: Device | None = None,
    ):
        if depths is None:
            depths = [2, 2, 18, 2]
        if num_heads is None:
            num_heads = [4, 8, 16, 32]

        super().__init__(
            ComputeShallow(embedding_dim=embedding_dim, device=device),
            SplitMultiView(),
            fl.Flatten(0, 1),
            SwinTransformer(
                embedding_dim=embedding_dim,
                depths=depths,
                num_heads=num_heads,
                window_size=window_size,
                device=device,
            ),
            fl.Distribute(*(Unflatten(0, (-1, 5)) for _ in range(5))),
            Pyramid(embedding_dim=embedding_dim, device=device),
            RearrangeMultiView(embedding_dim=embedding_dim, device=device),
            ShallowUpscaler(embedding_dim, device=device),
            fl.Conv2d(embedding_dim, n_logits, kernel_size=3, padding=1, device=device),
        )

    def init_context(self) -> Contexts:
        return {"mvanet": {"shallow": None}}
