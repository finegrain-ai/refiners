from typing import Iterable, cast

from torch import Tensor, device as Device, dtype as DType

import refiners.fluxion.layers as fl
from refiners.fluxion.context import Contexts
from refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock2d
from refiners.foundationals.latent_diffusion.range_adapter import RangeAdapter2d, RangeEncoder
from refiners.foundationals.latent_diffusion.unet import (
    ResidualAccumulator,
    ResidualBlock,
    ResidualConcatenator,
)


class TimestepEncoder(fl.Passthrough):
    def __init__(
        self,
        context_key: str = "timestep_embedding",
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            fl.UseContext("diffusion", "timestep"),
            RangeEncoder(320, 1280, device=device, dtype=dtype),
            fl.SetContext("range_adapter", context_key),
        )


class CLIPLCrossAttention(CrossAttentionBlock2d):
    def __init__(
        self,
        channels: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            channels=channels,
            context_embedding_dim=768,
            context_key="clip_text_embedding",
            num_attention_heads=8,
            use_bias=False,
            device=device,
            dtype=dtype,
        )


class DownBlocks(fl.Chain):
    def __init__(
        self,
        in_channels: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        self.in_channels = in_channels
        super().__init__(
            fl.Chain(
                fl.Conv2d(
                    in_channels=in_channels, out_channels=320, kernel_size=3, padding=1, device=device, dtype=dtype
                )
            ),
            fl.Chain(
                ResidualBlock(in_channels=320, out_channels=320, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=320, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=320, out_channels=320, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=320, device=device, dtype=dtype),
            ),
            fl.Chain(fl.Downsample(channels=320, scale_factor=2, padding=1, device=device, dtype=dtype)),
            fl.Chain(
                ResidualBlock(in_channels=320, out_channels=640, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=640, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=640, out_channels=640, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=640, device=device, dtype=dtype),
            ),
            fl.Chain(fl.Downsample(channels=640, scale_factor=2, padding=1, device=device, dtype=dtype)),
            fl.Chain(
                ResidualBlock(in_channels=640, out_channels=1280, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=1280, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=1280, out_channels=1280, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=1280, device=device, dtype=dtype),
            ),
            fl.Chain(fl.Downsample(channels=1280, scale_factor=2, padding=1, device=device, dtype=dtype)),
            fl.Chain(
                ResidualBlock(in_channels=1280, out_channels=1280, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=1280, out_channels=1280, device=device, dtype=dtype),
            ),
        )


class UpBlocks(fl.Chain):
    def __init__(
        self,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            fl.Chain(
                ResidualBlock(in_channels=2560, out_channels=1280, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=2560, out_channels=1280, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=2560, out_channels=1280, device=device, dtype=dtype),
                fl.Upsample(channels=1280, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=2560, out_channels=1280, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=1280, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=2560, out_channels=1280, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=1280, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=1920, out_channels=1280, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=1280, device=device, dtype=dtype),
                fl.Upsample(channels=1280, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=1920, out_channels=640, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=640, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=1280, out_channels=640, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=640, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=960, out_channels=640, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=640, device=device, dtype=dtype),
                fl.Upsample(channels=640, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=960, out_channels=320, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=320, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=640, out_channels=320, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=320, device=device, dtype=dtype),
            ),
            fl.Chain(
                ResidualBlock(in_channels=640, out_channels=320, device=device, dtype=dtype),
                CLIPLCrossAttention(channels=320, device=device, dtype=dtype),
            ),
        )


class MiddleBlock(fl.Chain):
    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        super().__init__(
            ResidualBlock(in_channels=1280, out_channels=1280, device=device, dtype=dtype),
            CLIPLCrossAttention(channels=1280, device=device, dtype=dtype),
            ResidualBlock(in_channels=1280, out_channels=1280, device=device, dtype=dtype),
        )


class SD1UNet(fl.Chain):
    """Stable Diffusion 1.5 U-Net.

    See [[arXiv:2112.10752] High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) for more details."""

    def __init__(
        self,
        in_channels: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize the U-Net.

        Args:
            in_channels: The number of input channels.
            device: The PyTorch device to use for computation.
            dtype: The PyTorch dtype to use for computation.
        """
        self.in_channels = in_channels
        super().__init__(
            TimestepEncoder(device=device, dtype=dtype),
            DownBlocks(in_channels=in_channels, device=device, dtype=dtype),
            fl.Sum(
                fl.UseContext(context="unet", key="residuals").compose(lambda x: x[-1]),
                MiddleBlock(device=device, dtype=dtype),
            ),
            UpBlocks(),
            fl.Chain(
                fl.GroupNorm(channels=320, num_groups=32, device=device, dtype=dtype),
                fl.SiLU(),
                fl.Conv2d(
                    in_channels=320,
                    out_channels=4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )
        for residual_block in self.layers(ResidualBlock):
            chain = residual_block.layer("Chain", fl.Chain)
            RangeAdapter2d(
                target=chain.layer("Conv2d_1", fl.Conv2d),
                channels=residual_block.out_channels,
                embedding_dim=1280,
                context_key="timestep_embedding",
                device=device,
                dtype=dtype,
            ).inject(chain)
        for n, block in enumerate(cast(Iterable[fl.Chain], self.DownBlocks)):
            block.append(ResidualAccumulator(n))
        for n, block in enumerate(cast(Iterable[fl.Chain], self.UpBlocks)):
            block.insert(0, ResidualConcatenator(-n - 2))

    def init_context(self) -> Contexts:
        return {
            "unet": {"residuals": [0.0] * 13},
            "diffusion": {"timestep": None},
            "range_adapter": {"timestep_embedding": None},
            "sampling": {"shapes": []},
        }

    def set_clip_text_embedding(self, clip_text_embedding: Tensor) -> None:
        """Set the CLIP text embedding.

        Note:
            This context is required by the `CLIPLCrossAttention` blocks.

        Args:
            clip_text_embedding: The CLIP text embedding.
        """
        self.set_context("cross_attention_block", {"clip_text_embedding": clip_text_embedding})

    def set_timestep(self, timestep: Tensor) -> None:
        """Set the timestep.

        Note:
            This context is required by `TimestepEncoder`.

        Args:
            timestep: The timestep.
        """
        self.set_context("diffusion", {"timestep": timestep})
