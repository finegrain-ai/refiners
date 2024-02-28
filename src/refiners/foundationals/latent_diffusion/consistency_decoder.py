from typing import Callable, Iterable, Tuple, cast

import torch
import torch.nn.functional as F
from torch import Tensor, device as Device, dtype as DType
from torch.nn import AvgPool2d as _AvgPool2d
from refiners.foundationals.latent_diffusion.solvers.solver import SolverParams
from refiners.fluxion.context import Contexts
from refiners.fluxion.layers import (
    Chain,
    Conv2d,
    Embedding,
    GroupNorm,
    Identity,
    Lambda,
    Linear,
    Passthrough,
    SetContext,
    SiLU,
    Sum,
    UseContext,
)
from refiners.fluxion.layers.sampling import Upsample
from refiners.fluxion.layers.module import Module
from refiners.foundationals.latent_diffusion.range_adapter import AffineRangeAdapter2d
from refiners.foundationals.latent_diffusion.solvers.consistency_decoder_solver import ConsistencyDecoderSolver
from refiners.foundationals.latent_diffusion.unet import ResidualAccumulator, ResidualConcatenator


class AvgPool2d(_AvgPool2d, Module):
    def __init__(self, kernel_size: Tuple[int, int] = (2, 2)):
        super().__init__(kernel_size=kernel_size)


# the model does not rely on the sin embeddings provided in range_adapters, so new class is required
class TimestepEmbedding(Passthrough):
    def __init__(self, n_time: int = 1024, n_emb: int = 320, n_out: int = 1280):
        super().__init__(
            UseContext("diffusion", "timestep"),
            Embedding(n_time, n_emb),
            Linear(n_emb, n_out),
            SiLU(),
            Linear(n_out, n_out),
            SetContext("range_adapter", "timestep_embedding"),
        )


class ImageEmbedding(Conv2d):
    def __init__(self, in_channels: int = 7, out_channels: int = 320) -> None:
        super().__init__(in_channels, out_channels, kernel_size=3, padding=1)


class ImageUnembedding(Chain):
    def __init__(self, in_channels: int = 320, out_channels: int = 6) -> None:
        super().__init__(
            GroupNorm(in_channels, 32), SiLU(), Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )


class ConvResBlock(Sum):
    def __init__(self, in_features: int = 320, out_features: int = 320) -> None:
        long = Chain(
            GroupNorm(in_features, 32),
            SiLU(),
            Conv2d(in_features, out_features, kernel_size=3, padding=1),
            GroupNorm(out_features, 32),  # the output of this will be adapted with an AffineRangeAdapter
            SiLU(),
            Conv2d(out_features, out_features, kernel_size=3, padding=1),
        )

        skip_conv = in_features != out_features
        skip = Conv2d(in_features, out_features, kernel_size=1, padding=0) if skip_conv else Identity()
        super().__init__(long, skip)
        self.out_features = out_features


class DownsamplingBlock(Sum):
    def __init__(self, in_features: int = 320, out_features: int = 320) -> None:
        long = Chain(
            GroupNorm(in_features, 32),
            SiLU(),
            AvgPool2d(),
            Conv2d(in_features, out_features, kernel_size=3, padding=1),
            GroupNorm(out_features, 32),  # the output of this will be adapted with an AffineRangeAdapter
            SiLU(),
            Conv2d(out_features, out_features, kernel_size=3, padding=1),
        )

        func: Callable[[Tensor], Tensor] = lambda x: F.avg_pool2d(x, (2, 2))
        skip = Lambda(func)

        super().__init__(long, skip)

        self.out_features = out_features


class UpsamplingBlock(Sum):
    def __init__(self, in_features: int = 320, out_features: int = 320) -> None:
        upsample_func : Callable[[Tensor], Tensor] = lambda x: F.interpolate(x, scale_factor=(2, 2), mode='nearest')
        long = Chain(
            GroupNorm(in_features, 32),
            SiLU(),
            Lambda(upsample_func),
            Conv2d(in_features, out_features, kernel_size=3, padding=1),
            GroupNorm(out_features, 32),  # the output of this will be adapted with an AffineRangeAdapter
            SiLU(),
            Conv2d(out_features, out_features, kernel_size=3, padding=1),
        )

        skip = Lambda(upsample_func)

        super().__init__(long, skip)
        self.out_features = out_features


class DownBlocks(Chain):
    pass


class MidBlocks(Chain):
    pass


class UpBlocks(Chain):
    pass


class ConsistencyDecoder(Chain):
    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        embed_time = TimestepEmbedding()

        down_blocks = DownBlocks(
            ImageEmbedding(),
            ConvResBlock(320, 320),
            ConvResBlock(320, 320),
            ConvResBlock(320, 320),
            DownsamplingBlock(320, 320),
            ConvResBlock(320, 640),
            ConvResBlock(640, 640),
            ConvResBlock(640, 640),
            DownsamplingBlock(640, 640),
            ConvResBlock(640, 1024),
            ConvResBlock(1024, 1024),
            ConvResBlock(1024, 1024),
            DownsamplingBlock(1024, 1024),
            ConvResBlock(1024, 1024),
            ConvResBlock(1024, 1024),
            ConvResBlock(1024, 1024),
        )

        mid_blocks = MidBlocks(
            ConvResBlock(1024, 1024),
            ConvResBlock(1024, 1024),
        )

        up_blocks = UpBlocks(
            ConvResBlock(1024 * 2, 1024),
            ConvResBlock(1024 * 2, 1024),
            ConvResBlock(1024 * 2, 1024),
            ConvResBlock(1024 * 2, 1024),
            UpsamplingBlock(1024, 1024),
            ConvResBlock(1024 * 2, 1024),
            ConvResBlock(1024 * 2, 1024),
            ConvResBlock(1024 * 2, 1024),
            ConvResBlock(1024 + 640, 1024),
            UpsamplingBlock(1024, 1024),
            ConvResBlock(1024 + 640, 640),
            ConvResBlock(640 * 2, 640),
            ConvResBlock(640 * 2, 640),
            ConvResBlock(320 + 640, 640),
            UpsamplingBlock(640, 640),
            ConvResBlock(320 + 640, 320),
            ConvResBlock(320 * 2, 320),
            ConvResBlock(320 * 2, 320),
            ConvResBlock(320 * 2, 320),
        )

        residual_down_blocks = DownBlocks()
        num_shortcuts: int = 0
        for n, block in enumerate(cast(Iterable[Chain], down_blocks)):
            residual_down_blocks.append(block)
            residual_down_blocks.append(ResidualAccumulator(n))
            num_shortcuts = n + 1

        residual_up_blocks = UpBlocks()
        reverse_count = num_shortcuts
        for n, block in enumerate(cast(Iterable[Chain], up_blocks)):
            if isinstance(block, ConvResBlock):
                residual_up_blocks.append(ResidualConcatenator(num_shortcuts - reverse_count - 1))
                reverse_count += 1
            residual_up_blocks.append(block)

        super().__init__(embed_time, residual_down_blocks, mid_blocks, residual_up_blocks, ImageUnembedding())

        for layer_type in [ConvResBlock, DownsamplingBlock, UpsamplingBlock]:
            for block in self.layers(layer_type):
                target = block[0].layer("GroupNorm_2", GroupNorm)

                AffineRangeAdapter2d(
                    target=target,
                    channels=block.out_features,
                    embedding_dim=1280,
                    context_key="timestep_embedding",
                    device=device,
                    dtype=dtype,
                ).inject(block[0])

        self.register_buffer(
            "means",
            torch.tensor([0.38862467, 0.02253063, 0.07381133, -0.0171294])[None, :, None, None],
            persistent=False,
        )
        self.register_buffer(
            "stds", torch.tensor([0.9654121, 1.0440036, 0.76147926, 0.77022034])[None, :, None, None], persistent=False
        )
        self.scaling_factor = 1
        self.solver = [ConsistencyDecoderSolver(num_inference_steps=2, params=SolverParams(num_train_timesteps=1024))]

    def set_timestep(self, timestep: Tensor):
        self.set_context("diffusion", {"timestep": timestep})

    def init_context(self) -> Contexts:
        return {
            "unet": {"residuals": [0.0] * 16},
            "diffusion": {"timestep": None},
            "range_adapter": {"timestep_embedding": None},
            "sampling": {"shapes": []},
        }

    def decode(self, z: Tensor):
        z = (z * self.scaling_factor - self.means) / self.stds
        z = F.interpolate(z, mode="nearest", scale_factor=8)
        batch_size, _, height, width = z.shape

        x_t = self.solver[0].init_noise_sigma * torch.randn(batch_size, 3, height, width, device=self.device)

        for t in self.solver[0].timesteps:
            model_input = torch.concat([self.solver[0].scale_model_input(x_t, int(t)), z], dim=1)
            self.set_timestep(timestep=t.unsqueeze(0))
            model_output = self(model_input)[:, :3, :, :]
            prev_sample = self.solver[0](x_t, model_output, int(t))
            x_t = prev_sample
        x_0 = x_t

        x = (x_0 + 1) / 2
        return x
