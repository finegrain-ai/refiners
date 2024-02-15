import math

import torch

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.adapter import Adapter
from refiners.fluxion.context import Contexts
from refiners.foundationals.latent_diffusion.range_adapter import RangeEncoder
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet


def compute_sinusoidal_embedding(
    x: torch.Tensor,
    embedding_dim: int,
) -> torch.Tensor:
    # Differences from compute_sinusoidal_embedding in RangeAdapter:
    # - we concat [sin, cos], it does the opposite ([cos, sin])
    # - we divide the exponent by half_dim - 1, it divides by half_dim

    half_dim = embedding_dim // 2
    # Note: it is important that this computation is done in float32.
    # The result can be cast to lower precision later if necessary.
    exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=x.device)
    exponent /= half_dim - 1
    embedding = x.unsqueeze(1).float() * torch.exp(exponent).unsqueeze(0)
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)

    assert embedding.shape == (x.shape[0], embedding_dim)
    return embedding


class ResidualBlock(fl.Residual):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            fl.UseContext("lcm", "condition_scale_embedding"),
            fl.Converter(),
            fl.Linear(in_features=in_channels, out_features=out_channels, bias=False, device=device, dtype=dtype),
        )


class LcmAdapter(fl.Chain, Adapter[SDXLUNet]):
    def __init__(
        self,
        target: SDXLUNet,
        condition_scale_embedding_dim: int = 256,
        condition_scale: float = 7.5,
    ) -> None:
        assert condition_scale_embedding_dim % 2 == 0
        self.condition_scale_embedding_dim = condition_scale_embedding_dim
        self.condition_scale = condition_scale
        with self.setup_adapter(target):
            super().__init__(target)

    def init_context(self) -> Contexts:
        return {"lcm": {"condition_scale_embedding": self.sinusoidal_embedding}}

    @property
    def sinusoidal_embedding(self) -> torch.Tensor:
        return compute_sinusoidal_embedding(
            torch.tensor([(self.condition_scale - 1) * 1000], device=self.device),
            embedding_dim=self.condition_scale_embedding_dim,
        )

    def set_condition_scale(self, scale: float) -> None:
        self.condition_scale = scale
        self.set_context("lcm", {"condition_scale_embedding": self.sinusoidal_embedding})

    def inject(self: "LcmAdapter", parent: fl.Chain | None = None) -> "LcmAdapter":
        ra = self.target.ensure_find(RangeEncoder)
        block = ResidualBlock(
            in_channels=self.condition_scale_embedding_dim,
            out_channels=ra.sinusoidal_embedding_dim,
            device=self.target.device,
            dtype=self.target.dtype,
        )
        ra.insert_before_type(fl.Linear, block)
        return super().inject(parent)

    def eject(self) -> None:
        ra = self.target.ensure_find(RangeEncoder)
        ra.remove(ra.ensure_find(ResidualBlock))
        super().eject()
