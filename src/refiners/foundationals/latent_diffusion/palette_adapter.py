from typing import TYPE_CHECKING, Generic, TypeVar

import torch
import torch.nn.functional as F

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.adapter import Adapter
from refiners.fluxion.layers.activations import Activation

if TYPE_CHECKING:
    from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
    from refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet

T = TypeVar("T", bound="SD1UNet | SDXLUNet")


class FillZeroRightPad(fl.Module):
    """Pad a tensor with zeros on the right to reach a given length."""

    # TODO: generalizable ?

    def __init__(
        self,
        max_length: int,
        dim: int,
    ) -> None:
        self.max_length = max_length
        self.dim = dim
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        dim_index = 1 + 2 * (x.ndim - 1) - 2 * self.dim
        pad = [0] * 2 * x.ndim
        pad[dim_index] = self.max_length - x.size(self.dim)
        return F.pad(x, pad, "constant", 0)


class EOSToken(fl.Chain):
    """Learnable token representing the class of the input."""

    def __init__(
        self,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim

        super().__init__(
            fl.Parameter(
                *(1, embedding_dim),
                device=device,
                dtype=dtype,
            ),
        )


class FeedForward(fl.Chain):
    """Apply two linear transformations interleaved by an activation function."""

    def __init__(
        self,
        embedding_dim: int,
        feedforward_dim: int,
        activation: Activation = fl.GeLU,  # type: ignore
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.feedforward_dim = feedforward_dim

        super().__init__(
            fl.Linear(
                in_features=embedding_dim,
                out_features=feedforward_dim,
                device=device,
                dtype=dtype,
            ),
            activation(),
            fl.Linear(
                in_features=feedforward_dim,
                out_features=embedding_dim,
                device=device,
                dtype=dtype,
            ),
        )


class SinusoidalEmbedding(fl.Module):
    """Compute sinusoidal embeddings for a given sequence of indices."""

    def __init__(
        self,
        dim_embedding: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.dim_embedding = dim_embedding
        super().__init__()

    def compute_sinusoidal_embedding(
        self,
        x: torch.Tensor,
        dim_embedding: int,
        period: int = 10_000,
    ):
        """Compute sinusoidal embeddings for a given sequence of indices.

        f(x, k) = sin( ωₖ * x ), if k is even
        f(x, k) = cos( ωₖ * x ), if k is odd
        where ωₖ = 1 / period ** (2 * k / dim_embedding)
        and k = 0, ..., dim_embedding // 2 - 1
        """
        half_dim = dim_embedding // 2

        omega = -torch.tensor(period).log()
        omega = omega * torch.arange(
            start=0,
            end=half_dim,
            dtype=torch.float32,
            device=x.device,
        )
        omega = omega / half_dim
        omega = omega.exp()

        theta = x.unsqueeze(-1) * omega.unsqueeze(0)
        embedding = torch.cat(
            [
                torch.cos(theta),
                torch.sin(theta),
            ],
            dim=-1,
        )

        return embedding

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.compute_sinusoidal_embedding(
            x=x,
            dim_embedding=self.dim_embedding,
        )


class ColorEncoder(fl.Chain):
    def __init__(
        self,
        dim_embeddings: int,
        dim_model: int,
        max_colors: int = 8,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            fl.Concatenate(
                fl.Chain(  # TODO: could be factored in a class ?
                    # (n_colors, 3)
                    SinusoidalEmbedding(
                        dim_embedding=dim_embeddings,
                        device=device,
                        dtype=dtype,
                    ),
                    # (n_colors, 3, dim_embeddings)
                    fl.Reshape(-1, dim_embeddings * 3),
                    # (n_colors, 3 * dim_embeddings)
                    FeedForward(
                        embedding_dim=dim_embeddings * 3,
                        feedforward_dim=dim_model,
                        device=device,
                        dtype=dtype,
                    ),
                    # (n_colors, dim_model)
                ),
                EOSToken(
                    embedding_dim=dim_embeddings * 3,
                    device=device,
                    dtype=dtype,
                ),
                dim=1,
            ),
            # (n_colors + 1, dim_model)
            FillZeroRightPad(
                max_length=max_colors,
                dim=1,
            ),
            # (max_colors, dim_model)
        )


# from refiners.foundationals.latent_diffusion.palette_adapter import *

# enc = ColorEncoder(64, 32)
# x = torch.randint(0, 255, (5, 3), dtype=torch.uint8).unsqueeze(0)
# enc(x)


class PaletteAdapter(Generic[T], fl.Chain, Adapter[T]):
    pass
