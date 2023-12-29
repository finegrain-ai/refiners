from typing import TYPE_CHECKING, Any, Generic, TypeVar

import torch
import torch.nn.functional as F

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.adapter import Adapter
from refiners.fluxion.layers.activations import Activation
from refiners.fluxion.layers.attentions import ScaledDotProductAttention

if TYPE_CHECKING:
    from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
    from refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet

T = TypeVar("T", bound="SD1UNet | SDXLUNet")
TPaletteAdapter = TypeVar("TPaletteAdapter", bound="PaletteAdapter[Any]")  # Self (see PEP 673)


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
        pad = [0] * 2 * x.ndim
        dim_index = 1 + 2 * (x.ndim - 1) - 2 * self.dim
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


class ColorEncoder(fl.Passthrough):
    def __init__(
        self,
        dim_embeddings: int,
        dim_model: int,
        max_colors: int = 8,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.dim_embeddings = dim_embeddings
        self.dim_model = dim_model
        self.max_colors = max_colors

        super().__init__(
            fl.UseContext("unet", "palette_colors"),
            # (n_colors: uint8, 3)
            fl.Concatenate(
                fl.Chain(
                    SinusoidalEmbedding(dim_embedding=dim_embeddings),
                    # (n_colors: float32, 3, dim_embeddings)
                    fl.Reshape(-1, dim_embeddings * 3),
                    # (n_colors, 3 * dim_embeddings)
                    FeedForward(
                        embedding_dim=dim_embeddings * 3,
                        feedforward_dim=dim_model,
                        device=device,
                        dtype=dtype,
                    ),  # TODO: changer l'output dim, là c'est la même que l'input dim
                    # (n_colors, dim_embeddings * 3)
                ),
                EOSToken(
                    embedding_dim=dim_embeddings * 3,
                    device=device,
                    dtype=dtype,
                ),  # (1, dim_embeddings * 3)
                dim=1,
            ),
            # (n_colors + 1, dim_embeddings * 3)
            FillZeroRightPad(
                max_length=max_colors + 1,
                dim=1,
            ),
            # (max_colors, dim_embeddings * 3)
            fl.SetContext("unet", "palette_embeddings"),
        )


class PaletteCrossAttention(fl.Chain):
    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 8,
        scale: float = 1.0,
    ) -> None:
        self.dim = dim
        self.num_heads = num_heads
        self.scale = scale

        super().__init__(
            fl.Distribute(
                fl.Identity(),  # Q
                fl.Chain(  # K
                    fl.UseContext("unet", "palette_embedding"),
                    fl.Linear(dim, dim),
                ),
                fl.Chain(  # V
                    fl.UseContext("unet", "palette_embedding"),
                    fl.Linear(dim, dim),
                ),
            ),
            ScaledDotProductAttention(num_heads=num_heads),
            fl.Lambda(func=self.scale_outputs),
        )

    def scale_outputs(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class CrossAttentionAdapter(fl.Chain, Adapter[fl.Attention]):
    def __init__(
        self,
        target: fl.Attention,
        scale: float = 1.0,
    ) -> None:
        self.scale = scale
        with self.setup_adapter(target):
            super().__init__(target)

    def inject(self, parent: fl.Chain | None = None) -> "CrossAttentionAdapter":
        # find the ScaledDotProductAttention
        scaled_dot_product_attention = self.target.ensure_find(ScaledDotProductAttention)
        # replace the ScaledDotProductAttention
        self.target.replace(
            old_module=scaled_dot_product_attention,
            new_module=fl.Sum(
                scaled_dot_product_attention,
                PaletteCrossAttention(
                    dim=self.target.embedding_dim,
                    num_heads=self.target.num_heads,
                    scale=self.scale,
                ),
            ),
        )
        return super().inject(parent)

    def eject(self) -> None:
        # find the PaletteCrossAttention
        palette_cross_attention = self.target.ensure_find(PaletteCrossAttention)
        # find it's parent
        parent = self.target.ensure_find_parent(palette_cross_attention)
        # remove the PaletteCrossAttention
        self.target.replace(
            old_module=parent,
            new_module=parent[0],
        )
        super().eject()


class PaletteAdapter(Generic[T], fl.Chain, Adapter[T]):
    def __init__(
        self,
        target: T,
        color_encoder: ColorEncoder,
    ) -> None:
        with self.setup_adapter(target):
            super().__init__(
                color_encoder,
                target,
            )

        self.sub_adapters = [
            CrossAttentionAdapter(target=cross_attn)
            for cross_attn in filter(lambda attn: type(attn) != fl.SelfAttention, target.layers(fl.Attention))
        ]

    def inject(self: "TPaletteAdapter", parent: fl.Chain | None = None) -> "TPaletteAdapter":
        for adapter in self.sub_adapters:
            adapter.inject()
        return super().inject(parent)

    def eject(self) -> None:
        for adapter in self.sub_adapters:
            adapter.eject()
        super().eject()
