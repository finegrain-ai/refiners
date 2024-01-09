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
TPaletteAdapter = TypeVar("TPaletteAdapter", bound="PaletteAdapter[Any]")


class FillZeroRightPad(fl.Module):  # TODO: generalizable ?
    """Pad a tensor with zeros on the right to reach a given length."""

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
        input_dim: int,
        intermediate_dim: int,
        output_dim: int,
        activation: Activation = fl.GeLU,  # type: ignore
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim

        super().__init__(
            fl.Linear(
                in_features=input_dim,
                out_features=intermediate_dim,
                device=device,
                dtype=dtype,
            ),
            activation(),
            fl.Linear(
                in_features=intermediate_dim,
                out_features=output_dim,
                device=device,
                dtype=dtype,
            ),
        )


class SinusoidalEmbedding(fl.Module):
    """Compute sinusoidal embeddings for a given sequence of indices.

    See https://www.tensorflow.org/text/tutorials/transformer?hl=en#the_embedding_and_positional_encoding_layer
    """

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

        f(x, k) = cos( ωₖ * x ), if k < half_dim
        f(x, k) = sin( ωₖ * x ), else

        where ωₖ = 1 / period ** (k / half_dim)
        and half_dim = dim_embedding // 2
        and k = 0, ..., half_dim - 1
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
    """Encode a sequence of (RGB) colors into a sequence of embeddings."""

    def __init__(
        self,
        dim_sinusoids: int,
        dim_embeddings: int,
        max_colors: int = 8,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.dim_sinusoids = dim_sinusoids
        self.dim_embeddings = dim_embeddings
        self.max_colors = max_colors

        super().__init__(
            fl.UseContext("palette", "colors"),
            # (batch, n_colors, 3): uint8
            fl.Concatenate(
                fl.Chain(
                    SinusoidalEmbedding(dim_embedding=dim_sinusoids),
                    # (batch, n_colors, 3, dim_sinusoids): float32
                    fl.Reshape(-1, dim_sinusoids * 3),
                    # (batch, n_colors, 3 * dim_sinusoids)
                    FeedForward(
                        input_dim=dim_sinusoids * 3,
                        intermediate_dim=dim_sinusoids * 3,
                        output_dim=dim_embeddings,
                        device=device,
                        dtype=dtype,
                    ),
                    # (batch, n_colors, dim_embeddings)
                ),
                EOSToken(
                    embedding_dim=dim_embeddings,
                    device=device,
                    dtype=dtype,
                ),  # (batch, 1, dim_embeddings)
                dim=1,
            ),
            # (batch, n_colors + 1, dim_embeddings)
            FillZeroRightPad(
                max_length=max_colors + 1,
                dim=1,
            ),
            # (batch, max_colors + 1, dim_embeddings)
            fl.SetContext("palette", "embeddings"),
        )


class PaletteCrossAttention(fl.Chain):
    """Cross attention module, using the palette embeddings for keys and values."""

    def __init__(
        self,
        linear_input_dim: int,
        linear_output_dim: int,
        num_heads: int,
        scale: float = 1.0,
    ) -> None:
        self.linear_input_dim = linear_input_dim
        self.linear_output_dim = linear_output_dim
        self.num_heads = num_heads
        self.scale = scale

        super().__init__(
            fl.Distribute(
                fl.Identity(),  # Q (from the original CrossAttention)
                fl.Chain(  # K (palette)
                    fl.UseContext("palette", "embeddings"),
                    fl.Linear(linear_input_dim, linear_output_dim),
                ),
                fl.Chain(  # V (palette)
                    fl.UseContext("palette", "embeddings"),
                    fl.Linear(linear_input_dim, linear_output_dim),
                ),
            ),
            ScaledDotProductAttention(num_heads=num_heads),
            fl.Multiply(scale=scale),
        )


class CrossAttentionAdapter(fl.Chain, Adapter[fl.Attention]):
    """Inject a PaletteCrossAttention module in a given Attention module."""

    def __init__(
        self,
        target: fl.Attention,
        palette_embeddings_dim: int,
        scale: float = 1.0,
    ) -> None:
        self.scale = scale
        self.palette_embeddings_dim = palette_embeddings_dim
        with self.setup_adapter(target):
            super().__init__(target)

    def inject(self, parent: fl.Chain | None = None) -> "CrossAttentionAdapter":
        # find the ScaledDotProductAttention
        scaled_dot_product_attention = self.target.ensure_find(ScaledDotProductAttention)
        # replace it by a Sum of the ScaledDotProductAttention and the PaletteCrossAttention
        self.target.replace(
            old_module=scaled_dot_product_attention,
            new_module=fl.Sum(
                scaled_dot_product_attention,
                PaletteCrossAttention(
                    linear_input_dim=self.palette_embeddings_dim,
                    linear_output_dim=self.target.embedding_dim,
                    num_heads=self.target.num_heads,
                    scale=self.scale,
                ),
            ),
        )
        return super().inject(parent)

    def eject(self) -> None:
        # find the PaletteCrossAttention
        palette_cross_attention = self.target.ensure_find(PaletteCrossAttention)
        # find it's parent (Sum)
        parent = self.target.ensure_find_parent(palette_cross_attention)
        # extract the original ScaledDotProductAttention out of the Sum
        self.target.replace(
            old_module=parent,
            new_module=parent[0],
        )
        super().eject()


class PaletteAdapter(Generic[T], fl.Chain, Adapter[T]):
    """Inject a ColorEncoder and PaletteCrossAttention modules in a given UNet module."""

    # TODO: adding a small LoRA to the Attention's projection layer could be worth trying

    def __init__(
        self,
        target: T,
        color_encoder: ColorEncoder,
    ) -> None:
        with self.setup_adapter(target):
            super().__init__(target)

        self.color_encoder = [color_encoder]
        self.cross_attention_adapters = [
            CrossAttentionAdapter(target=cross_attn, palette_embeddings_dim=color_encoder.dim_embeddings)
            for cross_attn in filter(lambda attn: type(attn) != fl.SelfAttention, target.layers(fl.Attention))
        ]

    def inject(self: "TPaletteAdapter", parent: fl.Chain | None = None) -> "TPaletteAdapter":
        # prepend the color encoder
        self.target.insert(0, self.color_encoder[0])
        # inject all the cross attention adapters
        for adapter in self.cross_attention_adapters:
            adapter.inject()
        return super().inject(parent)

    def eject(self) -> None:
        # remove the color encoder
        color_encoder = self.target.ensure_find(ColorEncoder)
        self.target.remove(color_encoder)
        # eject all the cross attention adapters
        for adapter in self.cross_attention_adapters:
            adapter.eject()
        super().eject()
