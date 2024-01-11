from typing import Generic, TypeVar

import torch
from torch.nn.utils.rnn import pad_sequence

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.adapter import Adapter
from refiners.fluxion.layers.activations import Activation
from refiners.fluxion.layers.attentions import ScaledDotProductAttention
from refiners.foundationals.latent_diffusion import SD1UNet, SDXLUNet

T = TypeVar("T", bound="SD1UNet | SDXLUNet")


class EOSToken(fl.Module):
    """Learnable token representing the class of the input."""

    def __init__(
        self,
        embedding_dim: int,
        value: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.value = value
        self.device = device
        self.dtype = dtype
        super().__init__()

    def forward(self, _) -> torch.Tensor:
        return torch.full(
            size=(1, self.embedding_dim),
            fill_value=self.value,
            device=self.device,
            dtype=self.dtype,
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
        period: int = 10_000,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.dim_embedding = dim_embedding
        self.period = period
        self.dtype = dtype
        super().__init__()

    def compute_sinusoidal_embedding(
        self,
        x: torch.Tensor,
        dim_embedding: int,
        period: int,
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
            device=x.device,
            dtype=self.dtype or torch.float32,
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
            period=self.period,
        )


class ColorEncoder(fl.Chain):
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
            # (n_colors, 3)
            fl.Concatenate(
                fl.Identity(),
                EOSToken(
                    value=-256,  # FIXME: hardcoded value
                    embedding_dim=3,  # FIXME: hardcoded value
                    device=device,
                    dtype=dtype,
                ),  # (1, 3)
            ),
            # (n_colors + 1, 3)
            SinusoidalEmbedding(
                dim_embedding=dim_sinusoids,
                dtype=dtype,
            ),
            # (n_colors + 1, 3, dim_sinusoids)
            fl.Flatten(start_dim=1),
            # (n_colors + 1, 3 * dim_sinusoids)
            FeedForward(
                input_dim=dim_sinusoids * 3,
                intermediate_dim=dim_sinusoids * 3,
                output_dim=dim_embeddings,
                device=device,
                dtype=dtype,
            ),
            # (n_colors + 1, dim_embeddings)
        )


class PaletteCrossAttention(fl.Chain):
    """Cross attention module, using the palette embeddings for keys and values."""

    def __init__(
        self,
        linear_input_dim: int,
        linear_output_dim: int,
        num_heads: int,
        scale: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.linear_input_dim = linear_input_dim
        self.linear_output_dim = linear_output_dim
        self.num_heads = num_heads

        super().__init__(
            fl.Distribute(
                fl.Identity(),  # Q (from the original CrossAttention)
                fl.Chain(  # K (palette)
                    fl.UseContext("palette", "embeddings"),
                    fl.Linear(
                        in_features=linear_input_dim,
                        out_features=linear_output_dim,
                        device=device,
                        dtype=dtype,
                    ),
                ),
                fl.Chain(  # V (palette)
                    fl.UseContext("palette", "embeddings"),
                    fl.Linear(
                        in_features=linear_input_dim,
                        out_features=linear_output_dim,
                        device=device,
                        dtype=dtype,
                    ),
                ),
            ),
            ScaledDotProductAttention(num_heads=num_heads),
            fl.Multiply(scale=scale),
        )

    @property
    def scale(self) -> float:
        multiply = self.ensure_find(fl.Multiply)
        return multiply.scale

    @scale.setter
    def scale(self, value: float) -> None:
        multiply = self.ensure_find(fl.Multiply)
        multiply.scale = value


class CrossAttentionAdapter(fl.Chain, Adapter[fl.Attention]):
    """Inject a PaletteCrossAttention module in a given Attention module."""

    def __init__(
        self,
        target: fl.Attention,
        palette_embeddings_dim: int,
        scale: float = 1.0,
    ) -> None:
        with self.setup_adapter(target):
            super().__init__(target)

        self.palette_cross_attention = [
            PaletteCrossAttention(
                linear_input_dim=palette_embeddings_dim,
                linear_output_dim=target.embedding_dim,
                num_heads=target.num_heads,
                scale=scale,
                device=target.device,
                dtype=target.dtype,
            )
        ]

    def inject(self, parent: fl.Chain | None = None) -> "CrossAttentionAdapter":
        # find the ScaledDotProductAttention
        scaled_dot_product_attention = self.target.ensure_find(ScaledDotProductAttention)
        # replace it by a Sum of the ScaledDotProductAttention and the PaletteCrossAttention
        self.target.replace(
            old_module=scaled_dot_product_attention,
            new_module=fl.Sum(
                scaled_dot_product_attention,
                self.palette_cross_attention[0],
            ),
        )
        return super().inject(parent)

    def eject(self) -> None:
        # find the parent of the PaletteCrossAttention (fl.Sum)
        parent = self.target.ensure_find_parent(self.palette_cross_attention[0])
        # replace the fl.Sum by the original ScaledDotProductAttention
        self.target.replace(
            old_module=parent,
            new_module=parent.pop(0),
        )
        # also pop the PaletteCrossAttention, to unlink its parent
        parent.pop()
        super().eject()

    @property
    def scale(self) -> float:
        return self.palette_cross_attention[0].scale

    @scale.setter
    def scale(self, value: float) -> None:
        self.palette_cross_attention[0].scale = value


class PaletteAdapter(Generic[T], fl.Chain, Adapter[T]):
    """Inject a ColorEncoder and PaletteCrossAttention modules in a given UNet module."""

    # TODO: adding a small LoRA to the Attention's projection layer could be worth trying

    def __init__(
        self,
        target: T,
        color_encoder: ColorEncoder,
        scale: float = 1.0,
    ) -> None:
        with self.setup_adapter(target):
            super().__init__(target)

        self._color_encoder = [color_encoder]
        self.cross_attention_adapters = [
            CrossAttentionAdapter(
                target=cross_attn,
                palette_embeddings_dim=color_encoder.dim_embeddings,
                scale=scale,
            )
            for cross_attn in filter(lambda attn: type(attn) != fl.SelfAttention, target.layers(fl.Attention))
        ]

    @property
    def color_encoder(self) -> ColorEncoder:
        return self._color_encoder[0]

    def inject(self, parent: fl.Chain | None = None) -> "PaletteAdapter[T]":
        """Inject the PaletteCrossAttention modules."""
        for adapter in self.cross_attention_adapters:
            adapter.inject()
        return super().inject(parent)

    def eject(self) -> None:
        """Eject the PaletteCrossAttention modules."""
        for adapter in self.cross_attention_adapters:
            adapter.eject()
        super().eject()

    def compute_palette_embeddings(self, palettes: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """Compute the palette embeddings for a given batch of colors."""
        embeddings: list[torch.Tensor] = []
        for palette in palettes:
            assert (
                len(palette) <= self.color_encoder.max_colors
            ), f"All palettes must have less than max_colors={self.color_encoder.max_colors} colors."
            # TODO: add some other assertions
            embeddings.append(self.color_encoder(palette))
        return pad_sequence(embeddings, batch_first=True)

    def set_palette_embeddings(self, embeddings: torch.Tensor) -> None:
        """Set the palette embeddings."""
        self.set_context("palette", {"embeddings": embeddings})

    @property
    def scale(self) -> float:
        return self.cross_attention_adapters[0].scale

    @scale.setter
    def scale(self, value: float) -> None:
        for adapter in self.cross_attention_adapters:
            adapter.scale = value
