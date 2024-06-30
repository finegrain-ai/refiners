from typing import TYPE_CHECKING, Any, Generic, Iterable, TypeVar

import torch.nn as nn
from torch import Tensor, cat, device as Device, dtype as DType

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.adapter import Adapter
from refiners.fluxion.context import Contexts
from refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock
from refiners.foundationals.latent_diffusion.range_adapter import RangeEncoder

if TYPE_CHECKING:
    from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
    from refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet

T = TypeVar("T", bound="SD1UNet | SDXLUNet")
TELLAAdapter = TypeVar("TELLAAdapter", bound="ELLAAdapter[Any]")


class LayerNormNoAffine(nn.LayerNorm, fl.Module):
    def __init__(
        self,
        normalized_shape: int | Iterable[int],
        eps: float = 1e-5,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(normalized_shape, eps=eps, elementwise_affine=False, device=device, dtype=dtype)  # type: ignore


class TimestepEncoder(fl.Passthrough):
    def __init__(
        self,
        time_embedding_dim: int,
        time_channel: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            fl.UseContext("diffusion", "timestep"),
            RangeEncoder(time_channel, time_embedding_dim, device=device, dtype=dtype),
            fl.SetContext("ella", "timestep_embedding"),
        )


class SquaredReLU(fl.ReLU):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x).pow(2)


class AdaLayerNorm(fl.Chain):
    def __init__(
        self,
        embedding_dim: int,
        time_embedding_dim: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            fl.Parallel(
                LayerNormNoAffine(embedding_dim, eps=1e-6, device=device, dtype=dtype),
                fl.Chain(
                    fl.UseContext("ella", "timestep_embedding"),
                    fl.SiLU(),
                    fl.Linear(time_embedding_dim, embedding_dim * 2, device=device, dtype=dtype),
                ),
            ),
            fl.Lambda(self._scale_shift_tensors),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        _linear: fl.Linear = self.ensure_find(fl.Linear)
        nn.init.zeros_(_linear.weight)
        nn.init.zeros_(_linear.bias)

    def _scale_shift_tensors(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        shift, scale = time_embedding.chunk(2, dim=-1)
        return x * (1 + scale) + shift


class ParameterInitialized(fl.Parameter):
    def __init__(
        self, *dims: int, requires_grad: bool = True, device: Device | str | None = None, dtype: DType | None = None
    ) -> None:
        super().__init__(*dims, requires_grad=requires_grad, device=device, dtype=dtype)
        nn.init.normal_(self.weight, mean=0, std=dims[1] ** 0.5)


class Latents(fl.Chain):
    def __init__(
        self,
        num_latents: int,
        width: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            ParameterInitialized(
                num_latents,
                width,
                device=device,
                dtype=dtype,
            ),
        )


class PerceiverAttention(fl.Chain):
    def __init__(
        self,
        width: int,
        num_heads: int,
        timestep_embedding_dim: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            fl.Distribute(
                AdaLayerNorm(width, timestep_embedding_dim, device=device, dtype=dtype),
                AdaLayerNorm(width, timestep_embedding_dim, device=device, dtype=dtype),
            ),
            fl.Parallel(
                fl.GetArg(index=1),
                fl.Lambda(func=self.to_kv),
                fl.Lambda(func=self.to_kv),
            ),
            fl.Attention(embedding_dim=width, num_heads=num_heads, device=device, dtype=dtype),
        )

    def to_kv(self, x: Tensor, latents: Tensor) -> Tensor:
        return cat((latents, x), dim=-2)


class OutputProjection(fl.Chain):
    def __init__(
        self, width: int, output_dim: int, device: Device | str | None = None, dtype: DType | None = None
    ) -> None:
        super().__init__(
            fl.Linear(width, output_dim, device=device, dtype=dtype),
            fl.LayerNorm(output_dim, device=device, dtype=dtype),
        )


class Transformer(fl.Chain):
    pass


class TransformerLayer(fl.Chain):
    pass


class FeedForward(fl.Chain):
    def __init__(
        self,
        width: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            fl.Linear(width, width * 4, device=device, dtype=dtype),
            SquaredReLU(),
            fl.Linear(width * 4, width, device=device, dtype=dtype),
        )


class PerceiverResampler(fl.Chain):
    def __init__(
        self,
        time_embedding_dim: int,
        width: int,
        num_layers: int,
        num_heads: int,
        num_latents: int,
        output_dim: int | None,
        input_dim: int | None,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            fl.Linear(input_dim, width, device=device, dtype=dtype) if input_dim else fl.Identity(),
            fl.SetContext("perceiver_resampler", "x"),
            Latents(num_latents, width, device=device, dtype=dtype),
            fl.Residual(
                fl.UseContext("ella", "timestep_embedding"),
                fl.SiLU(),
                fl.Linear(time_embedding_dim, width, device=device, dtype=dtype),
            ),
            Transformer(
                TransformerLayer(
                    fl.Residual(
                        fl.Parallel(fl.UseContext(context="perceiver_resampler", key="x"), fl.Identity()),
                        PerceiverAttention(width, num_heads, time_embedding_dim, device=device, dtype=dtype),
                    ),
                    fl.Residual(
                        AdaLayerNorm(width, time_embedding_dim, device=device, dtype=dtype),
                        FeedForward(width, device=device, dtype=dtype),
                    ),
                )
                for _ in range(num_layers)
            ),
            OutputProjection(width, output_dim, device=device, dtype=dtype) if output_dim else fl.Identity(),
        )

    def init_context(self) -> Contexts:
        return {"perceiver_resampler": {"x": None}}


class ELLA(fl.Passthrough):
    def __init__(
        self,
        time_channel: int,
        timestep_embedding_dim: int,
        width: int,
        num_layers: int,
        num_heads: int,
        num_latents: int,
        input_dim: int | None = None,
        out_dim: int | None = None,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            TimestepEncoder(timestep_embedding_dim, time_channel, device=device, dtype=dtype),
            fl.UseContext("adapted_cross_attention_block", "llm_text_embedding"),
            PerceiverResampler(
                timestep_embedding_dim,
                width,
                num_layers,
                num_heads,
                num_latents,
                out_dim,
                input_dim,
                device=device,
                dtype=dtype,
            ),
            fl.SetContext("ella", "latents"),
        )


class ELLACrossAttentionAdapter(fl.Chain, Adapter[fl.UseContext]):
    # TODO: concatenate the latents with the clip text embedding  https://github.com/TencentQQGYLab/ELLA/tree/main?tab=readme-ov-file#3-ellaclip-for-community-models
    def __init__(self, target: fl.UseContext) -> None:
        with self.setup_adapter(target):
            super().__init__(fl.UseContext("ella", "latents"))


class ELLAAdapter(Generic[T], fl.Chain, Adapter[T]):
    def __init__(self, target: T, latents_encoder: ELLA, weights: dict[str, Tensor] | None = None) -> None:
        if weights is not None:
            latents_encoder.load_state_dict(weights)

        self._latents_encoder = [latents_encoder]
        with self.setup_adapter(target):
            super().__init__(target)
        self.sub_adapters = [
            ELLACrossAttentionAdapter(use_context)
            for cross_attn in target.layers(CrossAttentionBlock)
            for use_context in cross_attn.layers(fl.UseContext)
        ]

    def inject(self: TELLAAdapter, parent: fl.Chain | None = None) -> TELLAAdapter:
        for adapter in self.sub_adapters:
            adapter.inject()
        self.target.insert(0, self.latents_encoder)
        return super().inject(parent)

    def eject(self) -> None:
        for adapter in self.sub_adapters:
            adapter.eject()
        self.target.pop(0)
        super().eject()

    @property
    def latents_encoder(self) -> ELLA:
        return self._latents_encoder[0]

    def set_llm_text_embedding(self, text_embedding: Tensor) -> None:
        self.set_context("adapted_cross_attention_block", {"llm_text_embedding": text_embedding})

    def init_context(self) -> Contexts:
        return {"ella": {"timestep_embedding": None, "latents": None}}
