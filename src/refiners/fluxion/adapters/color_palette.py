from typing import Any, List, TypeVar

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor, arange, device as Device, dtype as DType, float32, tensor, zeros
from torch.nn.functional import pad
from torch.nn import init

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.adapter import Adapter
from refiners.foundationals.latent_diffusion.image_prompt import CrossAttentionAdapter
from refiners.foundationals.latent_diffusion.range_adapter import compute_sinusoidal_embedding
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet

TSDNet = TypeVar("TSDNet", bound="SD1UNet | SDXLUNet")
TColorPaletteAdapter = TypeVar("TColorPaletteAdapter", bound="SD1ColorPaletteAdapter[Any]")  # Self (see PEP 673)


class ColorPaletteEncoder(fl.Chain):
    def __init__(
        self,
        embedding_dim: int,
        max_colors: int,
        model_dim: int = 256,
        sinuosidal_embedding_dim: int = 32,
        device: Device | str | None = None,
        dtype: DType = float32,
        context_key: str = "color_palette_embedding",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.max_colors = max_colors

        super().__init__(
            fl.Linear(in_features=3, out_features=model_dim, device=device, dtype=dtype),
            fl.Residual(fl.Lambda(self.compute_sinuosoidal_embedding)),
            fl.Linear(in_features=model_dim, out_features=model_dim, device=device, dtype=dtype),
            fl.GeLU(),
            fl.Linear(in_features=model_dim, out_features=embedding_dim, device=device, dtype=dtype),
            fl.Lambda(self.end_of_sequence_token),
            fl.Lambda(self.zero_right_padding),
        )

    def compute_sinuosoidal_embedding(
        self, x: Int[Tensor, "*batch n_colors 3"]
    ) -> Float[Tensor, "*batch n_colors 3 model_dim"]:
        range = arange(start=0, end=x.shape[1], dtype=self.dtype, device=x.device).unsqueeze(1)
        embedding = compute_sinusoidal_embedding(range, embedding_dim=self.model_dim)
        return embedding.squeeze(1).unsqueeze(0).repeat(x.shape[0], 1, 1).to(dtype=self.dtype)

    def compute_color_palette_embedding(
        self,
        x: Int[Tensor, "*batch n_colors 3"] | List[List[List[Int]]],
        negative_color_palette: None | Int[Tensor, "*batch n_colors 3"] = None,
    ) -> Float[Tensor, "cfg_batch n_colors 3"]:
        tensor_x = tensor(x, device=self.device, dtype=self.dtype)
        conditional_embedding = self(tensor_x)
        if tensor_x == negative_color_palette:
            return torch.cat(tensors=(conditional_embedding, conditional_embedding), dim=0)

        if negative_color_palette is None:
            # a palette without any color in it
            negative_color_palette = zeros(tensor_x.shape[0], 0, 3, dtype=self.dtype, device=self.device)

        negative_embedding = self(negative_color_palette)
        return torch.cat(tensors=(negative_embedding, conditional_embedding), dim=0)

    def end_of_sequence_token(
        self, x: Float[Tensor, "*batch colors embedding_dim"]
    ) -> Float[Tensor, "*batch colors_with_end embedding_dim"]:
        # Build a tensor of size (batch_size, 1, embedding_dim) with the end of string token
        # end _of string token is a dim_model vector with 1 in the last position
        numpy_end_of_sequence_token = np.zeros((1, self.embedding_dim))
        numpy_end_of_sequence_token[-1] = 1

        end_of_sequence_tensor: Float[Tensor, "*batch 1 embedding_dim"] = (
            tensor(numpy_end_of_sequence_token, device=x.device, dtype=x.dtype)
            .reshape(1, 1, -1)
            .repeat(x.shape[0], 1, 1)
        )

        with_eos = torch.cat((x, end_of_sequence_tensor), dim=1)
        return with_eos[:, : self.max_colors, :]

    def zero_right_padding(
        self, x: Float[Tensor, "*batch colors_with_end embedding_dim"]
    ) -> Float[Tensor, "*batch max_colors model_dim"]:
        # Zero padding for the right side
        padding_width = (self.max_colors - x.shape[1] % self.max_colors) % self.max_colors

        result = pad(x, (0, 0, 0, padding_width))
        return result


class SD1ColorPaletteAdapter(fl.Chain, Adapter[TSDNet]):
    # Prevent PyTorch module registration
    _color_palette_encoder: list[ColorPaletteEncoder]

    def __init__(
        self,
        target: TSDNet,
        color_palette_encoder: ColorPaletteEncoder,
        scale: float = 1.0,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        with self.setup_adapter(target):
            super().__init__(target)

        self._color_palette_encoder = [color_palette_encoder]

        self.sub_adapters: list[CrossAttentionAdapter] = [
            CrossAttentionAdapter(target=cross_attn, scale=scale)
            for cross_attn in filter(lambda attn: type(attn) != fl.SelfAttention, target.layers(fl.Attention))
        ]
    
    @property
    def weights(self) -> List[Tensor]:
        weights = []
        for adapter in self.sub_adapters:
            weights += adapter.weights
        return weights
    
    def zero_init(self) -> None:
        weights = self.weights
        for weight in weights:
            init.zeros_(weight)
            
    def inject(self, parent: fl.Chain | None = None) -> "SD1ColorPaletteAdapter[Any]":
        for adapter in self.sub_adapters:
            adapter.inject()
        return super().inject(parent)

    def eject(self) -> None:
        for adapter in self.sub_adapters:
            adapter.eject()
        super().eject()

    def set_scale(self, scale: float) -> None:
        for cross_attn in self.sub_adapters:
            cross_attn.scale = scale

    def set_color_palette_embedding(self, color_palette_embedding: Tensor) -> None:
        self.set_context("ip_adapter", {"clip_image_embedding": color_palette_embedding})

    @property
    def color_palette_encoder(self) -> ColorPaletteEncoder:
        return self._color_palette_encoder[0]
