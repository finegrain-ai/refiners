from typing import Any, List, TypeVar

import torch
from jaxtyping import Float, Int
from torch import Tensor, device as Device, dtype as DType, tensor, zeros
from torch.nn.functional import pad
from torch.nn import init

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.adapter import Adapter
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet
from refiners.foundationals.clip.common import PositionalEncoder
from refiners.foundationals.clip.text_encoder import TransformerLayer
from refiners.fluxion.layers.attentions import ScaledDotProductAttention
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder

TSDNet = TypeVar("TSDNet", bound="SD1UNet | SDXLUNet")

class ColorsTokenizer(fl.Module):
    def __init__(
        self,
        max_colors: int = 8,
    ) -> None:
        super().__init__()
        self.max_colors = max_colors
    
    def forward(self, colors: Float[Tensor, "*batch colors 3"]) -> Float[Tensor, "*batch max_colors 4"]:        
        
        colors = colors/128.0 - 1.0 # normalize to [-1, 1]
        
        colors = self.add_channel(colors)
        colors = self.zero_right_padding(colors)
        return colors
    
    def add_channel(
        self, x: Float[Tensor, "*batch colors 4"]
    ) -> Float[Tensor, "*batch colors_with_end 5"]:
        return torch.cat((x, torch.ones(x.shape[0], x.shape[1], 1, dtype=x.dtype, device=x.device)), dim=2)

    def zero_right_padding(
        self, x: Float[Tensor, "*batch colors_with_end embedding_dim"]
    ) -> Float[Tensor, "*batch max_colors feedforward_dim"]:
        # Zero padding for the right side
        padding_width = (self.max_colors - x.shape[1] % self.max_colors) % self.max_colors
        if x.shape[1] == 0:
            padding_width = self.max_colors
        result = pad(x, (0, 0, 0, padding_width))
        return result


class ColorEncoder(fl.Chain):
    def __init__(
        self,
        embedding_dim: int,
        device: Device | str | None = None,
        eps: float = 1e-5,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            fl.Linear(in_features=4, out_features=embedding_dim, bias=True, device=device, dtype=dtype),
            fl.LayerNorm(
                normalized_shape=embedding_dim,
                eps=eps,
                device=device,
                dtype=dtype,
            )
        )

class ColorPaletteEncoder(fl.Chain):
    # _lda: list[SD1Autoencoder]
    
    @property
    def lda(self):
        return self._lda[0]
    
    def __init__(
        self,
        # lda: SD1Autoencoder,
        embedding_dim: int = 4,
        hidden_dim: int = 10,
        max_colors: int = 8,
        # Remark : 
        # I have followed the CLIPTextEncoderL parameters
        # as default parameters here, might require some testing
        num_layers: int = 12,
        num_attention_heads: int = 12,
        feedforward_dim: int = 3072,
        layer_norm_eps: float = 1e-5,
        use_quick_gelu: bool = False,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        # self._lda = [lda]
        self.embedding_dim = embedding_dim
        self.max_colors = max_colors
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.feedforward_dim = feedforward_dim
        self.layer_norm_eps = layer_norm_eps
        self.use_quick_gelu = use_quick_gelu
        self.out_sequence_size = 512
        super().__init__(
           # fl.Lambda(self.lda_encode),
            ColorsTokenizer(
                max_colors=max_colors
            ),
            fl.Linear(in_features=4, out_features=hidden_dim, bias=True, device=device, dtype=dtype),
            fl.GeLU(),
            fl.Linear(in_features=hidden_dim, out_features=embedding_dim, bias=True, device=device, dtype=dtype)
        )
        if use_quick_gelu:
            for gelu, parent in self.walk(predicate=lambda m, _: isinstance(m, fl.GeLU)):
                parent.replace(old_module=gelu, new_module=fl.ApproximateGeLU())
    def lda_encode(self, x: Float[Tensor, "*batch num_colors 3"]) -> Float[Tensor, "*batch num_colors 4"]:
        device = x.device
        dtype = x.dtype
        batch_size = x.shape[0]
        num_colors = x.shape[1]
        if num_colors == 0:
            return x.reshape(batch_size, 0, 4)
        
        x = x.reshape(batch_size * num_colors, 3, 1, 1)
        x = x.repeat(1, 1, 8, 8).to(self.lda.device, self.lda.dtype)
        
        out = self.lda.encode(x).to(device, dtype)
        
        out = out.reshape(batch_size, num_colors, 4)
        return out
    
    def compute_color_palette_embedding(
        self,
        x: Int[Tensor, "*batch n_colors 3"] | List[List[List[int]]],
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


class PaletteCrossAttention(fl.Chain):
    def __init__(self, text_cross_attention: fl.Attention, scale: float = 1.0) -> None:
        self._scale = scale
        super().__init__(
            fl.Distribute(
                fl.Identity(),
                fl.Chain(
                    fl.UseContext(context="ip_adapter", key="palette_embedding"),
                    fl.Linear(
                        in_features=4,
                        out_features=text_cross_attention.inner_dim,
                        bias=text_cross_attention.use_bias,
                        device=text_cross_attention.device,
                        dtype=text_cross_attention.dtype,
                    ),
                ),
                fl.Chain(
                    fl.UseContext(context="ip_adapter", key="palette_embedding"),
                    fl.Linear(
                        in_features=4,
                        out_features=text_cross_attention.inner_dim,
                        bias=text_cross_attention.use_bias,
                        device=text_cross_attention.device,
                        dtype=text_cross_attention.dtype,
                    ),
                ),
            ),
            ScaledDotProductAttention(
                num_heads=text_cross_attention.num_heads, is_causal=text_cross_attention.is_causal
            ),
            fl.Multiply(self.scale),
        )

    @property
    def scale(self) -> float:
        return self._scale
    
    def log_shapes(self, query: Tensor, key: Tensor, value: Tensor) -> None:
        print("query", query.shape)
        print("key", key.shape)
        print("value", value.shape)
        return query, key, value
        

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value
        self.ensure_find(fl.Multiply).scale = value

class PaletteCrossAttentionAdapter(fl.Chain, Adapter[fl.Attention]):
    def __init__(
        self,
        target: fl.Attention,
        scale: float = 1.0,
    ) -> None:
        self._scale = scale
        with self.setup_adapter(target):
            clone = target.structural_copy()
            scaled_dot_product = clone.ensure_find(ScaledDotProductAttention)
            palette_cross_attention = PaletteCrossAttention(
                text_cross_attention=clone,
                scale=self.scale,
            )
            clone.replace(
                old_module=scaled_dot_product,
                new_module=fl.Sum(
                    scaled_dot_product,
                    palette_cross_attention,
                ),
            )
            super().__init__(
                clone,
            )

    @property
    def palette_cross_attention(self) -> PaletteCrossAttention:
        return self.ensure_find(PaletteCrossAttention)

    @property
    def image_key_projection(self) -> fl.Linear:
        return self.palette_cross_attention.Distribute[1].Linear

    @property
    def image_value_projection(self) -> fl.Linear:
        return self.palette_cross_attention.Distribute[2].Linear

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value
        self.palette_cross_attention.scale = value

    def load_weights(self, key_tensor: Tensor, value_tensor: Tensor) -> None:
        self.image_key_projection.weight = nn.Parameter(key_tensor)
        self.image_value_projection.weight = nn.Parameter(value_tensor)
        self.palette_cross_attention.to(self.device, self.dtype)
    
    @property
    def weights(self) -> list[Tensor]:
        return [self.image_key_projection.weight, self.image_value_projection.weight]


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

        self.sub_adapters: list[PaletteCrossAttentionAdapter] = [
            PaletteCrossAttentionAdapter(target=cross_attn, scale=scale)
            for cross_attn in filter(lambda attn: type(attn) != fl.SelfAttention, target.layers(fl.Attention))
        ]        
    
    @property
    def weights(self) -> List[Tensor]:
        weights : List[Tensor] = []
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
        self.set_context("ip_adapter", {"palette_embedding": color_palette_embedding})

    @property
    def color_palette_encoder(self) -> ColorPaletteEncoder:
        return self._color_palette_encoder[0]


