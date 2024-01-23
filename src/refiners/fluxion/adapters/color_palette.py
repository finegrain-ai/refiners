from typing import Any, List, TypeVar

import torch
from jaxtyping import Float, Int
from torch import Tensor, device as Device, dtype as DType, tensor, zeros
from torch.nn.functional import pad
from torch.nn import init

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.adapter import Adapter
from refiners.foundationals.latent_diffusion.image_prompt import CrossAttentionAdapter
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet
from refiners.foundationals.clip.common import PositionalEncoder
from refiners.foundationals.clip.text_encoder import TransformerLayer

TSDNet = TypeVar("TSDNet", bound="SD1UNet | SDXLUNet")

class ColorsTokenizer(fl.Module):
    def __init__(
        self,
        max_colors: int = 8,
    ) -> None:
        super().__init__()
        self.max_colors = max_colors
    
    def forward(self, colors: Float[Tensor, "*batch colors 3"]) -> Float[Tensor, "*batch max_colors 4"]:        
        colors = self.add_channel(colors)
        colors = self.zero_right_padding(colors)
        return colors
    
    def add_channel(
        self, x: Float[Tensor, "*batch colors 3"]
    ) -> Float[Tensor, "*batch colors_with_end 4"]:
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
    def __init__(
        self,
        embedding_dim: int = 768,
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
        self.embedding_dim = embedding_dim
        self.max_colors = max_colors
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.feedforward_dim = feedforward_dim
        self.layer_norm_eps = layer_norm_eps
        self.use_quick_gelu = use_quick_gelu
        super().__init__(
            ColorsTokenizer(
                max_colors=max_colors
            ),
            fl.Sum(
                ColorEncoder(
                    embedding_dim=embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
                PositionalEncoder(
                    max_sequence_length=max_colors,
                    embedding_dim=embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
            ),
            *(
                # Remark : 
                # The current transformer layer has a causal self-attention
                # It would be fair to test non-causal self-attention
                TransformerLayer(
                    embedding_dim=embedding_dim,
                    num_attention_heads=num_attention_heads,
                    feedforward_dim=feedforward_dim,
                    layer_norm_eps=layer_norm_eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ),
            fl.LayerNorm(normalized_shape=embedding_dim, eps=layer_norm_eps, device=device, dtype=dtype),
        )
        if use_quick_gelu:
            for gelu, parent in self.walk(predicate=lambda m, _: isinstance(m, fl.GeLU)):
                parent.replace(old_module=gelu, new_module=fl.ApproximateGeLU())

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
        # Remark : 
        # I've not renamed clip_image_embedding here
        # I feel we should not create a new naming for color_palette since it's the exact same component
        #
        # But rather one would just rename clip_image_embedding and ImageCrossAttention
        # 
        # Naming proposals could be : GenericCrossAttention, NonTextCrossAttention, MediaCrossAttention
        
        self.set_context("ip_adapter", {"clip_image_embedding": color_palette_embedding})

    @property
    def color_palette_encoder(self) -> ColorPaletteEncoder:
        return self._color_palette_encoder[0]
