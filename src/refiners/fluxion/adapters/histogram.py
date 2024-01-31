from typing import Any, List, TypeVar

from torch import Tensor, sort, flatten, cat, device as Device, dtype as DType, histogramdd, histogram, nn, stack, zeros_like
from torch.nn import init, L1Loss
from torch.nn.functional import mse_loss as _mse_loss

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.adapter import Adapter
from refiners.fluxion.layers.attentions import ScaledDotProductAttention
from refiners.foundationals.clip.image_encoder import ClassToken, PositionalEncoder, TransformerLayer
from refiners.foundationals.latent_diffusion.image_prompt import CrossAttentionAdapter
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet
from PIL import Image
from refiners.fluxion.utils import images_to_tensor

def images_to_histo_channels(images: List[Image.Image], color_bits: int = 8) -> List[Tensor]:
    img_tensor = images_to_tensor(images)
    sorted_channels = tensor_to_sorted_channels(img_tensor)
    return sorted_channels_to_histo_channels(sorted_channels, color_bits=color_bits)
    

def tensor_to_sorted_channels(image: Tensor, color_bits: int = 8) -> List[Tensor]:
    sorted_channels: List[Tensor] = []
    channels: List[Tensor] = image.split(1, dim=1)
    for channel in channels:
        # We extract RGB curves
        sorted_channel, _ = sort(flatten(channel, 1))
        sorted_channels.append(sorted_channel)
    return sorted_channels

def sorted_channels_to_histo_channels(sorted_channels: List[Tensor], color_bits: int = 8) -> List[Tensor]:
    histos: List[Tensor] = []
    for channel in sorted_channels:
        print(f"channel.shape: {channel.shape}")
        histograms: List[Tensor] = []
        for i in range(channel.shape[0]):
            elem = channel[i]
            histo, _ = histogram(
                        elem.cpu(),
                        bins=2**color_bits,
                        range= (0.0,1.0)
                    )
            histograms.append(histo/elem.numel())
        histo = stack(histograms)
        histos.append(histo)
    return histos

def histogram_to_histo_channels(histogram: Tensor) -> List[Tensor]:
    red = histogram.sum(dim=(2,3))
    green = histogram.sum(dim=(1,3))
    blue = histogram.sum(dim=(1,2))
    
    return [red, green, blue]

class ColorLoss(fl.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = L1Loss()

    def forward(self, actual: Tensor, expected: Tensor) -> Tensor:
        assert actual.shape == expected.shape, f"Shapes should match {actual.shape}/{expected.shape}"
        assert actual.shape[1] == 3, f"3 channels (R,G,B) image expected"
        return self.l1_loss(tensor_to_sorted_channels(actual), tensor_to_sorted_channels(expected))

class HistogramDistance(fl.Chain):
    def __init__(
        self,
        color_bits: int = 8,
    ) -> None:
        self.color_bits = color_bits
        super().__init__(fl.Lambda(func=self.kl_div))

    def kl_div(self, x: Tensor, y: Tensor) -> Tensor:
        return _mse_loss(x, y)


class HistogramExtractor(fl.Chain):
    def __init__(
        self,
        color_bits: int = 8,
    ) -> None:
        self.color_bits = color_bits
        self.color_size = 2**color_bits
        super().__init__(fl.Permute(0, 2, 3, 1), fl.Lambda(func=self.histogramdd))

    def histogramdd(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        num_pixels = x.shape[1] * x.shape[2]
        histograms: List[Tensor] = []
        device = x.device
        # x is a [0, 1] normalized image
        x = x * (self.color_size - 1)

        for i in range(batch_size):
            hist_dd = histogramdd(
                x[i].cpu(),
                bins=2**self.color_bits,
                range=[
                    0,
                    2**self.color_bits,
                    0,
                    2**self.color_bits,
                    0,
                    2**self.color_bits,
                ],
            )
            hist = hist_dd.hist / num_pixels
            histograms.append(hist)

        return stack(histograms).to(device)
    
    def images_to_histograms(self, images: List[Image.Image], device: Device | None = None, dtype : DType | None = None) -> Tensor:
        return self(images_to_tensor(images, device=device, dtype = dtype))


class Patch3dEncoder(fl.Chain):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 8,
        use_bias: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.use_bias = use_bias
        super().__init__(
            fl.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(self.patch_size, self.patch_size, self.patch_size),
                stride=(self.patch_size, self.patch_size, self.patch_size),
                use_bias=self.use_bias,
                device=device,
                dtype=dtype,
            ),
        )


class ViT3dEmbeddings(fl.Chain):
    def __init__(
        self,
        cube_size: int = 256,
        embedding_dim: int = 768,
        patch_size: int = 8,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.cube_size = cube_size
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        super().__init__(
            fl.Concatenate(
                ClassToken(embedding_dim, device=device, dtype=dtype),
                fl.Chain(
                    Patch3dEncoder(
                        in_channels=1,
                        out_channels=embedding_dim,
                        patch_size=patch_size,
                        use_bias=False,
                        device=device,
                        dtype=dtype,
                    ),
                    fl.Reshape((cube_size // patch_size) ** 3, embedding_dim),
                ),
                dim=1,
            ),
            fl.Residual(
                PositionalEncoder(
                    max_sequence_length=(cube_size // patch_size) ** 3 + 1,
                    embedding_dim=embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )


class HistogramEncoder(fl.Chain):
    def __init__(
        self,
        color_bits: int = 8,
        embedding_dim: int = 768,
        patch_size: int = 8,
        num_layers: int = 3,
        num_attention_heads: int = 3,
        feedforward_dim: int = 512,
        layer_norm_eps: float = 1e-5,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.color_bits = color_bits
        cube_size = 2**color_bits
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.feedforward_dim = feedforward_dim
        super().__init__(
            fl.Reshape(1, cube_size, cube_size, cube_size),
            ViT3dEmbeddings(
                cube_size=cube_size, embedding_dim=embedding_dim, patch_size=patch_size, device=device, dtype=dtype
            ),
            fl.LayerNorm(normalized_shape=embedding_dim, eps=layer_norm_eps, device=device, dtype=dtype),
            fl.Chain(
                TransformerLayer(
                    embedding_dim=embedding_dim,
                    feedforward_dim=feedforward_dim,
                    num_attention_heads=num_attention_heads,
                    layer_norm_eps=layer_norm_eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ),
        )

    def compute_histogram_embedding(
        self,
        x: Tensor,
        negative_histogram: None | Tensor = None,
    ) -> Tensor:
        conditional_embedding = self(x)
        if x == negative_histogram:
            return cat(tensors=(conditional_embedding, conditional_embedding), dim=0)

        if negative_histogram is None:
            # a uniform palette with all the colors at the same frequency
            numel: int = x.numel()
            if numel == 0:
                raise ValueError("Cannot compute histogram embedding for empty tensor")
            negative_histogram = (zeros_like(x) + 1.0) * 1 / numel

        negative_embedding = self(negative_histogram)
        return cat(tensors=(negative_embedding, conditional_embedding), dim=0)


class HistogramCrossAttention(fl.Chain):
    def __init__(self, text_cross_attention: fl.Attention, embedding_dim: int = 768, scale: float = 1.0) -> None:
        self._scale = scale
        super().__init__(
            fl.Distribute(
                fl.Identity(),
                fl.Chain(
                    fl.UseContext(context="ip_adapter", key="histogram_embedding"),
                    fl.Linear(
                        in_features=embedding_dim,
                        out_features=text_cross_attention.inner_dim,
                        bias=text_cross_attention.use_bias,
                        device=text_cross_attention.device,
                        dtype=text_cross_attention.dtype,
                    ),
                ),
                fl.Chain(
                    fl.UseContext(context="ip_adapter", key="histogram_embedding"),
                    fl.Linear(
                        in_features=embedding_dim,
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

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value
        self.ensure_find(fl.Multiply).scale = value


class HistogramCrossAttentionAdapter(fl.Chain, Adapter[fl.Attention]):
    def __init__(
        self,
        target: fl.Attention,
        scale: float = 1.0,
        embedding_dim: int = 768
    ) -> None:
        self._scale = scale
        with self.setup_adapter(target):
            clone = target.structural_copy()
            scaled_dot_product = clone.ensure_find(ScaledDotProductAttention)
            histogram_cross_attention = HistogramCrossAttention(
                text_cross_attention=clone,
                embedding_dim=embedding_dim,
                scale=self.scale,
            )
            clone.replace(
                old_module=scaled_dot_product,
                new_module=fl.Sum(
                    scaled_dot_product,
                    histogram_cross_attention,
                ),
            )
            super().__init__(
                clone,
            )

    @property
    def histogram_cross_attention(self) -> HistogramCrossAttention:
        return self.ensure_find(HistogramCrossAttention)

    @property
    def image_key_projection(self) -> fl.Linear:
        return self.histogram_cross_attention.Distribute[1].Linear

    @property
    def image_value_projection(self) -> fl.Linear:
        return self.histogram_cross_attention.Distribute[2].Linear

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value
        self.histogram_cross_attention.scale = value

    def load_weights(self, key_tensor: Tensor, value_tensor: Tensor) -> None:
        self.image_key_projection.weight = nn.Parameter(key_tensor)
        self.image_value_projection.weight = nn.Parameter(value_tensor)
        self.histogram_cross_attention.to(self.device, self.dtype)

    @property
    def weights(self) -> list[Tensor]:
        return [self.image_key_projection.weight, self.image_value_projection.weight]


TSDNet = TypeVar("TSDNet", bound="SD1UNet | SDXLUNet")


class SD1HistogramAdapter(fl.Chain, Adapter[TSDNet]):
    # Prevent PyTorch module registration
    _histogram_encoder: list[HistogramEncoder]

    def __init__(
        self,
        target: TSDNet,
        embedding_dim: int = 768,
        scale: float = 1.0,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        with self.setup_adapter(target):
            super().__init__(target)

        self.sub_adapters: list[CrossAttentionAdapter] = [
            HistogramCrossAttentionAdapter(target=cross_attn, scale=scale, embedding_dim=embedding_dim)
            for cross_attn in filter(lambda attn: type(attn) != fl.SelfAttention, target.layers(fl.Attention))
        ]

    @property
    def weights(self) -> List[Tensor]:
        weights: List[Tensor] = []
        for adapter in self.sub_adapters:
            weights += adapter.weights
        return weights

    def zero_init(self) -> None:
        weights = self.weights
        for weight in weights:
            init.zeros_(weight)

    def inject(self, parent: fl.Chain | None = None) -> "SD1HistogramAdapter[Any]":
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

    def set_histogram_embedding(self, histogram_embedding: Tensor) -> None:
        self.set_context("ip_adapter", {"histogram_embedding": histogram_embedding})
