from typing import Callable, List, TypeVar, Any
import refiners.fluxion.layers as fl
from torch import device as Device, dtype as DType, Tensor, histogramdd, stack
from torch.nn.functional import mse_loss as _mse_loss
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet
from refiners.fluxion.adapters.adapter import Adapter
from refiners.foundationals.latent_diffusion.image_prompt import CrossAttentionAdapter
from torch.nn import init

from refiners.foundationals.clip.image_encoder import ClassToken, PositionalEncoder, TransformerLayer

class HistogramDistance(fl.Chain):
    def __init__(
        self, 
        color_bits:int = 8,
    ) -> None:
        self.color_bits = color_bits
        super().__init__(
            fl.Lambda(func=self.kl_div)
        )
    def kl_div(self, x: Tensor, y: Tensor) -> Tensor:
        return _mse_loss(x, y)


class HistogramExtractor(fl.Chain):
    def __init__(
        self, 
        color_bits:int = 8,
    ) -> None:
        self.color_bits = color_bits
        super().__init__(
            fl.Permute(0, 2, 3, 1),
            fl.Lambda(func=self.histogramdd)
        )
    def histogramdd(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        num_pixels = (x.shape[1]*x.shape[2])
        histograms : List[Tensor] = []
        for i in range(batch_size):
            hist_dd = histogramdd(
                x[i], 
                bins=2**self.color_bits,
                range=[
                    0, 2**self.color_bits,
                    0, 2**self.color_bits,
                    0, 2**self.color_bits,
                ])
            hist = hist_dd.hist/num_pixels
            histograms.append(hist)
        
        return stack(histograms)
            
         

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
        output_dim: int = 512,
        patch_size: int = 8,
        num_layers: int = 3,
        num_attention_heads: int = 3,
        feedforward_dim: int = 512,
        layer_norm_eps: float = 1e-5,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.color_bits = color_bits
        cube_size = 2 ** color_bits
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.feedforward_dim = feedforward_dim
        cls_token_pooling: Callable[[Tensor], Tensor] = lambda x: x[:, 0, :]
        super().__init__(
            
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
            fl.Lambda(func=cls_token_pooling),
            fl.LayerNorm(normalized_shape=embedding_dim, eps=layer_norm_eps, device=device, dtype=dtype),
            fl.Linear(in_features=embedding_dim, out_features=output_dim, bias=False, device=device, dtype=dtype),
        )


TSDNet = TypeVar("TSDNet", bound="SD1UNet | SDXLUNet")

class SD1HistogramAdapter(fl.Chain, Adapter[TSDNet]):
    # Prevent PyTorch module registration
    _histogram_encoder: list[HistogramEncoder]

    def __init__(
        self,
        target: TSDNet,
        histogram_encoder: HistogramEncoder,
        scale: float = 1.0,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        with self.setup_adapter(target):
            super().__init__(target)

        self._histogram_encoder = [histogram_encoder]

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
        # Remark : 
        # I've not renamed clip_image_embedding here
        # I feel we should not create a new naming for color_palette since it's the exact same component
        #
        # But rather one would just rename clip_image_embedding and ImageCrossAttention
        # 
        # Naming proposals could be : GenericCrossAttention, NonTextCrossAttention, MediaCrossAttention
        
        self.set_context("ip_adapter", {"clip_image_embedding": histogram_embedding})

    @property
    def histogram_encoder(self) -> HistogramEncoder:
        return self._histogram_encoder[0]
