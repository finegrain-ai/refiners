from typing import Iterable
import refiners.fluxion.layers as fl
from src.refiners.fluxion.layers.module import Module
from torch import device as Device, dtype as DType, Tensor, histogramdd, stack
from torch.nn.functional import kl_div as _kl_div, mse_loss as _mse_loss

from src.refiners.foundationals.clip.image_encoder import ClassToken, PositionalEncoder, TransformerLayer

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
        histograms = []
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
