from typing import Callable

from torch import Tensor, device as Device, dtype as DType

import refiners.fluxion.layers as fl
from refiners.foundationals.clip.common import FeedForward, PositionalEncoder


class ClassToken(fl.Chain):
    def __init__(self, embedding_dim: int, device: Device | str | None = None, dtype: DType | None = None) -> None:
        self.embedding_dim = embedding_dim
        super().__init__(fl.Parameter(1, embedding_dim, device=device, dtype=dtype))


class PatchEncoder(fl.Chain):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 16,
        use_bias: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.use_bias = use_bias
        super().__init__(
            fl.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(self.patch_size, self.patch_size),
                stride=(self.patch_size, self.patch_size),
                use_bias=self.use_bias,
                device=device,
                dtype=dtype,
            ),
            fl.Permute(0, 2, 3, 1),
        )


class TransformerLayer(fl.Chain):
    def __init__(
        self,
        embedding_dim: int = 768,
        feedforward_dim: int = 3072,
        num_attention_heads: int = 12,
        layer_norm_eps: float = 1e-5,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.feedforward_dim = feedforward_dim
        self.num_attention_heads = num_attention_heads
        self.layer_norm_eps = layer_norm_eps
        super().__init__(
            fl.Residual(
                fl.LayerNorm(normalized_shape=embedding_dim, eps=layer_norm_eps, device=device, dtype=dtype),
                fl.SelfAttention(
                    embedding_dim=embedding_dim, num_heads=num_attention_heads, device=device, dtype=dtype
                ),
            ),
            fl.Residual(
                fl.LayerNorm(normalized_shape=embedding_dim, eps=layer_norm_eps, device=device, dtype=dtype),
                FeedForward(embedding_dim=embedding_dim, feedforward_dim=feedforward_dim, device=device, dtype=dtype),
            ),
        )


class ViTEmbeddings(fl.Chain):
    def __init__(
        self,
        image_size: int = 224,
        embedding_dim: int = 768,
        patch_size: int = 32,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        super().__init__(
            fl.Concatenate(
                ClassToken(embedding_dim, device=device, dtype=dtype),
                fl.Chain(
                    PatchEncoder(
                        in_channels=3,
                        out_channels=embedding_dim,
                        patch_size=patch_size,
                        use_bias=False,
                        device=device,
                        dtype=dtype,
                    ),
                    fl.Reshape((image_size // patch_size) ** 2, embedding_dim),
                ),
                dim=1,
            ),
            fl.Residual(
                PositionalEncoder(
                    max_sequence_length=(image_size // patch_size) ** 2 + 1,
                    embedding_dim=embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )


class CLIPImageEncoder(fl.Chain):
    """Contrastive Language-Image Pretraining (CLIP) image encoder.

    See [[arXiv:2103.00020] Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
    for more details.
    """

    def __init__(
        self,
        image_size: int = 224,
        embedding_dim: int = 768,
        output_dim: int = 512,
        patch_size: int = 32,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        feedforward_dim: int = 3072,
        layer_norm_eps: float = 1e-5,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize a CLIP image encoder.

        Args:
            image_size: The size of the input image.
            embedding_dim: The dimension of the embedding.
            output_dim: The dimension of the output.
            patch_size: The size of the patches.
            num_layers: The number of layers.
            num_attention_heads: The number of attention heads.
            feedforward_dim: The dimension of the feedforward layer.
            layer_norm_eps: The epsilon value for normalization.
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.feedforward_dim = feedforward_dim
        cls_token_pooling: Callable[[Tensor], Tensor] = lambda x: x[:, 0, :]
        super().__init__(
            ViTEmbeddings(
                image_size=image_size, embedding_dim=embedding_dim, patch_size=patch_size, device=device, dtype=dtype
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


class CLIPImageEncoderH(CLIPImageEncoder):
    """CLIP huge image encoder.

    See [[arXiv:2103.00020] Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
    for more details.

    Attributes:
        embedding_dim (int): 1280
        output_dim (int): 1024
        patch_size (int): 14
        num_layers (int): 32
        num_attention_heads (int): 16
        feedforward_dim (int): 5120
    """

    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        """Initialize CLIP huge image encoder.

        Args:
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """
        super().__init__(
            embedding_dim=1280,
            output_dim=1024,
            patch_size=14,
            num_layers=32,
            num_attention_heads=16,
            feedforward_dim=5120,
            device=device,
            dtype=dtype,
        )


class CLIPImageEncoderG(CLIPImageEncoder):
    """CLIP giant image encoder.

    See [[arXiv:2103.00020] Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
    for more details.

    Attributes:
        embedding_dim (int): 1664
        output_dim (int): 1280
        patch_size (int): 14
        num_layers (int): 48
        num_attention_heads (int): 16
        feedforward_dim (int): 8192
    """

    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        """Initialize CLIP giant image encoder.

        Args:
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """
        super().__init__(
            embedding_dim=1664,
            output_dim=1280,
            patch_size=14,
            num_layers=48,
            num_attention_heads=16,
            feedforward_dim=8192,
            device=device,
            dtype=dtype,
        )
