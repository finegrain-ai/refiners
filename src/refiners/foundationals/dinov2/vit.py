from math import sqrt

import torch
from torch import Tensor

import refiners.fluxion.layers as fl
from refiners.fluxion.context import Contexts
from refiners.fluxion.layers.activations import Activation
from refiners.fluxion.utils import interpolate


class ClassToken(fl.Chain):
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


class PositionalEmbedding(fl.Chain):
    """Learnable positional embedding."""

    def __init__(
        self,
        sequence_length: int,
        embedding_dim: int,
        patch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size

        super().__init__(
            fl.Parameter(
                *(sequence_length, embedding_dim),
                device=device,
                dtype=dtype,
            ),
        )


class InterpolateEmbedding(fl.Module):
    """Interpolate the positional embeddings to match the input shape."""

    def __init__(
        self,
        mode: str,
        antialias: bool,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.antialias = antialias
        self.patch_size = patch_size

    def forward(
        self,
        x: torch.Tensor,
        input: torch.Tensor,
    ) -> torch.Tensor:
        cls_embed = x[:, :1, :]  # -> (B, 1, D)
        patch_embed = x[:, 1:, :]  # -> (B, N, D)

        B = patch_embed.shape[0]
        N = patch_embed.shape[1]
        D = patch_embed.shape[2]
        M = int(sqrt(N))
        W = input.shape[2]
        H = input.shape[3]
        w = W // self.patch_size
        h = H // self.patch_size
        assert M * M == N, "The sequence length must be a square number."

        patch_embed = patch_embed.reshape(B, M, M, D)  # -> (B, M, M, D)
        patch_embed = patch_embed.permute(0, 3, 1, 2)  # -> (B, D, M, M)
        patch_embed = interpolate(
            x=patch_embed.to(dtype=torch.float32),
            mode=self.mode,
            antialias=self.antialias,
            size=torch.Size((w, h)),
        ).to(dtype=cls_embed.dtype)  # -> (B, D, w, h)
        patch_embed = patch_embed.permute(0, 2, 3, 1)  # -> (B, w, h, D)
        patch_embed = patch_embed.reshape(B, -1, D)  # -> (B, w*h, D)

        x = torch.cat((cls_embed, patch_embed), dim=1)  # -> (B, w*h+1, D)
        return x


class LayerScale(fl.WeightedModule):
    """Scale the input tensor by a learnable parameter."""

    def __init__(
        self,
        embedding_dim: int,
        init_value: float = 1.0,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        p = torch.nn.Parameter(
            torch.full(
                size=(embedding_dim,),
                fill_value=init_value,
                dtype=dtype,
                device=device,
            ),
        )

        self.register_parameter(name="weight", param=p)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.weight


class FeedForward(fl.Chain):
    """Apply two linear transformations interleaved by an activation function."""

    def __init__(
        self,
        embedding_dim: int,
        feedforward_dim: int,
        activation: Activation,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.feedforward_dim = feedforward_dim
        pre_activation_dim = feedforward_dim * 2 if isinstance(activation, fl.GLU) else feedforward_dim

        super().__init__(
            fl.Linear(
                in_features=embedding_dim,
                out_features=pre_activation_dim,
                device=device,
                dtype=dtype,
            ),
            activation,
            fl.Linear(
                in_features=feedforward_dim,
                out_features=embedding_dim,
                device=device,
                dtype=dtype,
            ),
        )


class PatchEncoder(fl.Chain):
    """Encode an image into a sequence of patches."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size

        super().__init__(
            fl.SetContext(context="dinov2_vit", key="input"),  # save the original input
            fl.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=patch_size,
                stride=patch_size,
                device=device,
                dtype=dtype,
            ),  # (N,3,H,W) -> (N,D,P,P)
            fl.Reshape(out_channels, -1),  # (N,D,P,P) -> (N,D,P²)
            fl.Transpose(1, 2),  # (N,D,P²) -> (N,P²,D)
        )


class TransformerLayer(fl.Chain):
    """Apply a multi-head self-attention mechanism to the input tensor."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        norm_eps: float,
        mlp_ratio: int,
        activation: Activation,
        feedforward_dim: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.norm_eps = norm_eps
        self.mlp_ratio = mlp_ratio
        self.feedforward_dim = feedforward_dim if feedforward_dim is not None else embedding_dim * mlp_ratio

        super().__init__(
            fl.Residual(
                fl.LayerNorm(
                    normalized_shape=embedding_dim,
                    eps=norm_eps,
                    device=device,
                    dtype=dtype,
                ),
                fl.SelfAttention(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    device=device,
                    dtype=dtype,
                ),
                LayerScale(
                    embedding_dim=embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
            ),
            fl.Residual(
                fl.LayerNorm(
                    normalized_shape=embedding_dim,
                    eps=norm_eps,
                    device=device,
                    dtype=dtype,
                ),
                FeedForward(
                    embedding_dim=embedding_dim,
                    feedforward_dim=self.feedforward_dim,
                    activation=activation,
                    device=device,
                    dtype=dtype,
                ),
                LayerScale(
                    embedding_dim=embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )


class Transformer(fl.Chain):
    """Alias for a Chain of TransformerLayer."""


class PositionalEncoder(fl.Residual):
    """Alias for a Residual."""


class Registers(fl.Concatenate):
    """Insert register tokens between CLS token and patches."""

    def __init__(
        self,
        num_registers: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.num_registers = num_registers
        self.embedding_dim = embedding_dim

        super().__init__(
            fl.Slicing(dim=1, end=1),
            fl.Parameter(
                *(num_registers, embedding_dim),
                device=device,
                dtype=dtype,
            ),
            fl.Slicing(dim=1, start=1),
            dim=1,
        )


class ViT(fl.Chain):
    """Vision Transformer (ViT) model.

    See [[arXiv:2010.11929] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
    for more details.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        patch_size: int = 16,
        image_size: int = 224,
        num_layers: int = 12,
        num_heads: int = 12,
        norm_eps: float = 1e-6,
        mlp_ratio: int = 4,
        num_registers: int = 0,
        activation: Activation = fl.GeLU(),
        feedforward_dim: int | None = None,
        interpolate_antialias: bool = False,
        interpolate_mode: str = "bicubic",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize a Vision Transformer (ViT) model.

        Args:
            embedding_dim: The dimension of the embedding.
            patch_size: The size of the patches.
            image_size: The size of the input image.
            num_layers: The number of layers.
            num_heads: The number of heads.
            norm_eps: The epsilon value for normalization.
            mlp_ratio: The ratio for the multi-layer perceptron (MLP).
            num_registers: The number of registers.
            activation: The activation function.
            feedforward_dim: The dimension of the feedforward layer.
            interpolate_antialias: Whether to use antialiasing for interpolation.
            interpolate_mode: The interpolation mode.
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """
        num_patches = image_size // patch_size
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.norm_eps = norm_eps
        self.mlp_ratio = mlp_ratio
        self.num_registers = num_registers
        self.feedforward_dim = feedforward_dim

        super().__init__(
            fl.Concatenate(
                ClassToken(
                    embedding_dim=embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
                PatchEncoder(
                    in_channels=3,
                    out_channels=embedding_dim,
                    patch_size=patch_size,
                    device=device,
                    dtype=dtype,
                ),
                dim=1,
            ),
            PositionalEncoder(
                PositionalEmbedding(
                    sequence_length=num_patches**2 + 1,
                    embedding_dim=embedding_dim,
                    patch_size=patch_size,
                    device=device,
                    dtype=dtype,
                ),
                fl.Chain(
                    fl.Parallel(
                        fl.Identity(),
                        fl.UseContext(context="dinov2_vit", key="input"),
                    ),
                    InterpolateEmbedding(
                        mode=interpolate_mode,
                        antialias=interpolate_antialias,
                        patch_size=patch_size,
                    ),
                ),
            ),
            Transformer(
                TransformerLayer(
                    embedding_dim=embedding_dim,
                    feedforward_dim=feedforward_dim,
                    activation=activation,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    norm_eps=norm_eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ),
            fl.LayerNorm(
                normalized_shape=embedding_dim,
                eps=norm_eps,
                device=device,
                dtype=dtype,
            ),
        )

        if self.num_registers > 0:
            registers = Registers(
                num_registers=num_registers,
                embedding_dim=embedding_dim,
                device=device,
                dtype=dtype,
            )
            self.insert_before_type(Transformer, registers)

    def init_context(self) -> Contexts:
        return {
            "dinov2_vit": {
                "input": None,
            },
        }
