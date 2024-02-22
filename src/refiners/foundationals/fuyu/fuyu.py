from dataclasses import dataclass
from torch import Size, Tensor, device as Device, dtype as DType

import refiners.fluxion.layers as fl
from refiners.foundationals.clip.tokenizer import CLIPTokenizer
from refiners.foundationals.fuyu.input_processor import InputEncoder
from refiners.foundationals.fuyu.transformers import FuyuTransformer, FuyuTransformerLayer


# @dataclass(frozen=True)
# class FuyuConfig8b:
#     embedding_dim: int = 4_096
#     feedforward_dim: int = 16_384
#     inner_dim: int | None = None
#     max_sequence_length: int = 16_384
#     vocabulary_size: int = 262_144
#     tokenizer: CLIPTokenizer | None = None
#     patch_size: int = 30
#     padding_value: int = 0
#     num_layers: int = 36
#     num_heads: int = 64
#     norm_eps: float = 1e-6
#     base: int = 10_000
#     use_bias: bool = True
#     is_causal: bool = True
#     is_optimized: bool = True
#     device: Device | str | None = None
#     dtype: DType | None = None


class Fuyu(fl.Chain):
    def __init__(
        self,
        embedding_dim: int = 4096,
        feedforward_dim: int = 16384,
        inner_dim: int | None = None,
        max_sequence_length: int = 16_384,
        vocabulary_size: int = 262_144,
        tokenizer: CLIPTokenizer | None = None,
        patch_size: int = 30,
        padding_value: int = 0,
        num_layers: int = 36,
        num_heads: int = 64,
        norm_eps: float = 1e-6,
        base: int = 10_000,
        use_bias: bool = True,
        is_causal: bool = True,
        is_optimized: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            InputEncoder(
                embedding_dim=embedding_dim,
                max_sequence_length=max_sequence_length,
                vocabulary_size=vocabulary_size,
                tokenizer=tokenizer,
                patch_size=patch_size,
                padding_value=padding_value,
                device=device,
                dtype=dtype,
            ),
            FuyuTransformer(
                FuyuTransformerLayer(
                    embedding_dim=embedding_dim,
                    inner_dim=inner_dim,
                    feedforward_dim=feedforward_dim,
                    num_heads=num_heads,
                    norm_eps=norm_eps,
                    base=base,
                    use_bias=use_bias,
                    is_causal=is_causal,
                    is_optimized=is_optimized,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ),
        )
        # Layer Norm
        # Linear
        # SoftMax
