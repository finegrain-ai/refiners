from dataclasses import dataclass
from typing import Any

from torch import device as Device, dtype as DType, float16 as Tfloat16, float32 as Tfloat32

import refiners.fluxion.layers as fl
from refiners.foundationals.fuyu.input_processor import InputEncoder
from refiners.foundationals.fuyu.tokenizer import FuyuTokenizer
from refiners.foundationals.fuyu.transformers import FuyuTransformer, FuyuTransformerLayer


def create_fuyu(config):
    model = Fuyu(
        embedding_dim=config.embedding_dim,
        feedforward_dim=config.feedforward_dim,
        max_sequence_length=config.max_sequence_length,
        vocabulary_size=config.vocabulary_size,
        tokenizer=config.tokenizer,
        patch_size=config.patch_size,
        padding_value=config.padding_value,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        norm_eps=config.norm_eps,
        base=config.base,
        partial_rotary_factor=config.partial_rotary_factor,
        use_bias=config.use_bias,
        is_causal=config.is_causal,
        is_optimized=config.is_optimized,
        device=config.device,
        dtype=config.dtype
    )
    return(model)

@dataclass(frozen=True)
class Fuyu8b:
    embedding_dim: int = 4_096
    feedforward_dim: int = 16_384
    max_sequence_length: int = 16_384
    vocabulary_size: int = 262_144
    tokenizer: FuyuTokenizer | None = FuyuTokenizer()
    patch_size: int = 30
    padding_value: float = 1.0/255
    num_layers: int = 36
    num_heads: int = 64
    norm_eps: float = 1e-5
    base: int = 25_000
    partial_rotary_factor: float = 0.5
    use_bias: bool = True
    is_causal: bool = True
    is_optimized: bool = True
    device: Device | str | None = 'cuda'
    dtype: DType | None = Tfloat16

class Fuyu(fl.Chain):
    def __init__(
        self,
        embedding_dim: int,
        feedforward_dim: int,
        max_sequence_length: int,
        vocabulary_size: int,
        tokenizer: FuyuTokenizer | None,
        patch_size: int,
        padding_value: int,
        num_layers: int,
        num_heads: int,
        norm_eps: float,
        base: int,
        partial_rotary_factor:float,
        use_bias: bool,
        is_causal: bool,
        is_optimized: bool,
        device: Device | str | None,
        dtype: DType | None
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
                dtype=dtype
            ),
            FuyuTransformer(
                FuyuTransformerLayer(
                    embedding_dim=embedding_dim,
                    feedforward_dim=feedforward_dim,
                    num_heads=num_heads,
                    norm_eps=norm_eps,
                    base=base,
                    partial_rotary_factor=partial_rotary_factor,
                    use_bias=use_bias,
                    is_causal=is_causal,
                    is_optimized=is_optimized,
                    device=device,
                    dtype=dtype
                )
                for _ in range(num_layers)
            ),
            fl.LayerNorm( 
                normalized_shape=embedding_dim,
                eps=norm_eps,
                device=device,
                dtype=dtype
            ),
            fl.Linear(
                in_features=embedding_dim,
                out_features=vocabulary_size,
                bias=False,
                device=device,
                dtype=dtype
            )
        )
    def init_context(self) -> dict[str, dict[str, Any]]:
        return {"attention": {"mask": None}}