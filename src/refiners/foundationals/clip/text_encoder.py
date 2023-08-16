from torch import Tensor, arange, device as Device, dtype as DType

from refiners.fluxion.layers import (
    ApproximateGeLU,
    GeLU,
    Linear,
    LayerNorm,
    Embedding,
    Chain,
    Sum,
    SelfAttention,
    Lambda,
    Residual,
)
from refiners.foundationals.clip.tokenizer import CLIPTokenizer


class PositionalTokenEncoder(Sum):
    structural_attrs = ["vocabulary_size", "positional_embedding_dim"]

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int,
        positional_embedding_dim: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.vocabulary_size = vocabulary_size
        self.positional_embedding_dim = positional_embedding_dim
        super().__init__(
            Embedding(
                num_embeddings=vocabulary_size,
                embedding_dim=embedding_dim,
                device=device,
                dtype=dtype,
            ),
            Chain(
                Lambda(func=self.get_position_ids),
                Embedding(
                    num_embeddings=positional_embedding_dim,
                    embedding_dim=embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )

    @property
    def position_ids(self) -> Tensor:
        return arange(end=self.positional_embedding_dim, device=self.device).reshape(1, -1)

    def get_position_ids(self, x: Tensor) -> Tensor:
        return self.position_ids[:, : x.shape[1]]


class FeedForward(Chain):
    structural_attrs = ["embedding_dim", "feedforward_dim"]

    def __init__(
        self,
        embedding_dim: int,
        feedforward_dim: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.feedforward_dim = feedforward_dim
        super().__init__(
            Linear(in_features=embedding_dim, out_features=feedforward_dim, device=device, dtype=dtype),
            GeLU(),
            Linear(in_features=feedforward_dim, out_features=embedding_dim, device=device, dtype=dtype),
        )


class TransformerLayer(Chain):
    structural_attrs = ["embedding_dim", "num_attention_heads", "feedforward_dim", "layer_norm_eps"]

    def __init__(
        self,
        embedding_dim: int,
        feedforward_dim: int,
        num_attention_heads: int = 1,
        layer_norm_eps: float = 1e-5,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.feedforward_dim = feedforward_dim
        self.layer_norm_eps = layer_norm_eps
        super().__init__(
            Residual(
                LayerNorm(
                    normalized_shape=embedding_dim,
                    eps=layer_norm_eps,
                    device=device,
                    dtype=dtype,
                ),
                SelfAttention(
                    embedding_dim=embedding_dim,
                    num_heads=num_attention_heads,
                    is_causal=True,
                    device=device,
                    dtype=dtype,
                ),
            ),
            Residual(
                LayerNorm(
                    normalized_shape=embedding_dim,
                    eps=layer_norm_eps,
                    device=device,
                    dtype=dtype,
                ),
                FeedForward(
                    embedding_dim=embedding_dim,
                    feedforward_dim=feedforward_dim,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )


class CLIPTextEncoder(Chain):
    structural_attrs = [
        "embedding_dim",
        "positional_embedding_dim",
        "vocabulary_size",
        "num_layers",
        "num_attention_heads",
        "feedforward_dim",
        "layer_norm_eps",
        "use_quick_gelu",
        "tokenizer",
    ]

    def __init__(
        self,
        embedding_dim: int = 768,
        positional_embedding_dim: int = 77,
        vocabulary_size: int = 49408,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        feedforward_dim: int = 3072,
        layer_norm_eps: float = 1e-5,
        use_quick_gelu: bool = False,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.positional_embedding_dim = positional_embedding_dim
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.feedforward_dim = feedforward_dim
        self.layer_norm_eps = layer_norm_eps
        self.use_quick_gelu = use_quick_gelu
        self.tokenizer = CLIPTokenizer()
        super().__init__(
            PositionalTokenEncoder(
                vocabulary_size=vocabulary_size,
                embedding_dim=embedding_dim,
                positional_embedding_dim=positional_embedding_dim,
                device=device,
                dtype=dtype,
            ),
            *(
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
            LayerNorm(normalized_shape=embedding_dim, eps=layer_norm_eps, device=device, dtype=dtype),
        )
        if use_quick_gelu:
            for gelu, parent in self.walk(predicate=lambda m, _: isinstance(m, GeLU)):
                parent.replace(old_module=gelu, new_module=ApproximateGeLU())

    def encode(self, text: str) -> Tensor:
        tokens = self.tokenizer(text, sequence_length=self.positional_embedding_dim).to(device=self.device)
        return self(tokens)

    @property
    def unconditional_text_embedding(self) -> Tensor:
        return self.encode(text="")


class CLIPTextEncoderL(CLIPTextEncoder):
    """
    CLIPTextEncoderL is the CLIP text encoder with the following parameters:
    embedding_dim=768
    num_layers=12
    num_attention_heads=12
    feedforward_dim=3072
    use_quick_gelu=True

    We replace the GeLU activation function with an approximate GeLU to comply with the original CLIP implementation
    of OpenAI (https://github.com/openai/CLIP/blob/main/clip/model.py#L166)
    """

    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        super().__init__(
            embedding_dim=768,
            num_layers=12,
            num_attention_heads=12,
            feedforward_dim=3072,
            use_quick_gelu=True,
            device=device,
            dtype=dtype,
        )


class CLIPTextEncoderH(CLIPTextEncoder):
    """
    CLIPTextEncoderH is the CLIP text encoder with the following parameters:
    embedding_dim=1024
    num_layers=23
    num_attention_heads=16
    feedforward_dim=4096
    """

    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        super().__init__(
            embedding_dim=1024,
            num_layers=23,
            num_attention_heads=16,
            feedforward_dim=4096,
            device=device,
            dtype=dtype,
        )


class CLIPTextEncoderG(CLIPTextEncoder):
    """
    CLIPTextEncoderG is the CLIP text encoder with the following parameters:
    embedding_dim=1280
    num_layers=32
    num_attention_heads=16
    feedforward_dim=5120
    """

    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        super().__init__(
            embedding_dim=1280,
            num_layers=32,
            num_attention_heads=20,
            feedforward_dim=5120,
            device=device,
            dtype=dtype,
        )
