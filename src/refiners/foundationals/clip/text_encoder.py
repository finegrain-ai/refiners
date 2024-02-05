from torch import device as Device, dtype as DType

import refiners.fluxion.layers as fl
from refiners.foundationals.clip.common import FeedForward, PositionalEncoder
from refiners.foundationals.clip.tokenizer import CLIPTokenizer


class TokenEncoder(fl.Embedding):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        super().__init__(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            device=device,
            dtype=dtype,
        )


class TransformerLayer(fl.Chain):
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
            fl.Residual(
                fl.LayerNorm(
                    normalized_shape=embedding_dim,
                    eps=layer_norm_eps,
                    device=device,
                    dtype=dtype,
                ),
                fl.SelfAttention(
                    embedding_dim=embedding_dim,
                    num_heads=num_attention_heads,
                    is_causal=True,
                    device=device,
                    dtype=dtype,
                ),
            ),
            fl.Residual(
                fl.LayerNorm(
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


class CLIPTextEncoder(fl.Chain):
    """Contrastive Language-Image Pretraining (CLIP) text encoder.

    See [[arXiv:2103.00020] Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
    for more details.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        max_sequence_length: int = 77,
        vocabulary_size: int = 49408,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        feedforward_dim: int = 3072,
        layer_norm_eps: float = 1e-5,
        use_quick_gelu: bool = False,
        tokenizer: CLIPTokenizer | None = None,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize CLIP text encoder.

        Args:
            embedding_dim: The embedding dimension.
            max_sequence_length: The maximum sequence length.
            vocabulary_size: The vocabulary size.
            num_layers: The number of layers.
            num_attention_heads: The number of attention heads.
            feedforward_dim: The feedforward dimension.
            layer_norm_eps: The epsilon value for layer normalization.
            use_quick_gelu: Whether to use the quick GeLU activation function.
            tokenizer: The tokenizer.
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.feedforward_dim = feedforward_dim
        self.layer_norm_eps = layer_norm_eps
        self.use_quick_gelu = use_quick_gelu
        super().__init__(
            tokenizer or CLIPTokenizer(sequence_length=max_sequence_length),
            fl.Converter(set_dtype=False),
            fl.Sum(
                TokenEncoder(
                    vocabulary_size=vocabulary_size,
                    embedding_dim=embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
                PositionalEncoder(
                    max_sequence_length=max_sequence_length,
                    embedding_dim=embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
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
            fl.LayerNorm(normalized_shape=embedding_dim, eps=layer_norm_eps, device=device, dtype=dtype),
        )
        if use_quick_gelu:
            for gelu, parent in self.walk(predicate=lambda m, _: isinstance(m, fl.GeLU)):
                parent.replace(
                    old_module=gelu,
                    new_module=fl.GeLU(approximation=fl.GeLUApproximation.SIGMOID),
                )


class CLIPTextEncoderL(CLIPTextEncoder):
    """CLIP large text encoder.

    Note:
        We replace the GeLU activation function with an approximate GeLU to comply with the original CLIP implementation
        of OpenAI (https://github.com/openai/CLIP/blob/a1d0717/clip/model.py#L166)

    See [[arXiv:2103.00020] Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
    for more details.

    Attributes:
        embedding_dim (int): 768
        num_layers (int): 12
        num_attention_heads (int): 12
        feedforward_dim (int): 3072
        use_quick_gelu (bool): True
    """

    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        """Initialize CLIP large text encoder.

        Args:
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """
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
    """CLIP huge text encoder.

    See [[arXiv:2103.00020] Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
    for more details.

    Attributes:
        embedding_dim (int): 1024
        num_layers (int): 23
        num_attention_heads (int): 16
        feedforward_dim (int): 4096
    """

    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        """Initialize CLIP huge text encoder.

        Args:
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """
        super().__init__(
            embedding_dim=1024,
            num_layers=23,
            num_attention_heads=16,
            feedforward_dim=4096,
            device=device,
            dtype=dtype,
        )


class CLIPTextEncoderG(CLIPTextEncoder):
    """CLIP giant text encoder.

    See [[arXiv:2103.00020] Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
    for more details.

    Attributes:
        embedding_dim (int): 1280
        num_layers (int): 32
        num_attention_heads (int): 20
        feedforward_dim (int): 5120
        tokenizer (CLIPTokenizer): CLIPTokenizer(pad_token_id=0)
    """

    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        """Initialize CLIP giant text encoder.

        Args:
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """
        tokenizer = CLIPTokenizer(pad_token_id=0)
        super().__init__(
            embedding_dim=1280,
            num_layers=32,
            num_attention_heads=20,
            feedforward_dim=5120,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
        )
