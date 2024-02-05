from torch import device as Device, dtype as DType
from torch.nn import Embedding as _Embedding

from refiners.fluxion.layers.module import WeightedModule


class Embedding(_Embedding, WeightedModule):
    """Embedding layer.

    This layer wraps [`torch.nn.Embedding`][torch.nn.Embedding].

    Receives:
        (Int[Tensor, "batch length"]):

    Returns:
        (Float[Tensor, "batch length embedding_dim"]):

    Example:
        ```py
        embedding = fl.Embedding(
            num_embeddings=10,
            embedding_dim=128
        )

        tensor = torch.randint(0, 10, (2, 10))
        output = embedding(tensor)

        assert output.shape == (2, 10, 128)
        ```
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        """Initializes the Embedding layer.

        Args:
            num_embeddings: The number of embeddings.
            embedding_dim: The dimension of the embeddings.
            device: The device to use for the embedding layer.
            dtype: The dtype to use for the embedding layer.
        """
        _Embedding.__init__(  # type: ignore
            self,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            device=device,
            dtype=dtype,
        )
