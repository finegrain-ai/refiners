from torch import device as Device, dtype as DType, Tensor, cat, zeros, ones
from torch.nn.functional import pad
from typing import Union, List, Tuple

import refiners.fluxion.layers as fl
from refiners.foundationals.fuyu.common import Padding
from refiners.foundationals.dinov2.vit import PatchEncoder
from refiners.foundationals.clip.tokenizer import CLIPTokenizer
from refiners.foundationals.clip.text_encoder import TokenEncoder


class TextEncoder(fl.Chain):
    def __init__(
        self,
        embedding_dim: int = 4_096,
        max_sequence_length: int = 16_384,
        vocabulary_size: int = 262_144,
        tokenizer: CLIPTokenizer | None = None,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize text encoder.
        Args:
            embedding_dim: The embedding dimension.
            max_sequence_length: The maximum sequence length.
            vocabulary_size: The vocabulary size.
            tokenizer: The tokenizer.
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """

        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.vocabulary_size = vocabulary_size
        super().__init__(
            tokenizer or CLIPTokenizer(sequence_length=max_sequence_length),
            fl.Converter(set_dtype=False),
            TokenEncoder(
                vocabulary_size=vocabulary_size,
                embedding_dim=embedding_dim,
                device=device,
                dtype=dtype,
            ),
        )


class LineBreakPatchEncoder(fl.Module):
    """
    A custom module that encodes image patches and inserts a specific 'linebreak' embedding
    at the end of each line of patches in an image.
    Args:
        in_channels (int): The number of channels in the input images.
        out_channels (int): The desired dimension of the embeddings
        patch_size (int): The size of each square patch. WARNING: The image width and height should be
        divisible by this number.
        device (Device | str | None, optional): The PyTorch device to use.
        dtype (Dtype | None, optional): The PyTorch data type to use.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size

        self.patch_encoder = PatchEncoder(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            patch_size=self.patch_size,
            device=device,
            dtype=dtype,
        )

        self.linebreak = fl.Parameter(dims=(1, 1, self.out_channels), requires_grad=True, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, c, h, w = x.shape
        patched_x = self.patch_encoder(x)

        f_linebreak = w // self.patch_size
        n_linebreak = h // self.patch_size

        # Reshape patched_x to introduce a slot for linebreaks
        patched_x = patched_x.view(batch_size, n_linebreak, f_linebreak, self.out_channels)
        # Create linebreak embeddings
        linebreak_embedding = self.linebreak.expand(batch_size, n_linebreak, 1, self.out_channels)
        # Concatenate linebreak embeddings
        patched_x_linebreak = cat((patched_x, linebreak_embedding), dim=2)
        # Reshape to final desired flat format
        patched_x_linebreak = patched_x_linebreak.view(batch_size, -1, self.out_channels)

        return patched_x_linebreak


class ImageEncoder(fl.chain):
    def __init__(
        self,
        patch_size: int = 30,
        embedding_dim: int = 4096,
        padding_value: int = 0,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        super().__init__(
            Padding(
                patch_size=patch_size,
                padding_value=padding_value
            ),
            LineBreakPatchEncoder(
                in_channels=3,
                out_channels=embedding_dim,
                patch_size=patch_size,
                device=device,
                dtype=dtype,
            ),
        )


class InputEncoder(fl.Module):
    def __init__(
        self,
        embedding_dim: int = 4096,
        max_sequence_length: int = 16_384,
        vocabulary_size: int = 262_144,
        tokenizer: CLIPTokenizer | None = None,
        patch_size: int = 30,
        padding_value: int = 0,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        super().__init__()

        self.text_encoder = TextEncoder(
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length,
            vocabulary_size=vocabulary_size,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
        )
        self.image_encoder = ImageEncoder(
            patch_size=patch_size, embedding_dim=embedding_dim, padding_value=padding_value, device=device, dtype=dtype
        )

    def forward(self, image: Tensor, text: Union[str, List[str]]) -> Tuple[Tensor, Tensor]:
        encoded_image = self.image_encoder(image)
        encoded_text = self.text_encoder(text)

        n_image_tokens = encoded_image.shape[1]
        n_text_tokens = encoded_text.shape[1]

        encoded_inputs = cat((encoded_image, encoded_text), dim=1)
        encoded_mask = cat((zeros(n_image_tokens), ones(n_text_tokens)))

        return encoded_inputs, encoded_mask
