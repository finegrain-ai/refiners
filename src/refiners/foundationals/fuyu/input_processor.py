from typing import List, Tuple, Union

from torch import Tensor, cat, device as Device, dtype as DType, tensor

import refiners.fluxion.layers as fl
from refiners.foundationals.clip.text_encoder import TokenEncoder
from refiners.foundationals.clip.tokenizer import CLIPTokenizer
from refiners.foundationals.fuyu.common import Padding


class ImageEncoder(fl.Chain):
    def __init__(
        self,
        patch_size: int = 30,
        embedding_dim: int = 4096,
        padding_value: int = 1,
        image_mean: float = 0.5,
        image_std: float = 0.5,
        use_bias: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None
    ) -> None:
        self.patch_size=patch_size
        self.embedding_dim=embedding_dim
        self.padding_value=padding_value

        super().__init__(
            Padding(  #Pad
                patch_size=self.patch_size,
                padding_value=self.padding_value
            ),
            fl.Lambda(  #Normalize
                lambda x : (x - image_mean)/image_std
            ),
            fl.Lambda(  
                lambda x : self.patchify(x)
            ),
            fl.Linear(
                in_features=3*self.patch_size**2,
                out_features=self.embedding_dim,
                bias=use_bias,
                device=device,
                dtype=dtype
            )
        )

    def patchify(self, x: Tensor) -> Tensor:
        _, c, _, _ = x.shape
        x_unfold = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        B, _, H_new, W_new, _, _ = x_unfold.shape
        patched = x_unfold.permute(0, 2, 3, 1, 4, 5).reshape(B, H_new * W_new, c * self.patch_size**2)
        return patched

class TextEncoder(fl.Chain):
    def __init__(
            self,
            tokenizer,
            max_sequence_length,
            vocabulary_size,
            embedding_dim,
            device,
            dtype
    ):
        self.max_sequence_length = max_sequence_length
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.token_encoder = [
            TokenEncoder(
                vocabulary_size=self.vocabulary_size,
                embedding_dim=self.embedding_dim,
                device=device,
                dtype=dtype
            )
        ]
        super().__init__(
            tokenizer or CLIPTokenizer(sequence_length=max_sequence_length),
            fl.Converter(set_dtype=False),
            self.token_encoder[0]
        )

class InputEncoder(fl.Module):
    def __init__(
        self,
        embedding_dim: int = 4096,
        max_sequence_length: int = 16_384,
        vocabulary_size: int = 262_144,
        tokenizer: CLIPTokenizer | None = None,
        patch_size: int = 30,
        padding_value: int = 1,
        linebreak_token: int = 71019,
        device: Device | str | None = None,
        dtype: DType | None = None
    ):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.patch_size = patch_size
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.padding_value = padding_value
        self.linebreak_token = linebreak_token
        self.device=device

        self.text_encoder = TextEncoder(
            tokenizer=tokenizer,
            embedding_dim=self.embedding_dim,
            max_sequence_length=self.max_sequence_length,
            vocabulary_size=self.vocabulary_size,
            device=device,
            dtype=dtype
        )

        self.image_encoder = ImageEncoder(
            patch_size=self.patch_size,
            embedding_dim=self.embedding_dim,
            padding_value=self.padding_value,
            device=device,
            dtype=dtype
        )

    def forward(self, image: Tensor, text: Union[str, List[str]]) -> Tuple[Tensor, Tensor]:
        b, _, h, w = image.shape
        patched_image = self.image_encoder(image)
        
        f_linebreak = w // self.patch_size
        n_linebreak = h // self.patch_size
        # Create linebreak embeddings
        linebreak = tensor([self.linebreak_token], device=self.device).long()
        linebreak_embedding = self.text_encoder.token_encoder[0](linebreak)
        linebreak_embedding = linebreak_embedding.expand(b, n_linebreak, 1, self.embedding_dim)
        # Reshape encoded_image to introduce a slot for linebreaks
        encoded_image = patched_image.view(b, n_linebreak, f_linebreak, self.embedding_dim)
        # Concatenate linebreak embeddings
        encoded_image = cat((encoded_image, linebreak_embedding), dim=2)
        # Reshape to final desired flat format
        encoded_image = encoded_image.view(b, -1, self.embedding_dim)

        encoded_text = self.text_encoder(text)

        # n_image_tokens = encoded_image.shape[1]
        # n_text_tokens = encoded_text.shape[1]

        encoded_inputs = cat((encoded_image, encoded_text), dim=1)
        # encoded_mask = cat((zeros(n_image_tokens), ones(n_text_tokens)))

        return encoded_inputs #, encoded_mask