from typing import List, Tuple

from torch import Tensor, bool as tbool, cat, device as Device, dtype as DType, ones, tensor, tril

import refiners.fluxion.layers as fl
from refiners.foundationals.clip.text_encoder import TokenEncoder
from refiners.foundationals.fuyu.common import Padding
from refiners.foundationals.fuyu.tokenizer import FuyuTokenizer


class ImageEncoder(fl.Chain):
    def __init__(
        self,
        patch_size: int = 30,
        embedding_dim: int = 4096,
        padding_value: float = 1.,
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
        B, _, _, _, _, _ = x_unfold.shape
        x_unfold = x_unfold.contiguous()
        x_unfold = x_unfold.view(B, c, -1, self.patch_size, self.patch_size)
        patched = x_unfold.permute(0, 2, 3, 4, 1).reshape(B, -1, c * self.patch_size**2)
        return patched

class InputEncoder(fl.ContextModule):
    def __init__(
        self,
        embedding_dim: int = 4096,
        max_sequence_length: int = 16_384,
        vocabulary_size: int = 262_144,
        tokenizer: FuyuTokenizer | None = None,
        patch_size: int = 30,
        padding_value: float = 1.,
        device: Device | str | None = None,
        dtype: DType | None = None
    ):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.patch_size = patch_size
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.padding_value = padding_value
        self.device=device
        self.dtype=dtype

        self.tokenizer = tokenizer or FuyuTokenizer()

        self.token_encoder = TokenEncoder(
                vocabulary_size=self.vocabulary_size,
                embedding_dim=self.embedding_dim,
                device=self.device,
                dtype=self.dtype
            )

        self.image_encoder = ImageEncoder(
            patch_size=self.patch_size,
            embedding_dim=self.embedding_dim,
            padding_value=self.padding_value,
            device=device,
            dtype=dtype
        )

    def forward(self, image: Tensor, text: List[str]) -> Tuple[Tensor, Tensor]:
        b, _, h, w = image.shape
        assert len(text) == b, "Incoherent number of images compared to the number of prompt"
        
        # Encode Images
        image = image.to(device=self.device, dtype=self.dtype)
        patched_image = self.image_encoder(image)

        h += (self.patch_size - h % self.patch_size) % self.patch_size
        w += (self.patch_size - w % self.patch_size) % self.patch_size
        
        f_linebreak = w // self.patch_size
        n_linebreak = h // self.patch_size
        # Create linebreak embeddings
        linebreak = tensor([self.tokenizer.newline_token_id], device=self.device).long()
        linebreak_embedding = self.token_encoder(linebreak)
        linebreak_embedding = linebreak_embedding.expand(b, n_linebreak, 1, self.embedding_dim)
        # Reshape encoded_image to introduce a slot for linebreaks
        encoded_image = patched_image.view(b, n_linebreak, f_linebreak, self.embedding_dim)
        # Concatenate linebreak embeddings
        encoded_image = cat((encoded_image, linebreak_embedding), dim=2)
        # Reshape to final desired flat format [b seq_len embedding_dim]
        encoded_image = encoded_image.view(b, -1, self.embedding_dim)

        # Tokenize and encode text
        tokens = [self.tokenizer(txt).to(self.device) for txt in text]
        encoded_text = [self.token_encoder(token) for token in tokens]

        # Initialize the 3D attention mask with ones
        max_text_len = max(et.shape[1] for et in encoded_text)
        max_len = max_text_len + encoded_image.shape[1]
        attn_mask = ones(b, max_len, max_len, device=self.device, dtype=tbool)

        padded_encoded_images = []
        for i, et in enumerate(encoded_text):
            padding_length = max_text_len - et.shape[1]
            if padding_length > 0:
                padding_tensor = tensor([self.tokenizer.pad_token['id']] * padding_length, device=self.device).long()
                padding_encoding = self.token_encoder(padding_tensor).view(1, padding_length, -1)
                # Concatenate the padding on the left of the encoded image
                padded_encoded_image = cat((padding_encoding, encoded_image[i].unsqueeze(0)), dim=1)
            else:
                # No padding needed, use the encoded image as is
                padded_encoded_image = encoded_image[i].unsqueeze(0)
            padded_encoded_images.append(padded_encoded_image)
            attn_mask[i, :padding_length, :] = 0
            attn_mask[i, :, :padding_length] = 0

        causal_mask = tril(ones((max_len, max_len), device=self.device, dtype=tbool))
        attn_mask = attn_mask & causal_mask.unsqueeze(0)
        context = self.use_context(context_name="attention")
        context.update({"mask": attn_mask})

        encoded_inputs = cat([cat((pim, et), dim=1) for pim, et in zip(padded_encoded_images, encoded_text)], dim=0)
        return encoded_inputs