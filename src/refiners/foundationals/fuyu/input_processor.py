from typing import List, Tuple

from torch import Tensor, bool as tbool, cat, device as Device, dtype as DType, ones, tensor, tril
from torchvision.transforms.functional import resize

import refiners.fluxion.layers as fl
from refiners.foundationals.clip.text_encoder import TokenEncoder
from refiners.foundationals.fuyu.common import PatchPadding
from refiners.foundationals.fuyu.tokenizer import FuyuTokenizer


class ImageEncoder(fl.Chain):
    """
    Image Encoding Layer.

    This layer pad, normalize, patchify and project a given tensor of images
    """
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
            PatchPadding(  #Pad
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
    """
    Input Encoding Layer.

    This layer encode both the text and the image and set them 
    in the right format as an input for the Fuyu model

    Warning:
        This layer doesn't handle yet batches

    """
    def __init__(
        self,
        embedding_dim: int = 4096,
        max_sequence_length: int = 16_384,
        vocabulary_size: int = 262_144,
        tokenizer: FuyuTokenizer | None = None,
        patch_size: int = 30,
        padding_value: float = 1.,
        max_size: Tuple[int] = (1920,1080),
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
        self.max_size=max_size

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

    def forward(self, image: Tensor, prompt: str, answer: str = None) -> Tuple[Tensor, Tensor]:
        _, _, h, w = image.shape

        if h > self.max_size[1] or w > self.max_size[0]:
            scale_factor = min(self.max_size[0]/w, self.max_size[1]/h)
            image = resize(image, [int(scale_factor*h), int(scale_factor*w)])
            _, _, h, w = image.shape
            
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
        linebreak_embedding = linebreak_embedding.expand(1, n_linebreak, 1, self.embedding_dim)
        # Reshape encoded_image to introduce a slot for linebreaks
        encoded_image = patched_image.view(1, n_linebreak, f_linebreak, self.embedding_dim)
        # Concatenate linebreak embeddings
        encoded_image = cat((encoded_image, linebreak_embedding), dim=2)
        # Reshape to final desired flat format [b seq_len embedding_dim]
        encoded_image = encoded_image.view(1, -1, self.embedding_dim)

        if answer is not None:
            token = cat(
                [
                    Tensor([[self.tokenizer.bos_token_id]]).to(self.device),
                    self.tokenizer(prompt).to(self.device),
                    Tensor([[self.tokenizer.boa_token_id]]).to(self.device),
                    self.tokenizer(answer).to(self.device)
                ],
                dim=1,
            ).to(int)
        else:
            token = cat(
                [
                    Tensor([[self.tokenizer.bos_token_id]]).to(self.device),
                    self.tokenizer(prompt).to(self.device),
                    Tensor([[self.tokenizer.boa_token_id]]).to(self.device)
                ],
                dim=1,
            ).to(int)

        # Tokenize and encode text
        encoded_text = self.token_encoder(token) 
        padded_encoded_images = encoded_image
        len_seq = encoded_text.shape[1] + encoded_image.shape[1]
        attn_mask = ones(1, len_seq , len_seq, device=self.device, dtype=tbool)
        causal_mask = tril(ones((1, len_seq, len_seq), device=self.device, dtype=tbool))
        attn_mask = attn_mask & causal_mask
        context = self.use_context(context_name="attention")
        context.update({"mask": attn_mask})

        encoded_inputs = cat((padded_encoded_images, encoded_text), dim=1)
        return encoded_inputs