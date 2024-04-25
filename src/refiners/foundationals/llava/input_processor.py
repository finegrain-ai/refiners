from typing import Tuple

import torch
import torchvision.transforms as transforms  # type: ignore
from PIL import Image
from torch import Tensor

import refiners.fluxion.layers as fl
from refiners.foundationals.clip import CLIPImageEncoderWithoutProj

from ..llama.tokenizer import LlamaTokenizer


class LlavaMultiModalProjector(fl.Chain):
    "LLava multimodal vision projector"

    def __init__(
        self,
        in_dim: int,
        feedforward_dim: int,
        out_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            fl.Linear(in_features=in_dim, out_features=feedforward_dim, device=device, dtype=dtype),
            fl.GeLU(),
            fl.Linear(in_features=feedforward_dim, out_features=out_dim, device=device, dtype=dtype),
        )


class LlavaInputEncoder(fl.ContextModule):
    """
    Llava input Encoding Layer.

    This layer encode both the text and the image and set them
    in the right format as an input for the Llava model
    """

    def __init__(
        self,
        image_size: int = 336,
        embedding_dim: int = 1024,
        patch_size: int = 14,
        num_layers: int = 24,
        num_attention_heads: int = 16,
        feedforward_dim: int = 4096,
        mm_feedforward_dim: int = 4096,
        layer_norm_eps: float = 1e-5,
        vocab_size: int = 32064,
        image_token_index: int = 32000,
        max_sequence_length: int = 16_384,
        tokenizer: LlamaTokenizer | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.image_token_index = image_token_index
        self.patch_size = patch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.image_size = image_size

        # store scale of the image during rescaling for bounding box coordinates handling
        # at generation time. Purpose: rescaling the coordinates at the original
        # image size once the generation is over.
        self.scale: float = 1

        self.tokenizer = tokenizer or LlamaTokenizer()

        self.text_token_encoder = fl.Embedding(vocab_size, embedding_dim, device, dtype)

        self.vision_tower = CLIPImageEncoderWithoutProj(
            image_size=image_size,
            embedding_dim=embedding_dim,
            patch_size=patch_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            feedforward_dim=feedforward_dim,
            layer_norm_eps=layer_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.mm_projector = LlavaMultiModalProjector(
            in_dim=embedding_dim, feedforward_dim=mm_feedforward_dim, out_dim=mm_feedforward_dim
        )

    def _merge_input_ids_with_image_features(
        self,
        image_features: Tensor,
        inputs_embeds: Tensor,
        input_ids: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Merge input IDs with image features.

        Args:
            image_features: Image features tensor.
            inputs_embeds: Input embeddings tensor.
            input_ids: Input IDs tensor.
            attention_mask: Attention mask tensor.

        Returns:
            final_embedding: Merged embedding tensor.
            final_attention_mask: Merged attention mask tensor.
            position_ids: Position IDs tensor.
        """
        attention_mask = torch.ones_like(input_ids)
        _, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))

        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        max_embed_dim = int(max_embed_dim.item())
        batch_indices, non_image_indices = torch.where(input_ids != self.image_token_index)

        # 2. Compute the positions where text should be written
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.empty(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.empty(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )

        # 4. Fill the embeddings based on the mask
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images
        image_to_overwrite = torch.full(
            size=(batch_size, max_embed_dim), fill_value=True, dtype=torch.bool, device=inputs_embeds.device
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None]

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError("Inconsistent image token count")

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]
        final_embedding[batch_indices, indices_to_mask] = 0

        return final_embedding, final_attention_mask, position_ids

    def forward(self, prompt: str, image: Tensor | None) -> Tensor:
        """
        Processes and encodes optional image and text data to create model inputs for the Llava model.

        Receives:
            prompt (str): A single prompt. This is the prompt that the model will use to generate a response.
            images (Tensor | None): An optional image tensor. if provided, should be in the format
                [1, C, H, W] where C is the number of channels, H is height, and W is width.

        Returns:
            Returns:
            (Float[Tensor, "1, seq_len, embedding_dim"])
            A tensor containing the encoded inputs for the model. This includes encoded image data and
            text data concatenated along the sequence dimension, suitable for input into the Llava model.

        Raises:
            ValueError: If the lengths of the images and prompts lists do not match or if the sequence length of the input
                is over  the max_sequence_length defined.
        """

        def _process_image(image: Tensor) -> Tuple[Tensor, float]:
            """
            Prepare the image for Llava model.

            Receives:
                image (Tensor): Image tensor in the format [1, C, H, W].

            Returns:
                (Float[Tensor, "1 C H W"], float)
                A tensor with the processed image and its corresponding scale_factor.
            """

            _, c, h, w = image.shape
            # if images are b&w duplicate on rgb channels
            if c == 1:
                image = torch.cat([image] * 3, dim=1)
            scale_factor = 1
            # if images are above the image size limit rescale them
            if h > self.image_size or w > self.image_size:
                scale_factor = min(self.image_size / h, self.image_size / w)
                image = image.squeeze(0) * 255
                image = image.byte().numpy().transpose(1, 2, 0)
                image_pil = Image.fromarray(image, "RGB")  # type: ignore[reportUnknownType]
                image_pil = image_pil.resize((int(scale_factor * w), int(scale_factor * h)), Image.Resampling.BILINEAR)  # type: ignore[reportUnknownType]
                # Convert back to tensor
                transform = transforms.ToTensor()
                image = transform(image_pil).unsqueeze(0)
                _, _, h, w = image.shape

            # for bboxs and points handling the scale_factor information need to be saved
            return image, scale_factor

        # preprocess image if available
        if image:
            pixel_values, scale = _process_image(image)
            self.scale = scale
        else:
            pixel_values = None

        # Encode text:
        prompt_token = self.tokenizer(prompt, scale_factor=self.scale)

        # 1. Extract the tokenized text input embeddings
        inputs_embeds = self.text_token_encoder(prompt_token)

        # 2. Merge text and images
        if pixel_values is not None and prompt_token.shape[1] != 1:
            image_outputs = self.vision_tower(pixel_values)
            selected_image_feature = image_outputs[:, 1:]
            image_features = self.mm_projector(selected_image_feature)
            inputs_embeds, _, _ = self._merge_input_ids_with_image_features(image_features, inputs_embeds, prompt_token)

        if inputs_embeds.shape[1] > self.max_sequence_length:
            raise ValueError("The max sequence length is reached.")
        return inputs_embeds
