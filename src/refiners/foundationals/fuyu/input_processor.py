from typing import Tuple

import torch
from PIL import Image
from torch import Tensor, device as Device, dtype as DType, tensor
from torchvision.transforms.functional import to_tensor  # type: ignore[reportUnknownVariableType]

import refiners.fluxion.layers as fl
from refiners.foundationals.clip.text_encoder import TokenEncoder
from refiners.foundationals.fuyu.common import PatchPadding
from refiners.foundationals.fuyu.tokenizer import FuyuTokenizer


class ImageEncoder(fl.Chain):
    """
    Encodes an input tensor of images through padding, normalization, patchification, and projection.

    Prepares image tensors for further processing by padding them to ensure they are divisible by the patch size,
    normalizing them based on provided mean and standard deviation values, breaking them down into patches, and finally
    projecting these patches into a specified embedding dimension.

    Args:
        patch_size (int): The size of the square patch to divide images into.
        embedding_dim (int): The dimension of the output embeddings after projection.
        padding_value (float): The value used for padding the images.
        image_mean (float): The mean value used for normalizing the images.
        image_std (float): The standard deviation used for normalizing the images.
        use_bias (bool): Whether to use bias in the linear layers.
        device (Device | str | None): The device on which the operations should be performed.
        dtype (DType | None): The data type to use for the operations.
    """

    def __init__(
        self,
        patch_size: int = 30,
        embedding_dim: int = 4096,
        padding_value: float = 1.0,
        image_mean: float = 0.5,
        image_std: float = 0.5,
        use_bias: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.padding_value = padding_value
        self.image_mean = image_mean
        self.image_std = image_std

        super().__init__(
            PatchPadding(patch_size=self.patch_size, padding_value=self.padding_value),
            fl.Lambda(func=self.normalize),
            fl.Lambda(func=self.patchify),
            fl.Linear(
                in_features=3 * self.patch_size**2,
                out_features=self.embedding_dim,
                bias=use_bias,
                device=device,
                dtype=dtype,
            ),
        )

    def patchify(self, x: Tensor) -> Tensor:
        """
        Transforms an image tensor into a set of flattened, non-overlapping patches.
        """
        _, c, _, _ = x.shape
        x_unfold = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        B, _, _, _, _, _ = x_unfold.shape
        x_unfold = x_unfold.contiguous()
        x_unfold = x_unfold.view(B, c, -1, self.patch_size, self.patch_size)
        patched = x_unfold.permute(0, 2, 3, 4, 1).reshape(B, -1, c * self.patch_size**2)
        return patched

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.image_mean) / self.image_std


class InputEncoder(fl.ContextModule):
    """
    Input Encoding Layer.

    This layer encode both the text and the image and set them
    in the right format as an input for the Fuyu model
    """

    def __init__(
        self,
        tokenizer: FuyuTokenizer,
        embedding_dim: int = 4096,
        max_sequence_length: int = 16_384,
        vocabulary_size: int = 262_144,
        patch_size: int = 30,
        padding_value: float = 1.0 / 255,
        max_size: Tuple[int, int] = (1080, 1920),  # h w
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.patch_size = patch_size
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.padding_value = padding_value
        self.device = device
        self.dtype = dtype
        self.max_size = max_size

        # store scales of the different images during rescaling of a given batch for bounding box coordinates handling
        # at generation time. Purpose: rescaling the coordinates at the original image size once the generation is over.
        self.scales_list: list[float] = []

        self.tokenizer = tokenizer

        self.token_encoder = TokenEncoder(
            vocabulary_size=self.vocabulary_size, embedding_dim=self.embedding_dim, device=self.device, dtype=self.dtype
        )

        self.image_encoder = ImageEncoder(
            patch_size=self.patch_size,
            embedding_dim=self.embedding_dim,
            padding_value=self.padding_value,
            device=device,
            dtype=dtype,
        )

    def forward(self, images: list[Tensor], prompts: list[str], answers: list[str] | None = None) -> Tensor:
        """
        Processes and encodes image and text data to create model inputs for the Fuyu model.

        Takes a batch of images and corresponding text prompts, and optionally text answers,
        processes and encodes them into a uniform tensor format suitable for input to the Fuyu model.
        Handles image resizing, padding, tokenization and encoding of text. Additionally,
        it generates attention masks for handling variable sequence lengths in batch processing.

        Receives:
            images (list[Tensor]): A list of image tensors, each tensor should be in the format [1, C, H, W]
                where C is the number of channels, H is height, and W is width.
            prompts (list[str]): A list of text strings, each corresponding to an image. These are prompts
                that the model will use to generate responses.
            answers (list[str] | None, optional): An optional list of text strings providing answers
                corresponding to each prompt for continuing the sequential process of generation.
                If provided, it is used along with the prompts for input encoding.
                Defaults to None
        Returns:
            Returns:
            (Float[Tensor, "batch seq_len embedding_dim"])
            A tensor containing the encoded inputs for the model. This includes encoded image data and
            text data concatenated along the sequence dimension, suitable for input into the Fuyu model.

        Raises:
            ValueError: If the lengths of the images and prompts lists do not match, if answers are provided
                and their length does not match the length of the prompts or if the sequence length of the input
                is over  the max_sequence_length defined.
        """

        if len(images) != len(prompts):
            raise ValueError("The number of images must be equal to the number of prompts.")
        if answers is not None and len(answers) != len(prompts):
            raise ValueError("The number of answers must be equal to the number of prompts if answers are provided.")

        b = len(images)
        # preprocess batch
        images = self.process_batch_images(images)

        # encode images
        encoded_images: list[Tensor] = []
        for image in images:
            _, _, h, w = image.shape
            patched_image = self.image_encoder(image.to(device=self.device, dtype=self.dtype))

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
            encoded_image = torch.cat((encoded_image, linebreak_embedding), dim=2)
            # Reshape to final desired flat format [1 seq_len embedding_dim]
            encoded_image = encoded_image.view(1, -1, self.embedding_dim)
            encoded_images.append(encoded_image)

        # encode texts
        encoded_texts: list[Tensor] = []
        for idx, prompt in enumerate(prompts):
            prompt_token = self.tokenizer(prompt, scale_factor=self.scales_list[idx])
            token = torch.cat(
                [Tensor([[self.tokenizer.bos_token_id]]), prompt_token, Tensor([[self.tokenizer.boa_token_id]])], dim=1
            )
            if answers is not None:
                answer_token = self.tokenizer(answers[idx])
                token = torch.cat([token, answer_token], dim=1)
            encoded_text = self.token_encoder(token.to(device=self.device, dtype=torch.int64))
            encoded_texts.append(encoded_text)

        # Initialize the 3D attention mask with ones
        max_len = max(et.shape[1] + im.shape[1] for et, im in zip(encoded_texts, encoded_images))
        attn_mask = torch.ones(b, 1, max_len, max_len, device=self.device, dtype=torch.bool)

        padded_encoded_images: list[Tensor] = []
        for idx, (encoded_text, encoded_image) in enumerate(zip(encoded_texts, encoded_images)):
            padding_length = max_len - (encoded_text.shape[1] + encoded_image.shape[1])
            if padding_length > 0:
                padding_tensor = tensor([self.tokenizer.pad_token["id"]] * padding_length, device=self.device).long()
                padding_encoding = self.token_encoder(padding_tensor).unsqueeze(0)
                # Concatenate the padding on the left of the encoded image
                padded_encoded_image = torch.cat((padding_encoding, encoded_image), dim=1)
            else:
                # No padding needed, use the encoded image as is
                padded_encoded_image = encoded_image
            padded_encoded_images.append(padded_encoded_image)
            attn_mask[idx, :, :padding_length, :] = 0
            attn_mask[idx, :, :, :padding_length] = 0

        causal_mask = torch.tril(torch.ones((b, 1, max_len, max_len), device=self.device, dtype=torch.bool))
        attn_mask = attn_mask & causal_mask

        context = self.use_context(context_name="attention")
        context.update({"mask": attn_mask})

        encoded_inputs = torch.cat(
            [
                torch.cat((padded_encoded_image, encoded_text), dim=1)
                for padded_encoded_image, encoded_text in zip(padded_encoded_images, encoded_texts)
            ],
            dim=0,
        )

        if encoded_inputs.shape[1] > self.max_sequence_length:
            raise ValueError("The max sequence length is reached.")
        return encoded_inputs

    def process_batch_images(self, images: list[Tensor]) -> list[Tensor]:
        """
        Processes a batch of image tensors: ensuring all images have three channels,
        resizing images that exceed max dimensions, and padding all images to have uniform dimensions.

        Receives:
            images (list[Tensor]): list of image tensors in the format [1, C, H, W].

        Returns:
            (Float[Tensor, "batch C H W"])
            A batch tensor with all processed images concatenated along the batch dimension.
        """
        # for bboxs and points handling these information need to be saved
        scales_list: list[float] = []
        for im_idx, image in enumerate(images):
            _, c, h, w = image.shape

            # if images are b&w duplicate on rgb channels
            if c == 1:
                image = torch.cat([image] * 3, dim=1)

            scale_factor = 1
            # if images are above the max size limit rescale them
            if h > self.max_size[0] or w > self.max_size[1]:
                scale_factor = min(self.max_size[0] / h, self.max_size[1] / w)
                image = Image.fromarray((image.squeeze(0) * 255).byte().numpy().transpose(1, 2, 0), "RGB")  # type: ignore[reportUnknownType]
                image = image.resize((int(scale_factor * w), int(scale_factor * h)), Image.Resampling.BILINEAR)  # type: ignore[reportUnknownType]
                image = to_tensor(image).unsqueeze(0)
                _, _, h, w = image.shape
            scales_list.append(scale_factor)

            images[im_idx] = image

        self.scales_list = scales_list

        return images
