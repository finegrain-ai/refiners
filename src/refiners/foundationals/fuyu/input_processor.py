from typing import List, Tuple

from torch import Tensor, bool as tbool, cat, device as Device, dtype as DType, ones, tensor, tril
from torchvision.transforms.functional import pad, resize

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

class InputEncoder(fl.ContextModule):
    """
    Input Encoding Layer.

    This layer encode both the text and the image and set them 
    in the right format as an input for the Fuyu model
    """
    def __init__(
        self,
        embedding_dim: int = 4096,
        max_sequence_length: int = 16_384,
        vocabulary_size: int = 262_144,
        tokenizer: FuyuTokenizer | None = None,
        patch_size: int = 30,
        padding_value: float = 1.,
        max_size: Tuple[int] = (1080, 1920), #h w
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

    def forward(self, images: List[Tensor], prompts: list[str], answers: list[str] | None = None) -> Tensor:
        """
        Processes and encodes image and text data to create model inputs for the Fuyu model.
        
        Takes a batch of images and corresponding text prompts, and optionally text answers, 
        processes and encodes them into a uniform tensor format suitable for input to the Fuyu model. 
        Handles image resizing, padding, tokenization and encoding of text. Additionally, 
        it generates attention masks for handling variable sequence lengths in batch processing.

        Receives:
            images (List[Tensor]): A list of image tensors, each tensor should be in the format [C, H, W] 
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

        # preprocess batch
        images = self.process_batch_images(images)
        images = images.to(device=self.device, dtype=self.dtype)
        b, c, h, w = images.shape
        
        # Encode Images
        patched_images = self.image_encoder(images)

        h += (self.patch_size - h % self.patch_size) % self.patch_size
        w += (self.patch_size - w % self.patch_size) % self.patch_size
        
        f_linebreak = w // self.patch_size
        n_linebreak = h // self.patch_size
        # Create linebreak embeddings
        linebreak = tensor([self.tokenizer.newline_token_id], device=self.device).long()
        linebreak_embedding = self.token_encoder(linebreak)
        linebreak_embedding = linebreak_embedding.expand(b, n_linebreak, 1, self.embedding_dim)
        # Reshape encoded_image to introduce a slot for linebreaks
        encoded_images = patched_images.view(b, n_linebreak, f_linebreak, self.embedding_dim)
        # Concatenate linebreak embeddings
        encoded_images = cat((encoded_images, linebreak_embedding), dim=2)
        # Reshape to final desired flat format [b seq_len embedding_dim]
        encoded_images = encoded_images.view(b, -1, self.embedding_dim)
        
        tokens = []
        if answers is not None:
            for prompt, answer in zip(prompts, answers): 
                token = cat(
                    [
                        Tensor([[self.tokenizer.bos_token_id]]).to(self.device),
                        self.tokenizer(prompt).to(self.device),
                        Tensor([[self.tokenizer.boa_token_id]]).to(self.device),
                        self.tokenizer(answer).to(self.device)
                    ],
                    dim=1,
                ).to(int)
                tokens.append(token)
        else:
            for prompt in prompts:
                token = cat(
                    [
                        Tensor([[self.tokenizer.bos_token_id]]).to(self.device),
                        self.tokenizer(prompt).to(self.device),
                        Tensor([[self.tokenizer.boa_token_id]]).to(self.device)
                    ],
                    dim=1,
                ).to(int)
                tokens.append(token)

        # Tokenize and encode text
        encoded_texts = [self.token_encoder(token) for token in tokens]
         # Initialize the 3D attention mask with ones
        max_text_len = max(et.shape[1] for et in encoded_texts)
        max_len = max_text_len + encoded_images.shape[1]
        attn_mask = ones(b, max_len, max_len, device=self.device, dtype=tbool)

        padded_encoded_images = []
        for idx, encoded_text in enumerate(encoded_texts):
            padding_length = max_text_len - encoded_text.shape[1]
            if padding_length > 0:
                padding_tensor = tensor([self.tokenizer.pad_token['id']] * padding_length, device=self.device).long()
                padding_encoding = self.token_encoder(padding_tensor).view(1, padding_length, -1)
                # Concatenate the padding on the left of the encoded image
                padded_encoded_image = cat((padding_encoding, encoded_images[idx].unsqueeze(0)), dim=1)
            else:
                # No padding needed, use the encoded image as is
                padded_encoded_image = encoded_images[idx].unsqueeze(0)
            padded_encoded_images.append(padded_encoded_image)
            attn_mask[idx, :padding_length, :] = 0
            attn_mask[idx, :, :padding_length] = 0

        causal_mask = tril(ones((b, max_len, max_len), device=self.device, dtype=tbool))
        attn_mask = attn_mask & causal_mask

        context = self.use_context(context_name="attention")
        context.update({"mask": attn_mask})

        encoded_inputs = cat([cat((padded_encoded_image, encoded_text), dim=1) for padded_encoded_image, encoded_text in zip(padded_encoded_images, encoded_texts)], dim=0)

        if encoded_inputs.shape[1] > self.max_sequence_length:
            raise ValueError("The max sequence length is reached.")
        return encoded_inputs
    
    def process_batch_images(self, images: List[Tensor]) -> List[Tensor]:
        """
        Processes a batch of image tensors: ensuring all images have three channels,
        resizing images that exceed max dimensions, and padding all images to have uniform dimensions.

        Receives:
            images (List[Tensor]): List of image tensors in the format [C, H, W].

        Returns:
            (Float[Tensor, "batch C H W"])
            A batch tensor with all processed images concatenated along the batch dimension.
        """

        max_h, max_w = 0, 0
        for im_idx, image in enumerate(images):
            _, c, h, w = image.shape
            
            # if images are b&w duplicate on rgb channels
            if c == 1:
                image = cat([image] * 3, dim=1)

            # if images are above the max size limite rescale them
            if h > self.max_size[0] or w > self.max_size[1]:
                scale_factor = min(self.max_size[0]/h, self.max_size[1]/w)
                image = resize(image, [int(scale_factor*h), int(scale_factor*w)])
                _, _, h, w = image.shape

            images[im_idx] = image
            max_h = max(max_h, h)
            max_w = max(max_w, w)
        
        # padd images to max width and height
        for im_idx, image in enumerate(images):
            _, _, h, w = image.shape
            # Calculate padding
            pad_top = (max_h - h) // 2
            pad_bottom = max_h - h - pad_top
            pad_left = (max_w - w) // 2
            pad_right = max_w - w - pad_left

            # Apply padding
            image = pad(image, padding=(pad_left, pad_top, pad_right, pad_bottom), fill=self.padding_value, padding_mode="constant")
            images[im_idx] = image
        
        images = cat(images, dim=0)
        return images
