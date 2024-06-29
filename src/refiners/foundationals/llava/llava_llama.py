import re

import numpy as np
import torch
from PIL import Image

import refiners.fluxion.layers as fl
from refiners.fluxion.utils import image_to_tensor, no_grad

from ..llama.llama import LlamaModelWoEmbedding
from ..llama.tokenizer import LlamaTokenizer
from .input_processor import LlavaInputEncoder


class LlavaLlama(fl.Chain):
    "Inspired from the the HuggingFace 'llava-hf/llava-1.5-7b-hf' model."

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
        text_num_layers: int = 32,
        max_position_embeddings: int = 4096,
        text_feedforward_dim: int = 11008,
        text_num_att_heads: int = 32,
        text_num_kv_heads: int = 32,
        vocab_size: int = 32064,
        text_layer_norm_eps: float = 1e-5,
        image_token_index: int = 32000,
        max_sequence_length: int = 16_384,
        tokenizer: LlamaTokenizer = LlamaTokenizer(),
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.tokenizer = tokenizer
        self.input_encoder = LlavaInputEncoder(
            image_size,
            embedding_dim,
            patch_size,
            num_layers,
            num_attention_heads,
            feedforward_dim,
            mm_feedforward_dim,
            layer_norm_eps,
            vocab_size,
            image_token_index,
            max_sequence_length,
            self.tokenizer,
            device,
            dtype,
        )

        super().__init__(
            self.input_encoder,
            LlamaModelWoEmbedding(
                dim=mm_feedforward_dim,
                n_layers=text_num_layers,
                max_position_embeddings=max_position_embeddings,
                feedforward_dim=text_feedforward_dim,
                n_att_heads=text_num_att_heads,
                n_kv_heads=text_num_kv_heads,
                vocab_size=vocab_size,
                norm_eps=text_layer_norm_eps,
                device=device,
                dtype=dtype,
            ),
            fl.Linear(in_features=mm_feedforward_dim, out_features=vocab_size, bias=False, device=device, dtype=dtype),
        )

    def generate(self, prompt: str, image: Image.Image | None) -> str:
        """
        Generate answer for a prompt and an optional corresponding image.
        Receives:
            prompts (str)
            image (Image.Image)
        Returns:
            (str)
        """

        scale = 1.0
        answer = ""
        if not image:
            with no_grad():
                predictions = self.forward(prompt, None)
        else:
            tensor_image = image_to_tensor(image)
            # add batch_dimension and set it to 1
            tensor_image.unsqueeze(0)
            with no_grad():
                predictions = self.forward(prompt, image)

        # Get scale of the image
        scale = self.input_encoder.scale
        next_tokens = torch.argmax(predictions[:, -1, :], dim=-1)
        self.process_next_tokens(next_tokens, answer)
        final_answer = self.rescale_answer(answer, scale)

        return final_answer

    def process_next_tokens(
        self,
        next_tokens: torch.Tensor,
        answer: str,
    ) -> None:
        """
        Process a batch of token ids, update the current answers and active lists InPlace.
        Receives:
            next_tokens (Int[Tensor, "active_batch"])
            answers (List[str, "batch"])
            active_indices (List[int, "active_batch"])
            active_in_coords (List[bool, "batch"])
        """
        tokenizer = self.tokenizer
        active_in_coords = False
        for _, next_token in enumerate(next_tokens):
            token_id = int(next_token.item())
            # End of generation
            if token_id == tokenizer.eos_token["id"]:
                next_token_text = ""

            # The model begins to generate coordinates
            elif token_id in [tokenizer.token_bbox_open_id, tokenizer.token_point_open_id]:
                next_token_text = tokenizer.id_to_token[token_id]
                next_token_text = next_token_text.replace(tokenizer.token_bbox_open, tokenizer.text_bbox_open)
                next_token_text = next_token_text.replace(tokenizer.token_point_open, tokenizer.text_point_open)
                active_in_coords = True

            # The model ends coordinates generation
            elif token_id in [tokenizer.token_bbox_close_id, tokenizer.token_point_close_id]:
                next_token_text = tokenizer.id_to_token[token_id]
                next_token_text = next_token_text.replace(tokenizer.token_bbox_close, tokenizer.text_bbox_close)
                next_token_text = next_token_text.replace(tokenizer.token_point_close, tokenizer.text_point_close)

            else:
                # Basic processing
                if not active_in_coords:
                    next_token_text = tokenizer.id_to_token[token_id].replace(
                        tokenizer.replace_char, tokenizer.replace_pattern
                    )
                    next_token_text = next_token_text.replace(tokenizer.newline_model_token, "\n")

                # Coordinates processing
                else:
                    next_token_text = tokenizer.id_to_token[token_id]
                    next_token_text += ","

            if answer == "":
                answer = (
                    next_token_text[1:] if next_token_text[0] == " " else next_token_text
                )  # Avoid starting the sentence with a space
            else:
                answer += next_token_text

    def rescale_answer(self, answer: str, scale: float) -> str:
        """
        Rescale the coordinates within a list of model answers using the list of scales provided.
        Receives:
            answers (str)
            scales (float)
        Returns:
            (str)
        """
        tokenizer = self.tokenizer

        regex_pattern = re.compile(
            f"({tokenizer.text_bbox_open}|{tokenizer.text_bbox_close}|{tokenizer.text_point_open}|{tokenizer.text_point_close})"
        )

        rescaled_answer = ""

        answer_split = regex_pattern.split(answer)
        rescaled_answer = ""
        for i, elem in enumerate(answer_split):
            if i > 0 and answer_split[i - 1] in [tokenizer.text_bbox_open, tokenizer.text_point_open]:
                points_coordinates = elem.split(",")
                points_coordinates = [
                    float(point_coordinate.strip())
                    for point_coordinate in points_coordinates
                    if point_coordinate.strip() != ""
                ]
                points_coordinates = [
                    str(np.round((point_coordinate / scale) * 2).astype(int)) for point_coordinate in points_coordinates
                ]
                rescaled_answer += ",".join(points_coordinates)
            else:
                rescaled_answer += elem
        return rescaled_answer
