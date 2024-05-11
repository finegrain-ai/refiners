import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, device as Device, dtype as DType

import refiners.fluxion.layers as fl
from refiners.fluxion.utils import image_to_tensor, no_grad
from refiners.foundationals.fuyu.input_processor import InputEncoder
from refiners.foundationals.fuyu.tokenizer import FuyuTokenizer
from refiners.foundationals.fuyu.transformers import FuyuTransformer, FuyuTransformerLayer


class Fuyu(fl.Chain):
    """
    Initialize a Fuyu model

    Args:
        embedding_dim: The dimension of the embedding.
        feedforward_dim: The dimension of the feedforward layer.
        max_sequence_length: The maximum length of the input sequences.
        vocabulary_size: The size of the tokenizer vocabulary.
        tokenizer: The tokenizer used.
        patch_size: The size of the patches.
        padding_value: The value used for padding inputs.
        num_layers: The number of layers.
        num_heads: The number of heads.
        norm_eps: The epsilon used in Layer Norms.
        base: The base used in RoPE.
        partial_rotary_factor: The factor for partial rotary in RoPE.
        use_bias: Whether to use bias in the linear layers.
        is_optimized: Use of optimized attention.
        device: The PyTorch device to use.
        dtype: The PyTorch data type to use.
    """

    def __init__(
        self,
        embedding_dim: int,
        feedforward_dim: int,
        max_sequence_length: int,
        vocabulary_size: int,
        tokenizer: FuyuTokenizer,
        patch_size: int,
        padding_value: float,
        num_layers: int,
        num_heads: int,
        norm_eps: float,
        base: int,
        partial_rotary_factor: float,
        use_bias: bool,
        is_optimized: bool,
        device: Device | str | None,
        dtype: DType | None,
    ) -> None:
        self.tokenizer = [tokenizer]
        self.input_encoder = [
            InputEncoder(
                embedding_dim=embedding_dim,
                max_sequence_length=max_sequence_length,
                vocabulary_size=vocabulary_size,
                tokenizer=self.tokenizer[0],
                patch_size=patch_size,
                padding_value=padding_value,
                device=device,
                dtype=dtype,
            )
        ]

        super().__init__(
            self.input_encoder[0],
            FuyuTransformer(
                FuyuTransformerLayer(
                    embedding_dim=embedding_dim,
                    feedforward_dim=feedforward_dim,
                    num_heads=num_heads,
                    norm_eps=norm_eps,
                    base=base,
                    partial_rotary_factor=partial_rotary_factor,
                    use_bias=use_bias,
                    is_optimized=is_optimized,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ),
            fl.LayerNorm(normalized_shape=embedding_dim, eps=norm_eps, device=device, dtype=dtype),
            fl.Linear(in_features=embedding_dim, out_features=vocabulary_size, bias=False, device=device, dtype=dtype),
        )

    def init_context(self) -> dict[str, dict[str, Any]]:
        return {"attention": {"mask": None}}

    def generate(self, images: list[Image.Image], prompts: list[str], max_len_generation: int = 50) -> list[str]:
        """
        Generate answers for a list of images and prompts. Inference by batch.

        Receives:
            images (list[Image.Image, "batch"])
            prompts (list[str, "batch"])
            max_len_generation (int)

        Returns:
            (list[str, "batch"])
        """
        tensor_images = [image_to_tensor(image) for image in images]

        i = 0
        # Answers of the model for each prompt in the batch
        answers = [""] * len(tensor_images)
        # Indices of the prompts that are still active i.e haven't reached end of sentence
        active_indices = list(range(len(tensor_images)))
        # Incomplete answers to the currently active prompts, initialized to None
        active_answers = None
        # Indicates if the model is in coordinates generation mode for the given answer
        active_in_coords = [False] * len(tensor_images)
        # Indicates the rescale factor for every image of the batch. Initialized to 0 and
        # updated after the first forward call
        scales_list = [1.0] * len(tensor_images)

        with no_grad():
            while i < max_len_generation and len(active_indices) > 0:
                active_images = [tensor_images[idx] for idx in active_indices]
                active_prompts = [prompts[idx] for idx in active_indices]
                predictions = self.forward(active_images, active_prompts, active_answers)

                # Get scales of all the images of the initial batch
                if i == 0:
                    scales_list = self.input_encoder[0].scales_list

                next_tokens = torch.argmax(predictions[:, -1, :], dim=-1)
                self.process_next_tokens(next_tokens, answers, active_indices, active_in_coords)

                active_answers = [answers[idx] for idx in active_indices]
                i += 1

        final_answers = self.rescale_answers(answers, scales_list)

        return final_answers

    def process_next_tokens(
        self,
        next_tokens: Tensor,
        answers: list[str],
        active_indices: list[int],
        active_in_coords: list[bool],
    ) -> None:
        """
        Process a batch of token ids, update the current answers and active lists InPlace.

        Receives:
            next_tokens (Int[Tensor, "active_batch"])
            answers (list[str, "batch"])
            active_indices (list[int, "active_batch"])
            active_in_coords (list[bool, "batch"])
        """
        tokenizer = self.tokenizer[0]

        to_remove: list[int] = []
        for idx, next_token in enumerate(next_tokens):
            token_id = int(next_token.item())
            # End of generation
            if token_id == tokenizer.eos_token["id"]:
                to_remove.append(active_indices[idx])
                next_token_text = ""

            # The model begins to generate coordinates
            elif token_id in [tokenizer.token_bbox_open_id, tokenizer.token_point_open_id]:
                next_token_text = tokenizer.id_to_token[token_id]
                next_token_text = next_token_text.replace(tokenizer.token_bbox_open, tokenizer.text_bbox_open)
                next_token_text = next_token_text.replace(tokenizer.token_point_open, tokenizer.text_point_open)
                active_in_coords[active_indices[idx]] = True

            # The model ends coordinates generation
            elif token_id in [tokenizer.token_bbox_close_id, tokenizer.token_point_close_id]:
                next_token_text = tokenizer.id_to_token[token_id]
                next_token_text = next_token_text.replace(tokenizer.token_bbox_close, tokenizer.text_bbox_close)
                next_token_text = next_token_text.replace(tokenizer.token_point_close, tokenizer.text_point_close)
                # Remove last comma
                answers[active_indices[idx]] = answers[active_indices[idx]][:-1]
                active_in_coords[active_indices[idx]] = False

            else:
                # Basic processing
                if not active_in_coords[active_indices[idx]]:
                    next_token_text = tokenizer.id_to_token[token_id].replace(
                        tokenizer.replace_char, tokenizer.replace_pattern
                    )
                    next_token_text = next_token_text.replace(tokenizer.newline_model_token, "\n")

                # Coordinates processing
                else:
                    next_token_text = tokenizer.id_to_token[token_id]
                    next_token_text += ","

            if answers[active_indices[idx]] == "":
                answers[active_indices[idx]] = (
                    next_token_text[1:] if next_token_text[0] == " " else next_token_text
                )  # Avoid starting the sentence with a space
            else:
                answers[active_indices[idx]] += next_token_text

        # Remove the indices that have reached the EOS token.
        for idx in reversed(to_remove):  # Reverse to avoid index shifting issues.
            active_indices.remove(idx)

    def rescale_answers(self, answers: list[str], scales: list[float]) -> list[str]:
        """
        Rescale the coordinates within a list of model answers using the list of scales provided.

        Receives:
            answers (list[str, "batch"])
            scales (list[float, "batch"])

        Returns:
            (list[str, "batch"])
        """
        tokenizer = self.tokenizer[0]

        regex_pattern = re.compile(
            f"({tokenizer.text_bbox_open}|{tokenizer.text_bbox_close}|{tokenizer.text_point_open}|{tokenizer.text_point_close})"
        )

        rescaled_answers: list[str] = []

        for idx, answer in enumerate(answers):
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
                        str(np.round((point_coordinate / scales[idx]) * 2).astype(int))
                        for point_coordinate in points_coordinates
                    ]
                    rescaled_answer += ",".join(points_coordinates)
                else:
                    rescaled_answer += elem
            rescaled_answers.append(rescaled_answer)

        return rescaled_answers


class Fuyu8b(Fuyu):
    """
    Fuyu base configuration with 8b parameters
    see [https://www.adept.ai/blog/fuyu-8b]

    Attributes:
        embedding_dim (int): 4_096
        feedforward_dim (int): 16_384
        max_sequence_length (int): 16_384
        vocabulary_size (int): 262_144
        patch_size (int): 30
        padding_value (float): 1.0 / 255
        num_layers (int): 36
        num_heads (int): 64
        norm_eps (float): 1e-5
        base (int): 25_000
        partial_rotary_factor (float): 0.5
        use_bias (bool): True
        is_optimized (bool): False
    """

    def __init__(
        self,
        tokenizer_path: str | Path,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            embedding_dim=4_096,
            feedforward_dim=16_384,
            max_sequence_length=16_384,
            vocabulary_size=262_144,
            tokenizer=FuyuTokenizer(tokenizer_path),
            patch_size=30,
            padding_value=1.0 / 255,
            num_layers=36,
            num_heads=64,
            norm_eps=1e-5,
            base=25_000,
            partial_rotary_factor=0.5,
            use_bias=True,
            is_optimized=False,
            device=device,
            dtype=dtype,
        )
