import re
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from PIL import Image
from torch import Tensor, device as Device, dtype as DType

import refiners.fluxion.layers as fl
from refiners.fluxion.utils import no_grad
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

    def generate(self, images: List[Image.Image], prompts: List[str], max_len_generation: int = 50):
        """
        Generate answers for a list of images and prompts. Inference by batch,
        rescale image if they are over self.InputEncoder.max_size

        Receives:
            images (List[Image.Image, "batch"])
            prompts (List[str, "batch"])
            max_len_generation (int)

        Returns:
            (List[str, "batch"])
        """
        tensor_images = [(Tensor(np.array(image) / 255)).permute(2, 0, 1).unsqueeze(0) for image in images]

        i = 0
        answers = [""] * len(tensor_images)
        final_answers = [""] * len(tensor_images)

        active_indices = list(range(len(tensor_images)))
        active_answers = None
        active_in_coords = [False] * len(tensor_images)
        scales_list = [1] * len(tensor_images)

        tokenizer = self.tokenizer[0]

        with no_grad():
            while i < max_len_generation and len(active_indices) > 0:
                active_images = [tensor_images[idx] for idx in active_indices]
                active_prompts = [prompts[idx] for idx in active_indices]
                predictions = self.forward(active_images, active_prompts, active_answers)

                # Get scales of all the images of the initial batch
                if i == 0:
                    scales_list = self.input_encoder[0].scales_list

                next_tokens = torch.argmax(predictions[:, -1, :], dim=-1)

                to_remove: List[int] = []
                for idx, next_token in enumerate(next_tokens):
                    token_id = int(next_token.item())
                    # end of generation
                    if token_id == tokenizer.eos_token["id"]:
                        final_answers[active_indices[idx]] = answers[active_indices[idx]]
                        to_remove.append(active_indices[idx])
                        next_token_text = ""

                    # the model begins to generate coordinates
                    elif token_id in [tokenizer.token_bbox_open_id, tokenizer.token_point_open_id]:
                        next_token_text = tokenizer.id_to_token[token_id]
                        next_token_text = next_token_text.replace(tokenizer.token_bbox_open, tokenizer.text_bbox_open)
                        next_token_text = next_token_text.replace(tokenizer.token_point_open, tokenizer.text_point_open)
                        active_in_coords[active_indices[idx]] = True

                    # the model ends coordinates generation
                    elif token_id in [tokenizer.token_bbox_close_id, tokenizer.token_point_close_id]:
                        next_token_text = tokenizer.id_to_token[token_id]
                        next_token_text = next_token_text.replace(tokenizer.token_bbox_close, tokenizer.text_bbox_close)
                        next_token_text = next_token_text.replace(
                            tokenizer.token_point_close, tokenizer.text_point_close
                        )
                        # remove last comma
                        answers[active_indices[idx]] = answers[active_indices[idx]][:-1]
                        active_in_coords[active_indices[idx]] = False

                    else:
                        # basic processing
                        if not active_in_coords[active_indices[idx]]:
                            next_token_text = tokenizer.id_to_token[token_id].replace(
                                tokenizer.replace_char, tokenizer.replace_pattern
                            )
                            next_token_text = next_token_text.replace(tokenizer.newline_model_token, "\n")
                            # avoid starting the sentence with a space
                            if i == 0 and next_token_text[0] == " ":
                                next_token_text = next_token_text[1:]
                        # coordinates processing
                        else:
                            next_token_text = tokenizer.id_to_token[token_id]
                            next_token_text += ","

                    if answers[active_indices[idx]] == "":
                        answers[active_indices[idx]] = next_token_text
                    else:
                        answers[active_indices[idx]] += next_token_text

                # Remove the indices that have reached the EOS token.
                for idx in reversed(to_remove):  # Reverse to avoid index shifting issues.
                    active_indices.remove(idx)

                active_answers = [answers[idx] for idx in active_indices]
                i += 1

            # For any prompts that did not reach EOS, set their final answer now.
            for idx in active_indices:
                final_answers[idx] = answers[idx]

            regex_pattern = re.compile(
                f"({tokenizer.text_bbox_open}|{tokenizer.text_bbox_close}|{tokenizer.text_point_open}|{tokenizer.text_point_close})"
            )

        # Rescale answers coordinates to the original image size
        for idx, answer in enumerate(final_answers):
            answer_split = regex_pattern.split(answer)
            final_answer = ""
            for i, elem in enumerate(answer_split):
                if i > 0 and answer_split[i - 1] in [tokenizer.text_bbox_open, tokenizer.text_point_open]:
                    points_coordinates = elem.split(",")
                    points_coordinates = [
                        float(point_coordinate.strip())
                        for point_coordinate in points_coordinates
                        if point_coordinate.strip() != ""
                    ]
                    points_coordinates = [
                        str(np.round((point_coordinate / scales_list[idx]) * 2).astype(int))
                        for point_coordinate in points_coordinates
                    ]
                    final_answer += ",".join(points_coordinates)
                else:
                    final_answer += elem
            final_answers[idx] = final_answer
        return final_answers


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
