from dataclasses import dataclass
from typing import Any, List

import numpy as np
from torch import Tensor, argmax, device as Device, dtype as DType, float16 as Tfloat16, no_grad

import refiners.fluxion.layers as fl
from refiners.foundationals.fuyu.input_processor import InputEncoder
from refiners.foundationals.fuyu.tokenizer import FuyuTokenizer
from refiners.foundationals.fuyu.transformers import FuyuTransformer, FuyuTransformerLayer


def create_fuyu(config):
    """
    create a fuyu model based on the config provided

    Example:
        ```py
        config = Fuyu8b
        network = create_fuyu(config)
        ```
    """
    model = Fuyu(
        embedding_dim=config.embedding_dim,
        feedforward_dim=config.feedforward_dim,
        max_sequence_length=config.max_sequence_length,
        vocabulary_size=config.vocabulary_size,
        tokenizer=config.tokenizer,
        patch_size=config.patch_size,
        padding_value=config.padding_value,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        norm_eps=config.norm_eps,
        base=config.base,
        partial_rotary_factor=config.partial_rotary_factor,
        use_bias=config.use_bias,
        is_optimized=config.is_optimized,
        device=config.device,
        dtype=config.dtype
    )
    model.eval()
    return(model)

@dataclass(frozen=True)
class Fuyu8b:
    """
    config with the base argument for Fuyu8b
    """
    embedding_dim: int = 4_096
    feedforward_dim: int = 16_384
    max_sequence_length: int = 16_384
    vocabulary_size: int = 262_144
    tokenizer: FuyuTokenizer | None = FuyuTokenizer()
    patch_size: int = 30
    padding_value: float = 1.0/255
    num_layers: int = 36
    num_heads: int = 64
    norm_eps: float = 1e-5
    base: int = 25_000
    partial_rotary_factor: float = 0.5
    use_bias: bool = True
    is_optimized: bool = True
    device: Device | str | None = 'cuda'
    dtype: DType | None = Tfloat16

class Fuyu(fl.Chain):
    """
    Implements the Fuyu model
    see [https://www.adept.ai/blog/fuyu-8b]
    """
    def __init__(
        self,
        embedding_dim: int,
        feedforward_dim: int,
        max_sequence_length: int,
        vocabulary_size: int,
        tokenizer: FuyuTokenizer | None,
        patch_size: int,
        padding_value: int,
        num_layers: int,
        num_heads: int,
        norm_eps: float,
        base: int,
        partial_rotary_factor:float,
        use_bias: bool,
        is_optimized: bool,
        device: Device | str | None,
        dtype: DType | None
    ) -> None:
        super().__init__(
            InputEncoder(
                embedding_dim=embedding_dim,
                max_sequence_length=max_sequence_length,
                vocabulary_size=vocabulary_size,
                tokenizer=tokenizer,
                patch_size=patch_size,
                padding_value=padding_value,
                device=device,
                dtype=dtype
            ),
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
                    dtype=dtype
                )
                for _ in range(num_layers)
            ),
            fl.LayerNorm( 
                normalized_shape=embedding_dim,
                eps=norm_eps,
                device=device,
                dtype=dtype
            ),
            fl.Linear(
                in_features=embedding_dim,
                out_features=vocabulary_size,
                bias=False,
                device=device,
                dtype=dtype
            )
        )
    def init_context(self) -> dict[str, dict[str, Any]]:
        return {"attention": {"mask": None}}
    
    def generate(self, images: List, prompts: List[str], max_len_generation=50):
        """
        Generate answers for a list of images and prompts. Doesn't infer by batch 
        to preserve the native resolution of each image

        Receives:
            images (List[PIL.Image, "batch"])
            prompts (List[str, "batch"])
            max_len_generation (int)

        Returns:
            (List[str, "batch"])
        """
        tokenizer = self.InputEncoder.tokenizer
        final_answer = []
        for image, prompt in zip(images, prompts):
            image=(Tensor(np.array(image)/255)).permute(2,0,1).unsqueeze(0)
            i = 0
            answer = None
            stop_token = False
            while i<max_len_generation and not stop_token:
                with no_grad():
                    predictions = self.forward([image], [prompt], [answer] if answer is not None else answer)
                next_token = argmax(predictions[:,-1,:], dim=-1)
                if next_token.item() != tokenizer.eos_token['id']:
                    next_token_text = tokenizer.id_to_token[next_token.item()].replace(tokenizer.replace_char, tokenizer.replace_pattern)
                    next_token_text = next_token_text.replace(tokenizer.newline_model_token, '\n')
                    if answer is None:
                        answer = next_token_text
                    else:
                        answer += next_token_text
                    i+=1
                else :
                    stop_token=True
            final_answer.append(answer)
        return final_answer
    
    def batched_generate(self, images: List, prompts: list[str], max_len_generation=50):
        """
        Generate answers for a list of images and prompts. Inference by batch,
        rescale image if they are over self.InputEncoder.max_size, pad other images 
        to the largest image size

        Receives:
            images (List[PIL.Image, "batch"])
            prompts (List[str, "batch"])
            max_len_generation (int)

        Returns:
            (List[str, "batch"])
        """
        tokenizer = self.InputEncoder.tokenizer
        tensor_images = [(Tensor(np.array(image)/255)).permute(2,0,1).unsqueeze(0) for image in images]

        i = 0
        answers = [None] * len(tensor_images)
        final_answer = [''] * len(tensor_images)
        active_indices = list(range(len(tensor_images)))
        active_answers = None
        while i<max_len_generation and len(active_indices)>0:
            active_images = [tensor_images[idx] for idx in active_indices]
            active_prompts = [prompts[idx] for idx in active_indices]

            with no_grad():
                predictions = self.forward(active_images, active_prompts, active_answers)
            next_tokens = argmax(predictions[:,-1,:], dim=-1)

            to_remove = []
            for idx, next_token in enumerate(next_tokens):
                if next_token.item() != tokenizer.eos_token['id']:
                    next_token_text = tokenizer.id_to_token[next_token.item()].replace(tokenizer.replace_char, tokenizer.replace_pattern)
                    next_token_text = next_token_text.replace(tokenizer.newline_model_token, '\n')
                    if answers[active_indices[idx]] is None:
                        answers[active_indices[idx]] = next_token_text
                    else:
                        answers[active_indices[idx]] += next_token_text
                else:
                    final_answer[active_indices[idx]] = answers[active_indices[idx]]
                    to_remove.append(active_indices[idx])

            # Remove the indices that have reached the EOS token.
            for idx in reversed(to_remove):  # Reverse to avoid index shifting issues.
                active_indices.remove(idx)
            
            active_answers = [answers[idx] for idx in active_indices]
            i+=1

        # For any prompts that did not reach EOS, set their final answer now.
        for idx in active_indices:
            final_answer[idx] = answers[idx]

        return final_answer
