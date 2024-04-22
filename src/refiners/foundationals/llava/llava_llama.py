import torch

import refiners.fluxion.layers as fl

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
