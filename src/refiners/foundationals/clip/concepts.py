from refiners.adapters.adapter import Adapter
from refiners.foundationals.clip.text_encoder import CLIPTextEncoder, TokenEncoder
from refiners.foundationals.clip.tokenizer import CLIPTokenizer
import refiners.fluxion.layers as fl
from typing import cast

from torch import Tensor, cat, zeros
import torch.nn.functional as F
from torch.nn import Parameter
import re


class ConceptExtender(fl.Chain, Adapter[CLIPTextEncoder]):
    """
    Extends the vocabulary of a CLIPTextEncoder with one or multiple new concepts, e.g. obtained via the Textual Inversion technique.

    Example:
        import torch
        from refiners.foundationals.clip.concepts import ConceptExtender
        from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
        from refiners.fluxion.utils import load_from_safetensors

        encoder = CLIPTextEncoderL(device="cuda")
        tensors = load_from_safetensors("CLIPTextEncoderL.safetensors")
        encoder.load_state_dict(tensors)

        cat_embedding = torch.load("cat_embedding.bin")["<this-cat>"]
        dog_embedding = torch.load("dog_embedding.bin")["<that-dog>"]

        extender = ConceptExtender(encoder)
        extender.add_concept(token="<this-cat>", embedding=cat_embedding)
        extender.inject()
        # New concepts can be added at any time
        extender.add_concept(token="<that-dog>", embedding=dog_embedding)

        # Now the encoder can be used with the new concepts
    """

    def __init__(self, target: CLIPTextEncoder) -> None:
        with self.setup_adapter(target):
            super().__init__(target)

        try:
            token_encoder, self.token_encoder_parent = next(target.walk(TokenEncoder))
        except StopIteration:
            raise RuntimeError("TokenEncoder not found.")

        try:
            clip_tokenizer, self.clip_tokenizer_parent = next(target.walk(CLIPTokenizer))
        except StopIteration:
            raise RuntimeError("Tokenizer not found.")

        self.embedding_extender = EmbeddingExtender(token_encoder)
        self.token_extender = TokenExtender(clip_tokenizer)

    def add_concept(self, token: str, embedding: Tensor) -> None:
        self.embedding_extender.add_embedding(embedding)
        self.token_extender.add_token(token, self.embedding_extender.num_embeddings - 1)

    def inject(self: "ConceptExtender", parent: fl.Chain | None = None) -> "ConceptExtender":
        self.embedding_extender.inject(self.token_encoder_parent)
        self.token_extender.inject(self.clip_tokenizer_parent)
        return super().inject(parent)

    def eject(self) -> None:
        self.embedding_extender.eject()
        self.token_extender.eject()
        super().eject()


class EmbeddingExtender(fl.Chain, Adapter[TokenEncoder]):
    old_weight: Parameter
    new_weight: Parameter
    weight: Tensor

    def __init__(
        self,
        target: TokenEncoder,
    ) -> None:
        with self.setup_adapter(target):
            super().__init__(fl.Lambda(func=self.lookup))
        self.old_weight = cast(Parameter, target.weight)
        self.new_weight = Parameter(
            zeros([0, target.embedding_dim], device=target.device, dtype=target.dtype)
        )  # requires_grad=True by default
        self.weight = cat([self.old_weight, self.new_weight])

    # Use F.embedding instead of nn.Embedding to make sure that gradients can only be computed for the new embeddings
    def lookup(self, x: Tensor) -> Tensor:
        return F.embedding(x, self.weight)

    def add_embedding(self, embedding: Tensor) -> None:
        assert embedding.shape == (self.old_weight.shape[1],)
        self.new_weight = Parameter(
            cat([self.new_weight, embedding.unsqueeze(0).to(self.new_weight.device, self.new_weight.dtype)])
        )
        self.weight = cat([self.old_weight, self.new_weight])

    @property
    def num_embeddings(self) -> int:
        return self.weight.shape[0]


class TokenExtender(fl.Chain, Adapter[CLIPTokenizer]):
    def __init__(self, target: CLIPTokenizer) -> None:
        with self.setup_adapter(target):
            super().__init__(
                CLIPTokenizer(
                    vocabulary_path=target.vocabulary_path,
                    sequence_length=target.sequence_length,
                    start_of_text_token_id=target.start_of_text_token_id,
                    end_of_text_token_id=target.end_of_text_token_id,
                    pad_token_id=target.pad_token_id,
                )
            )

    def add_token(self, token: str, token_id: int) -> None:
        tokenizer = self.find(layer_type=CLIPTokenizer)
        assert tokenizer is not None, "Tokenizer not found."
        assert token_id not in tokenizer.token_to_id_mapping.values()
        tokenizer.token_to_id_mapping[token] = token_id
        current_pattern = tokenizer.token_pattern.pattern
        new_pattern = token + "|" + current_pattern
        tokenizer.token_pattern = re.compile(new_pattern, re.IGNORECASE)
        # Define the keyword as its own smallest subtoken
        tokenizer.byte_pair_encoding_cache[token] = token
