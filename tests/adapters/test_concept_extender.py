import pytest
import torch

from refiners.fluxion.utils import no_grad
from refiners.foundationals.clip.concepts import ConceptExtender
from refiners.foundationals.clip.text_encoder import CLIPTextEncoder, CLIPTextEncoderL
from refiners.foundationals.clip.tokenizer import CLIPTokenizer


@no_grad()
@pytest.mark.parametrize("k_encoder", [CLIPTextEncoderL])
def test_inject_eject(k_encoder: type[CLIPTextEncoder], test_device: torch.device):
    encoder = k_encoder(device=test_device)
    initial_repr = repr(encoder)

    extender = ConceptExtender(encoder)

    cat_embedding = torch.randn((encoder.embedding_dim,), device=test_device)
    extender.add_concept(token="<token1>", embedding=cat_embedding)

    extender_2 = ConceptExtender(encoder)

    assert repr(encoder) == initial_repr
    extender.inject()
    assert repr(encoder) != initial_repr

    with pytest.raises(AssertionError) as no_nesting:
        extender_2.inject()
    assert str(no_nesting.value) == "ConceptExtender cannot be nested, add concepts to the injected instance instead."

    with pytest.raises(AssertionError) as no_nesting:
        ConceptExtender(encoder)
    assert str(no_nesting.value) == "ConceptExtender cannot be nested, add concepts to the injected instance instead."

    dog_embedding = torch.randn((encoder.embedding_dim,), device=test_device)
    extender.add_concept(token="<token2>", embedding=dog_embedding)
    extender.eject()

    extender_2.inject().eject()
    ConceptExtender(encoder)  # no exception
    assert repr(encoder) == initial_repr

    tokenizer = encoder.ensure_find(CLIPTokenizer)
    assert len(tokenizer.encode("<token1>")) > 3
    assert len(tokenizer.encode("<token2>")) > 3

    extender.inject()

    tokenizer = encoder.ensure_find(CLIPTokenizer)
    assert tokenizer.encode("<token1>").equal(
        torch.tensor(
            [
                tokenizer.start_of_text_token_id,
                tokenizer.end_of_text_token_id + 1,
                tokenizer.end_of_text_token_id,
            ]
        )
    )
    assert tokenizer.encode("<token2>").equal(
        torch.tensor(
            [
                tokenizer.start_of_text_token_id,
                tokenizer.end_of_text_token_id + 2,
                tokenizer.end_of_text_token_id,
            ]
        )
    )
    assert len(tokenizer.encode("<token3>")) > 3
