from pathlib import Path
from warnings import warn

import pytest
import torch
import transformers  # type: ignore

from refiners.fluxion.utils import load_from_safetensors, no_grad
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
from refiners.foundationals.clip.tokenizer import CLIPTokenizer

long_prompt = """
Above these apparent hieroglyphics was a figure of evidently pictorial intent,
though its impressionistic execution forbade a very clear idea of its nature.
It seemed to be a sort of monster, or symbol representing a monster, of a form
which only a diseased fancy could conceive. If I say that my somewhat extravagant
imagination yielded simultaneous pictures of an octopus, a dragon, and a human
caricature, I shall not be unfaithful to the spirit of the thing. A pulpy,
tentacled head surmounted a grotesque and scaly body with rudimentary wings;
but it was the general outline of the whole which made it most shockingly frightful.
Behind the figure was a vague suggestion of a Cyclopean architectural background.
"""

PROMPTS = [
    "",  # empty
    "a cute cat",  # padded
    long_prompt,  # truncated
    "64k",  # FG-362 - encoded as 3 tokens
]


@pytest.fixture(scope="module")
def our_encoder(test_weights_path: Path, test_device: torch.device) -> CLIPTextEncoderL:
    weights = test_weights_path / "CLIPTextEncoderL.safetensors"
    if not weights.is_file():
        warn(f"could not find weights at {weights}, skipping")
        pytest.skip(allow_module_level=True)
    encoder = CLIPTextEncoderL(device=test_device)
    tensors = load_from_safetensors(weights)
    encoder.load_state_dict(tensors)
    return encoder


@pytest.fixture(scope="module")
def runwayml_weights_path(test_weights_path: Path):
    r = test_weights_path / "runwayml" / "stable-diffusion-v1-5"
    if not r.is_dir():
        warn(f"could not find RunwayML weights at {r}, skipping")
        pytest.skip(allow_module_level=True)
    return r


@pytest.fixture(scope="module")
def ref_tokenizer(runwayml_weights_path: Path) -> transformers.CLIPTokenizer:
    return transformers.CLIPTokenizer.from_pretrained(runwayml_weights_path, subfolder="tokenizer")  # type: ignore


@pytest.fixture(scope="module")
def ref_encoder(runwayml_weights_path: Path, test_device: torch.device) -> transformers.CLIPTextModel:
    return transformers.CLIPTextModel.from_pretrained(runwayml_weights_path, subfolder="text_encoder").to(test_device)  # type: ignore


def test_basics(ref_tokenizer: transformers.CLIPTokenizer, our_encoder: CLIPTextEncoderL):
    assert ref_tokenizer.model_max_length == 77  # type: ignore
    assert our_encoder.max_sequence_length == 77


@pytest.fixture(params=PROMPTS)
def prompt(request: pytest.FixtureRequest):
    return request.param


def test_encoder(
    prompt: str,
    ref_tokenizer: transformers.CLIPTokenizer,
    ref_encoder: transformers.CLIPTextModel,
    our_encoder: CLIPTextEncoderL,
    test_device: torch.device,
):
    ref_tokens = ref_tokenizer(  # type: ignore
        prompt,
        padding="max_length",
        max_length=ref_tokenizer.model_max_length,  # type: ignore
        truncation=True,
        return_tensors="pt",
    ).input_ids
    assert isinstance(ref_tokens, torch.Tensor)
    tokenizer = our_encoder.ensure_find(CLIPTokenizer)
    our_tokens = tokenizer(prompt)
    assert torch.equal(our_tokens, ref_tokens)

    with no_grad():
        ref_embeddings = ref_encoder(ref_tokens.to(test_device))[0]
        our_embeddings = our_encoder(prompt)

    assert ref_embeddings.shape == (1, 77, 768)
    assert our_embeddings.shape == (1, 77, 768)

    # FG-336 - Not strictly equal because we do not use the same implementation
    # of self-attention. We use `scaled_dot_product_attention` which can have
    # numerical differences depending on the backend.
    # Also we use FP16 weights.
    assert (our_embeddings - ref_embeddings).abs().max() < 0.01


def test_list_string_tokenizer(
    prompt: str,
    our_encoder: CLIPTextEncoderL,
):
    tokenizer = our_encoder.ensure_find(CLIPTokenizer)

    # batched inputs
    double_tokens = tokenizer([prompt, prompt[0:3]])
    assert double_tokens.shape[0] == 2
