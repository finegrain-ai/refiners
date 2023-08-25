import torch
import pytest

from warnings import warn
from pathlib import Path

from refiners.foundationals.clip.concepts import ConceptExtender
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
from refiners.foundationals.clip.tokenizer import CLIPTokenizer
from refiners.fluxion.utils import load_from_safetensors

from diffusers import StableDiffusionPipeline  # type: ignore
import transformers  # type: ignore


PROMPTS = [
    "a cute cat",  # a simple prompt
    "This artwork is inspired by <gta5-artwork> and uses a <cat-toy> as a prop",  # prompt with two added concepts
]


@pytest.fixture(scope="module")
def our_encoder_with_new_concepts(
    test_weights_path: Path,
    test_device: torch.device,
    cat_embedding_textual_inversion: torch.Tensor,
    gta5_artwork_embedding_textual_inversion: torch.Tensor,
) -> CLIPTextEncoderL:
    weights = test_weights_path / "CLIPTextEncoderL.safetensors"
    if not weights.is_file():
        warn(f"could not find weights at {weights}, skipping")
        pytest.skip(allow_module_level=True)
    encoder = CLIPTextEncoderL(device=test_device)
    tensors = load_from_safetensors(weights)
    encoder.load_state_dict(tensors)
    concept_extender = ConceptExtender(encoder)
    concept_extender.add_concept("<cat-toy>", cat_embedding_textual_inversion)
    concept_extender.add_concept("<gta5-artwork>", gta5_artwork_embedding_textual_inversion)
    concept_extender.inject()
    return encoder


@pytest.fixture(scope="module")
def ref_sd15_with_new_concepts(runwayml_weights_path: Path, test_textual_inversion_path: Path, test_device: torch.device):
    pipe = StableDiffusionPipeline.from_pretrained(runwayml_weights_path, torch_dtype=torch.float16).to(test_device) # type: ignore
    pipe.load_textual_inversion(test_textual_inversion_path / "cat-toy")  # type: ignore
    pipe.load_textual_inversion(test_textual_inversion_path / "gta5-artwork")  # type: ignore
    return pipe


@pytest.fixture(scope="module")
def runwayml_weights_path(test_weights_path: Path):
    r = test_weights_path / "runwayml" / "stable-diffusion-v1-5"
    if not r.is_dir():
        warn(f"could not find RunwayML weights at {r}, skipping")
        pytest.skip(allow_module_level=True)
    return r


@pytest.fixture(scope="module")
def ref_tokenizer_with_new_concepts(ref_sd15_with_new_concepts: StableDiffusionPipeline) -> transformers.CLIPTokenizer:
    return ref_sd15_with_new_concepts.tokenizer  # type: ignore


@pytest.fixture(scope="module")
def ref_encoder_with_new_concepts(ref_sd15_with_new_concepts: StableDiffusionPipeline) -> transformers.CLIPTextModel:
    return ref_sd15_with_new_concepts.text_encoder


@pytest.fixture(params=PROMPTS)
def prompt(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(scope="module")
def gta5_artwork_embedding_textual_inversion(test_textual_inversion_path: Path) -> torch.Tensor:
    return torch.load(test_textual_inversion_path / "gta5-artwork" / "learned_embeds.bin")["<gta5-artwork>"]  # type: ignore


@pytest.fixture(scope="module")
def cat_embedding_textual_inversion(test_textual_inversion_path: Path) -> torch.Tensor:
    return torch.load(test_textual_inversion_path / "cat-toy" / "learned_embeds.bin")["<cat-toy>"]  # type: ignore


def test_encoder(
    prompt: str,
    ref_tokenizer_with_new_concepts: transformers.CLIPTokenizer,
    ref_encoder_with_new_concepts: transformers.CLIPTextModel,
    our_encoder_with_new_concepts: CLIPTextEncoderL,
    test_device: torch.device,
):
    ref_tokens = ref_tokenizer_with_new_concepts(  # type: ignore
        prompt,
        padding="max_length",
        max_length=ref_tokenizer_with_new_concepts.model_max_length,  # type: ignore
        truncation=True,
        return_tensors="pt",
    ).input_ids
    assert isinstance(ref_tokens, torch.Tensor)
    tokenizer = our_encoder_with_new_concepts.find(layer_type=CLIPTokenizer)
    assert tokenizer is not None
    our_tokens = tokenizer(prompt)
    assert torch.equal(our_tokens, ref_tokens)

    with torch.no_grad():
        ref_embeddings = ref_encoder_with_new_concepts(ref_tokens.to(test_device))[0]
        our_embeddings = our_encoder_with_new_concepts(prompt)

    assert ref_embeddings.shape == (1, 77, 768)
    assert our_embeddings.shape == (1, 77, 768)

    # See `test_encoder` in test_text_encoder.py for details about the tolerance (0.04)
    assert (our_embeddings - ref_embeddings).abs().max() < 0.04
