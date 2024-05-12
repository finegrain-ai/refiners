import pickle
from pathlib import Path
from warnings import warn

import pytest
from PIL import Image
from torch import Tensor, device as Device

from refiners.fluxion.utils import image_to_tensor, load_from_safetensors, no_grad
from refiners.foundationals.fuyu.fuyu import Fuyu, Fuyu8b


@pytest.fixture(scope="module")
def ref_logits(test_ref_path: Path) -> Tensor:
    logits = test_ref_path / "logits.safetensors"
    if not logits.is_file():
        warn(f"could not find reference logits at {logits}, skipping")
        pytest.skip(allow_module_level=True)
    return load_from_safetensors(logits)["logits"]


@pytest.fixture(scope="module")
def ref_generation(test_ref_path: Path) -> list[str]:
    generation = test_ref_path / "generation.pkl"
    if not generation.is_file():
        warn(f"could not find reference generation at {generation}, skipping")
        pytest.skip(allow_module_level=True)

    with generation.open("rb") as fp:
        ref_generation = pickle.load(fp)
    return ref_generation


@pytest.fixture(scope="module")
def our_model(test_weights_path: Path, test_vocab_path: Path, test_device: Device) -> Fuyu:
    weights = test_weights_path / f"fuyu.safetensors"
    vocab = test_vocab_path / f"tokenizer.json.gz"

    if not weights.is_file():
        warn(f"could not find weights at {weights}, skipping")
        pytest.skip(allow_module_level=True)

    if not vocab.is_file():
        warn(f"could not find weights at {vocab}, skipping")
        pytest.skip(allow_module_level=True)

    model = Fuyu8b(tokenizer_path=vocab, device=test_device)
    tensors = load_from_safetensors(weights)
    model.load_state_dict(tensors)
    return model


@no_grad()
def test_forward(our_model: Fuyu, ref_logits: Tensor, images: list[Image.Image], prompts: list[str]) -> None:
    """
    Tests the consistency of output features between the reference model and our model under random prompts.

    Args:
        our_model (Fuyu): Our model to be tested against ref_model.
        ref_logits (Tensor): the reference logits
        images (list[Image.Image]): test images to be used
        prompts (list[str]): test prompts to be used
    Raises:
        AssertionError: If the outputs of our model differ from the reference by a margin greater than 1e-3.
    """

    x = [image_to_tensor(image) for image in images]
    our_output = our_model(x, prompts).cpu()

    assert (our_output - ref_logits).abs().max() < 1e-3


@no_grad()
def test_generation(our_model: Fuyu, ref_generation: list[str], images: list[Image.Image], prompts: list[str]) -> None:
    """
    Tests the consistency of generation between the reference model and our model for a given pair (prompt, image).

    Args:
        our_model (Fuyu): Our model to be tested against ref_model.
        ref_generation (list[str]): reference answers
        images (list[Image.Image]): test images to be used
        prompts (list[str]): test prompts to be used
    Raises:
        AssertionError: If the generated answers of our model differ from the reference.
    """

    our_generation = our_model.generate(images=images, prompts=prompts, max_len_generation=100)
    our_generation = [answer.strip() for answer in our_generation]

    assert our_generation == ref_generation
