import random
from pathlib import Path
from typing import List
from warnings import warn

import pytest
import torch
from PIL import Image
from torch import Tensor, device as Device
from transformers import FuyuForCausalLM, FuyuProcessor  # type: ignore[reportMissingTypeStubs]

from refiners.fluxion.utils import load_from_safetensors, manual_seed, no_grad, tensor_to_image
from refiners.foundationals.fuyu.fuyu import Fuyu, Fuyu8b


def _img_open(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")  # type: ignore


@pytest.fixture(scope="module")
def our_model(test_weights_path: Path, test_vocab_path: Path, test_device: Device) -> Fuyu:
    weights = test_weights_path / f"fuyu8b.safetensors"
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


@pytest.fixture(scope="module")
def ref_processor() -> FuyuProcessor:
    return FuyuProcessor.from_pretrained(pretrained_model_name_or_path="adept/fuyu-8b")  # type: ignore


@pytest.fixture(scope="module")
def ref_model(test_device: Device) -> FuyuForCausalLM:
    return FuyuForCausalLM.from_pretrained(pretrained_model_name_or_path="adept/fuyu-8b").to(  # type: ignore
        device=test_device
    )


@no_grad()
def test_forward(
    ref_model: FuyuForCausalLM, ref_processor: FuyuProcessor, our_model: Fuyu, test_device: Device
) -> None:
    """
    Tests the consistency of output features between the reference model and our model under random prompts.

    Args:
        ref_model (FuyuForCausalLM): The reference model.
        ref_processor (FuyuProcessor): The processor used for preparing input data for ref_model.
        our_model (Fuyu): Our model to be tested against ref_model.
        test_device (torch.device): The device (e.g., CPU or GPU) to perform the test on.

    Raises:
        AssertionError: If the outputs of the models differ by a margin greater than 1e-3.
    """

    manual_seed(42)
    x = (torch.randint(0, 256, size=(1, 3, 512, 512), dtype=torch.uint8) / 255).float()
    x_pil = tensor_to_image(x)

    prompts = [
        "Describe this image.\n",
        "Is there a cat in the image?\n",
        "What is the emotion of the person?\n",
        "What is the main object in this image?\n",
    ]
    p = random.choice(prompts)

    ref_input = ref_processor(images=x_pil, text=p, return_tensors="pt").to(device=test_device)  # type: ignore[reportUnknownMemberType]
    ref_output = ref_model(**ref_input)["logits"]
    our_output = our_model([x], [p])

    assert (our_output - ref_output).abs().max() < 1e-3


@no_grad()
def test_generation(
    ref_model: FuyuForCausalLM, ref_processor: FuyuProcessor, our_model: Fuyu, test_device: Device
) -> None:
    """
    Tests the consistency of generation between the reference model and our model for a given pair (prompt, image).

    Args:
        ref_model (FuyuForCausalLM): The reference model.
        ref_processor (FuyuProcessor): The processor used for preparing input data for ref_model.
        our_model (Fuyu): Our model to be tested against ref_model.
        test_device (torch.device): The device (e.g., CPU or GPU) to perform the test on.

    Raises:
        AssertionError: If the generated answers of the models differ.
    """
    manual_seed(42)

    assets = Path(__file__).parent.parent.parent.parent / "assets"
    images = [
        _img_open(assets / "dragon_quest_slime.jpg"),
        _img_open(assets / "dropy.png"),
        _img_open(assets / "pokemon_cat.png"),
        _img_open(assets / "logo_dark.png"),
    ]
    prompts = [
        "Describe this image.\n",
        "What is the main object in this image?\n",
        "Is there a cat in the image?\n",
        "Generate a coco-style caption.\n",
    ]
    p = random.choices(prompts, k=len(images))

    # Get inputs processed for our reference model
    ref_input: Dict[str, Tensor] = ref_processor(images=images, text=p, return_tensors="pt").to(device=test_device)  # type: ignore[reportUnknownMemberType]
    # Get tokens of the answer generated
    ref_output: Tensor = ref_model.generate(**ref_input, max_new_tokens=100, use_cache=False)  # type: ignore[reportUnknownMemberType]
    # Decodes tokens and get the element after the beginning of answer token
    ref_generation: List[str] = ref_processor.batch_decode(ref_output, skip_special_tokens=True)  # type: ignore[reportUnknownMemberType]
    ref_generation = [answer.split("\x04")[1].strip() for answer in ref_generation]  # type: ignore[reportUnknownMemberType]

    our_generation = our_model.generate(images=images, prompts=p, max_len_generation=100)

    assert our_generation == ref_generation
