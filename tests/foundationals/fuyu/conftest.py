import pickle
from pathlib import Path
from typing import Dict

import pytest
from PIL import Image
from torch import Tensor, device as Device
from transformers import FuyuForCausalLM, FuyuProcessor  # type: ignore[reportMissingTypeStubs]

from refiners.fluxion.utils import no_grad, save_to_safetensors


def _img_open(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")  # type: ignore


@pytest.fixture(scope="package")
def test_vocab_path() -> Path:
    return Path(__file__).parent / "test_vocab"


@pytest.fixture(scope="package")
def test_ref_path() -> Path:
    return Path(__file__).parent / "test_ref"


@pytest.fixture(scope="package")
def images() -> list[Image.Image]:
    assets = Path(__file__).parent.parent.parent.parent / "assets"
    return [
        _img_open(assets / "dragon_quest_slime.jpg"),
        _img_open(assets / "dropy_logo.png"),
        _img_open(assets / "pokemon_cat.png"),
    ]


@pytest.fixture(scope="package")
def prompts() -> list[str]:
    return ["Describe this image.\n", "What is the emotion of the person?\n", "Is there a cat in the image?\n"]


@no_grad()
@pytest.fixture(scope="module", autouse=True)
def setup_reference(test_ref_path: Path, test_device: Device, images: list[Image.Image], prompts: list[str]) -> None:
    """
    Generate the generated answer from the reference model and store them in the test_ref_path folder.

    Args:
        test_ref_path (Path): Path to the folder containing the references
        test_device (torch.device): The device (e.g., CPU or GPU) to perform the test on.
        images (list[Image.Image]): test images to be used
        prompts (list[str]): test prompts to be used
    """

    if not test_ref_path.exists():
        test_ref_path.mkdir(parents=True, exist_ok=True)

    ref_processor: FuyuProcessor = FuyuProcessor.from_pretrained(pretrained_model_name_or_path="adept/fuyu-8b")  # type: ignore
    ref_model: FuyuForCausalLM = FuyuForCausalLM.from_pretrained(pretrained_model_name_or_path="adept/fuyu-8b").to(  # type: ignore
        device=test_device
    )
    ref_input: Dict[str, Tensor] = ref_processor(images=images, text=prompts, return_tensors="pt").to(  # type: ignore[reportUnknownVariableType]
        device=test_device
    )

    # Generate logits of a single forward pass
    logits_path = test_ref_path / "logits.safetensors"
    ref_output: Tensor = ref_model(**ref_input)["logits"]  # type: ignore[reportUnknownVariableType]
    save_to_safetensors(logits_path, {"logits": ref_output})

    # Generate answers to the given prompts
    answers_path = test_ref_path / "generation.pkl"
    ref_output: Tensor = ref_model.generate(**ref_input, max_new_tokens=100, use_cache=False)  # type: ignore[reportUnknownMemberType]
    ref_generation: list[str] = ref_processor.batch_decode(ref_output, skip_special_tokens=True)  # type: ignore[reportUnknownMemberType]
    # get elements after beginning of answer token and strip
    ref_generation = [answer.split("\x04")[1].strip() for answer in ref_generation]  # type: ignore[reportUnknownMemberType]
    with answers_path.open("wb") as fp:
        pickle.dump(ref_generation, fp)
