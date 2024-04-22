import random
from pathlib import Path
from warnings import warn

import pytest
import torch
from PIL import Image
from torch import device as Device
from torchvision.transforms.functional import (  # type: ignore[reportMissingTypeStubs]
    to_pil_image,  # type: ignore[reportUnknownVariableType]
)
from transformers import FuyuForCausalLM, FuyuProcessor  # type: ignore[reportMissingTypeStubs]

from refiners.fluxion.utils import load_from_safetensors, manual_seed, no_grad
from refiners.foundationals.fuyu.fuyu import Fuyu, Fuyu8b, create_fuyu


@pytest.fixture(scope="module")
def our_model(test_weights_path: Path, test_device: Device) -> Fuyu:
    weights = test_weights_path / f"fuyu8b.safetensors"

    if not weights.is_file():
        warn(f"could not find weights at {weights}, skipping")
        pytest.skip(allow_module_level=True)

    config = Fuyu8b().with_device(test_device)
    model = create_fuyu(config)
    tensors = load_from_safetensors(weights)
    model.load_state_dict(tensors)
    return model


@pytest.fixture(scope="module")
def ref_processor() -> FuyuProcessor:
    processor: FuyuProcessor = FuyuProcessor.from_pretrained(pretrained_model_name_or_path="adept/fuyu-8b")  # type: ignore[reportUnknownMemberType, reportAssignmentType]
    return processor


@pytest.fixture(scope="module")
def ref_model(test_device: Device) -> FuyuForCausalLM:
    model: FuyuForCausalLM = FuyuForCausalLM.from_pretrained(pretrained_model_name_or_path="adept/fuyu-8b").to(  # type: ignore
        device=test_device
    )
    return model  # type: ignore[reportUnknownVariableType]


def test_model(ref_model: FuyuForCausalLM, ref_processor: FuyuProcessor, our_model: Fuyu, test_device: Device):
    """
    Tests the consistency of output features between the reference model and our model under random prompts.

    Args:
        ref_model (FuyuForCausalLM): The reference model.
        ref_processor (FuyuProcessor): The processor used for preparing input data for ref_model.
        our_model (Fuyu): Our model to be tested against ref_model.
        test_device (torch.device): The device (e.g., CPU or GPU) to perform the test on.

    Warning:
        The ref model from transformers can't be put on the device without the installation of accelerate

    Raises:
        AssertionError: If the outputs of the models differ by a margin greater than 1e-3.
    """

    manual_seed(42)
    x = torch.rand(3, 512, 512)
    x_pil: Image.Image = to_pil_image(x)  # type: ignore[reportUnknownVariableType]

    prompts = [
        "Describe this image. \n",
        "Is there a cat in the image? \n",
        "What is the emotion of the person? \n",
        "What is the main object in this image? \n",
    ]
    p = random.choice(prompts)

    with no_grad():
        ref_input = ref_processor(text=p, images=x_pil, return_tensors="pt").to(device=test_device)  # type: ignore[reportUnknownMemberType]
        ref_output = ref_model(**ref_input)["logits"]
        our_output = our_model([x.unsqueeze(0)], [p])

    assert (our_output - ref_output).abs().max() < 1e-3
