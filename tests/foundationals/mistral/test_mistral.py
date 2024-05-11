from pathlib import Path
from warnings import warn

import pytest
import torch
from torch import device as Device
from transformers import MistralForCausalLM  # type: ignore

from refiners.fluxion.utils import load_from_safetensors, manual_seed
from refiners.foundationals.mistral import Mistral7b


@pytest.fixture(scope="module")
def our_model(test_weights_path: Path, test_device: Device) -> Mistral7b:
    weights = test_weights_path / "mistral_7b_v0.1.safetensors"

    if not weights.is_file():
        warn(f"could not find weights at {weights}, skipping")
        pytest.skip(allow_module_level=True)

    model = Mistral7b(device=test_device)
    tensors = load_from_safetensors(weights)
    model.load_state_dict(tensors)
    return model


@pytest.fixture(scope="module")
def ref_model(test_device: Device, token: str) -> MistralForCausalLM:
    model: MistralForCausalLM = MistralForCausalLM.from_pretrained(  # type: ignore
        pretrained_model_name_or_path="mistralai/Mistral-7B-v0.1",
        token=token,
        device=test_device
    )
    return model


def test_model(ref_model: MistralForCausalLM, our_model: Mistral7b):
    manual_seed(2)
    input_ids = torch.randint(0, 10000, size=(1, 100))

    ref_out = ref_model(input_ids)["logits"]
    our_out = our_model(input_ids)

    assert torch.allclose(ref_out, our_out, atol=1e-4)
