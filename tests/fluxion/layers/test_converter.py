from typing import Any, Callable
from warnings import warn

import pytest
import torch

import refiners.fluxion.layers as fl
from refiners.fluxion.layers.chain import ChainError, Distribute


def test_converter_device_single_tensor(test_device: torch.device) -> None:
    if test_device.type != "cuda":
        warn("only running on CUDA, skipping")
        pytest.skip()

    chain = fl.Chain(
        fl.Converter(set_device=True, set_dtype=False),
        fl.Linear(in_features=1, out_features=1, device=test_device),
    )

    tensor = torch.randn(1, 1)
    converted_tensor = chain(tensor)

    assert converted_tensor.device == torch.device(device=test_device)


def test_converter_dtype_single_tensor() -> None:
    chain = fl.Chain(
        fl.Converter(set_device=False, set_dtype=True),
        fl.Linear(in_features=1, out_features=1, dtype=torch.float64),
    )

    tensor = torch.randn(1, 1).to(dtype=torch.float32)
    converted_tensor = chain(tensor)

    assert converted_tensor.dtype == torch.float64


def test_converter_multiple_tensors(test_device: torch.device) -> None:
    if test_device.type != "cuda":
        warn("only running on CUDA, skipping")
        pytest.skip()

    chain = fl.Chain(
        fl.Converter(set_device=True, set_dtype=True),
        Distribute(
            fl.Linear(in_features=1, out_features=1, device=test_device, dtype=torch.float64),
            fl.Linear(in_features=1, out_features=1, device=test_device, dtype=torch.float64),
        ),
    )

    tensor1 = torch.randn(1, 1)
    tensor2 = torch.randn(1, 1)

    converted_tensor1, converted_tensor2 = chain(tensor1, tensor2)

    assert converted_tensor1.device == torch.device(device=test_device)
    assert converted_tensor1.dtype == torch.float64
    assert converted_tensor2.device == torch.device(device=test_device)
    assert converted_tensor2.dtype == torch.float64


def test_converter_no_parent_device_or_dtype() -> None:
    identity: Callable[[Any], Any] = lambda x: x
    chain = fl.Chain(
        fl.Lambda(func=identity),
        fl.Converter(set_device=True, set_dtype=False),
    )

    tensor = torch.randn(1, 1)

    with pytest.raises(expected_exception=ChainError):
        chain(tensor)
