import torch
import pytest
import refiners.fluxion.layers as fl
from refiners.fluxion.layers.chain import Distribute


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_converter_device_single_tensor() -> None:
    chain = fl.Chain(
        fl.Converter(set_device=True, set_dtype=False),
        fl.Linear(in_features=1, out_features=1, device="cuda:0"),
    )

    tensor = torch.randn(1, 1)
    converted_tensor = chain(tensor)

    assert converted_tensor.device == torch.device(device="cuda:0")


def test_converter_dtype_single_tensor() -> None:
    chain = fl.Chain(
        fl.Converter(set_device=False, set_dtype=True),
        fl.Linear(in_features=1, out_features=1, dtype=torch.float64),
    )

    tensor = torch.randn(1, 1).to(dtype=torch.float32)
    converted_tensor = chain(tensor)

    assert converted_tensor.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_converter_multiple_tensors() -> None:
    chain = fl.Chain(
        fl.Converter(set_device=True, set_dtype=True),
        Distribute(
            fl.Linear(in_features=1, out_features=1, device="cuda:0", dtype=torch.float64),
            fl.Linear(in_features=1, out_features=1, device="cuda:0", dtype=torch.float64),
        ),
    )

    tensor1 = torch.randn(1, 1)
    tensor2 = torch.randn(1, 1)

    converted_tensor1, converted_tensor2 = chain(tensor1, tensor2)

    assert converted_tensor1.device == torch.device(device="cuda:0")
    assert converted_tensor1.dtype == torch.float64
    assert converted_tensor2.device == torch.device(device="cuda:0")
    assert converted_tensor2.dtype == torch.float64


def test_converter_no_parent_device_or_dtype() -> None:
    chain = fl.Chain(
        fl.Lambda(func=(lambda x: x)),
        fl.Converter(set_device=True, set_dtype=False),
    )

    tensor = torch.randn(1, 1)

    with pytest.raises(expected_exception=ValueError):
        chain(tensor)
