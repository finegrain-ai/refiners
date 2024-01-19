import pickle
from dataclasses import dataclass
from pathlib import Path
from warnings import warn

import pytest
import torch
from PIL import Image
from torch import device as Device, dtype as DType
from torchvision.transforms.functional import gaussian_blur as torch_gaussian_blur  # type: ignore

from refiners.fluxion import layers as fl
from refiners.fluxion.utils import (
    gaussian_blur,
    image_to_tensor,
    load_tensors,
    manual_seed,
    no_grad,
    summarize_tensor,
    tensor_to_image,
)


@dataclass
class BlurInput:
    kernel_size: int | tuple[int, int]
    sigma: float | tuple[float, float] | None = None
    image_height: int = 512
    image_width: int = 512
    batch_size: int | None = 1
    dtype: DType = torch.float32


BLUR_INPUTS: list[BlurInput] = [
    BlurInput(kernel_size=9),
    BlurInput(kernel_size=9, batch_size=None),
    BlurInput(kernel_size=9, sigma=1.0),
    BlurInput(kernel_size=9, sigma=1.0, image_height=768),
    BlurInput(kernel_size=(9, 5), sigma=(1.0, 0.8)),
    BlurInput(kernel_size=9, dtype=torch.float16),
]


@pytest.fixture(params=BLUR_INPUTS)
def blur_input(request: pytest.FixtureRequest) -> BlurInput:
    return request.param


def test_gaussian_blur(test_device: Device, blur_input: BlurInput) -> None:
    if test_device.type == "cpu" and blur_input.dtype == torch.float16:
        warn("half float is not supported on the CPU because of `torch.mm`, skipping")
        pytest.skip()
    manual_seed(2)
    tensor = torch.randn(3, blur_input.image_height, blur_input.image_width, device=test_device, dtype=blur_input.dtype)
    if blur_input.batch_size is not None:
        tensor = tensor.expand(blur_input.batch_size, -1, -1, -1)

    ref_blur = torch_gaussian_blur(tensor, blur_input.kernel_size, blur_input.sigma)  # type: ignore
    our_blur = gaussian_blur(tensor, blur_input.kernel_size, blur_input.sigma)

    assert torch.equal(our_blur, ref_blur)


def test_image_to_tensor() -> None:
    image = Image.new("RGB", (512, 512))

    assert image_to_tensor(image).shape == (1, 3, 512, 512)
    assert image_to_tensor(image.convert("L")).shape == (1, 1, 512, 512)
    assert image_to_tensor(image.convert("RGBA")).shape == (1, 4, 512, 512)


def test_tensor_to_image() -> None:
    assert tensor_to_image(torch.zeros(1, 3, 512, 512)).mode == "RGB"
    assert tensor_to_image(torch.zeros(1, 1, 512, 512)).mode == "L"
    assert tensor_to_image(torch.zeros(1, 4, 512, 512)).mode == "RGBA"
    assert tensor_to_image(torch.zeros(1, 3, 512, 512, dtype=torch.bfloat16)).mode == "RGB"


def test_summarize_tensor() -> None:
    assert summarize_tensor(torch.zeros(1, 3, 512, 512).int())
    assert summarize_tensor(torch.zeros(1, 3, 512, 512).float())
    assert summarize_tensor(torch.zeros(1, 3, 512, 512).double())
    assert summarize_tensor(torch.complex(torch.zeros(1, 3, 512, 512), torch.zeros(1, 3, 512, 512)))
    assert summarize_tensor(torch.zeros(1, 3, 512, 512).bfloat16())
    assert summarize_tensor(torch.zeros(1, 3, 512, 512).bool())
    assert summarize_tensor(torch.zeros(1, 0, 512, 512).int())


def test_no_grad() -> None:
    x = torch.randn(1, 1, requires_grad=True)

    with torch.no_grad():
        y = x + 1
        assert not y.requires_grad

    with no_grad():
        z = x + 1
        assert not z.requires_grad

    w = x + 1
    assert w.requires_grad


def test_load_tensors_valid_pickle(tmp_path: Path) -> None:
    pickle_path = tmp_path / "valid.pickle"

    tensors = {"easy-as.weight": torch.tensor([1.0, 2.0, 3.0])}
    torch.save(tensors, pickle_path)  # type: ignore
    loaded_tensor = load_tensors(pickle_path)
    assert torch.equal(loaded_tensor["easy-as.weight"], tensors["easy-as.weight"])

    tensors = {"easy-as.weight": torch.tensor([1, 2, 3]), "hello": "world"}
    torch.save(tensors, pickle_path)  # type: ignore

    with pytest.raises(AssertionError):
        loaded_tensor = load_tensors(pickle_path)


def test_load_tensors_invalid_pickle(tmp_path: Path) -> None:
    invalid_pickle_path = tmp_path / "invalid.pickle"
    model = fl.Chain(fl.Linear(1, 1))
    torch.save(model, invalid_pickle_path)  # type: ignore
    with pytest.raises(
        pickle.UnpicklingError,
    ):
        load_tensors(invalid_pickle_path)
