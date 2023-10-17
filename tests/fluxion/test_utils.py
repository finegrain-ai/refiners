from dataclasses import dataclass
from warnings import warn

from torchvision.transforms.functional import gaussian_blur as torch_gaussian_blur  # type: ignore
from torch import device as Device, dtype as DType
from PIL import Image
import pytest
import torch

from refiners.fluxion.utils import gaussian_blur, image_to_tensor, manual_seed, tensor_to_image


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
