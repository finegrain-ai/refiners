from dataclasses import dataclass

from torchvision.transforms.functional import gaussian_blur as torch_gaussian_blur  # type: ignore
import pytest
import torch

from refiners.fluxion.utils import gaussian_blur, manual_seed


@dataclass
class BlurInput:
    kernel_size: int | tuple[int, int]
    sigma: float | tuple[float, float] | None = None
    image_height: int = 512
    image_width: int = 512
    batch_size: int | None = 1


BLUR_INPUTS: list[BlurInput] = [
    BlurInput(kernel_size=9),
    BlurInput(kernel_size=9, batch_size=None),
    BlurInput(kernel_size=9, sigma=1.0),
    BlurInput(kernel_size=9, sigma=1.0, image_height=768),
    BlurInput(kernel_size=(9, 5), sigma=(1.0, 0.8)),
]


@pytest.fixture(params=BLUR_INPUTS)
def blur_input(request: pytest.FixtureRequest) -> BlurInput:
    return request.param


def test_gaussian_blur(blur_input: BlurInput) -> None:
    manual_seed(2)
    tensor = torch.randn(3, blur_input.image_height, blur_input.image_width)
    if blur_input.batch_size is not None:
        tensor = tensor.expand(blur_input.batch_size, -1, -1, -1)

    ref_blur = torch_gaussian_blur(tensor, blur_input.kernel_size, blur_input.sigma)  # type: ignore
    our_blur = gaussian_blur(tensor, blur_input.kernel_size, blur_input.sigma)

    assert torch.equal(our_blur, ref_blur)
