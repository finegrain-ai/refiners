from pathlib import Path

import pytest
import torch
from PIL import Image
from tests.utils import ensure_similar_images

from refiners.solutions import BoxSegmenter


@pytest.fixture(scope="module")
def ref_path(test_e2e_path: Path) -> Path:
    return test_e2e_path / "test_solutions_ref"


@pytest.fixture(scope="module")
def ref_shelves(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "shelves.jpg").convert("RGB")


@pytest.fixture
def expected_box_segmenter_plant_mask(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_box_segmenter_plant_mask.png")


@pytest.fixture
def expected_box_segmenter_spray_mask(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_box_segmenter_spray_mask.png")


@pytest.fixture
def expected_box_segmenter_spray_cropped_mask(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "expected_box_segmenter_spray_cropped_mask.png")


def test_box_segmenter(
    box_segmenter_weights_path: Path,
    ref_shelves: Image.Image,
    expected_box_segmenter_plant_mask: Image.Image,
    expected_box_segmenter_spray_mask: Image.Image,
    expected_box_segmenter_spray_cropped_mask: Image.Image,
    test_device: torch.device,
):
    segmenter = BoxSegmenter(weights=box_segmenter_weights_path, device=test_device)

    plant_mask = segmenter(ref_shelves, box_prompt=(504, 82, 754, 368))
    ensure_similar_images(plant_mask.convert("RGB"), expected_box_segmenter_plant_mask.convert("RGB"))

    spray_box = (461, 542, 594, 823)
    spray_mask = segmenter(ref_shelves, box_prompt=spray_box)
    ensure_similar_images(spray_mask.convert("RGB"), expected_box_segmenter_spray_mask.convert("RGB"))

    # Test left and bottom padding.
    off_l, off_b = 11, 7
    shelves_cropped = ref_shelves.crop((spray_box[0] - off_l, 0, ref_shelves.width, spray_box[3] + off_b))
    spray_cropped_box = (off_l, spray_box[1], spray_box[2] - spray_box[0] + off_l, spray_box[3])
    spray_cropped_mask = segmenter(shelves_cropped, box_prompt=spray_cropped_box)
    ensure_similar_images(spray_cropped_mask.convert("RGB"), expected_box_segmenter_spray_cropped_mask.convert("RGB"))
