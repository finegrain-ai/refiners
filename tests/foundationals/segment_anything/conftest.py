from pathlib import Path
from warnings import warn

from pytest import fixture, skip


@fixture(scope="package")
def ref_path(test_sam_path: Path) -> Path:
    return test_sam_path / "test_sam_ref"


@fixture(scope="package")
def sam_h_weights(test_weights_path: Path) -> Path:
    sam_h_weights = test_weights_path / "segment-anything-h.safetensors"
    if not sam_h_weights.is_file():
        warn(f"could not find weights at {sam_h_weights}, skipping")
        skip(allow_module_level=True)
    return sam_h_weights
