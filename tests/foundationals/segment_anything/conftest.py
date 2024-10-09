import gc
from pathlib import Path

from pytest import fixture


@fixture(autouse=True)
def ensure_gc():
    # Avoid GPU OOMs
    # See https://github.com/pytest-dev/pytest/discussions/8153#discussioncomment-214812
    gc.collect()


@fixture(scope="package")
def ref_path(test_sam_path: Path) -> Path:
    return test_sam_path / "test_sam_ref"
