import os
from pathlib import Path

import torch
from pytest import fixture

PARENT_PATH = Path(__file__).parent


@fixture(scope="session")
def test_device() -> torch.device:
    test_device = os.getenv("REFINERS_TEST_DEVICE")
    if not test_device:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(test_device)


@fixture(scope="session")
def test_weights_path() -> Path:
    from_env = os.getenv("REFINERS_TEST_WEIGHTS_DIR")
    return Path(from_env) if from_env else PARENT_PATH / "weights"


@fixture(scope="session")
def test_e2e_path() -> Path:
    return PARENT_PATH / "e2e"


@fixture(scope="session")
def test_textual_inversion_path() -> Path:
    return PARENT_PATH / "foundationals" / "clip" / "test_concepts_ref"


@fixture(scope="session")
def test_sam_path() -> Path:
    return PARENT_PATH / "foundationals" / "segment_anything"
