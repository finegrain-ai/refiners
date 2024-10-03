import os
from pathlib import Path
from typing import Callable

import torch
from pytest import FixtureRequest, fixture, skip

from refiners.fluxion.utils import str_to_dtype

PARENT_PATH = Path(__file__).parent

collect_ignore = ["weights", "repos", "datasets"]
collect_ignore_glob = ["*_ref"]


@fixture(scope="session")
def test_device() -> torch.device:
    test_device = os.getenv("REFINERS_TEST_DEVICE")
    if not test_device:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(test_device)


def dtype_fixture_factory(params: list[str]) -> Callable[[FixtureRequest], torch.dtype]:
    @fixture(scope="session", params=params)
    def dtype_fixture(request: FixtureRequest) -> torch.dtype:
        torch_dtype = str_to_dtype(request.param)
        if torch_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            skip("bfloat16 is not supported on this test device")
        return torch_dtype

    return dtype_fixture


test_dtype_fp32_bf16_fp16 = dtype_fixture_factory(["float32", "bfloat16", "float16"])
test_dtype_fp32_fp16 = dtype_fixture_factory(["float32", "float16"])
test_dtype_fp32_bf16 = dtype_fixture_factory(["float32", "bfloat16"])
test_dtype_fp16_bf16 = dtype_fixture_factory(["float16", "bfloat16"])


@fixture(scope="session")
def test_weights_path() -> Path:
    from_env = os.getenv("REFINERS_TEST_WEIGHTS_DIR")
    return Path(from_env) if from_env else PARENT_PATH / "weights"


@fixture(scope="session")
def test_datasets_path() -> Path:
    from_env = os.getenv("REFINERS_TEST_DATASETS_DIR")
    return Path(from_env) if from_env else PARENT_PATH / "datasets"


@fixture(scope="session")
def test_repos_path() -> Path:
    from_env = os.getenv("REFINERS_TEST_REPOS_DIR")
    return Path(from_env) if from_env else PARENT_PATH / "repos"


@fixture(scope="session")
def test_e2e_path() -> Path:
    return PARENT_PATH / "e2e"


@fixture(scope="session")
def test_textual_inversion_path() -> Path:
    return PARENT_PATH / "foundationals" / "clip" / "test_concepts_ref"


@fixture(scope="session")
def test_sam_path() -> Path:
    return PARENT_PATH / "foundationals" / "segment_anything"
