from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10  # type: ignore

from refiners.foundationals import dinov2
from refiners.training_utils.metrics import dinov2_frechet_distance


class CifarDataset(Dataset[torch.Tensor]):
    def __init__(self, ds: Dataset[list[torch.Tensor]], max_len: int = 512) -> None:
        self.ds = ds
        ds_length = len(self.ds)  # type: ignore
        self.length = min(ds_length, max_len)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.ds[i][0]


@pytest.fixture(scope="module")
def dinov2_l(
    dinov2_large_weights_path: Path,
    test_device: torch.device,
) -> dinov2.DINOv2_large:
    model = dinov2.DINOv2_large(device=test_device)
    model.load_from_safetensors(dinov2_large_weights_path)
    return model


def test_dinov2_frechet_distance(test_datasets_path: Path, dinov2_l: dinov2.DINOv2_large) -> None:
    path = str(test_datasets_path / "CIFAR10")

    ds_train = CifarDataset(
        CIFAR10(
            root=path,
            train=True,
            download=True,
            transform=dinov2.preprocess,
        )
    )

    ds_test = CifarDataset(
        CIFAR10(
            root=path,
            train=False,
            download=True,
            transform=dinov2.preprocess,
        )
    )

    # Computed using dgm-eval (https://github.com/layer6ai-labs/dgm-eval)
    # with interpolate_offset=0 and random_sample=False.
    expected_d = 837.978

    d = dinov2_frechet_distance(ds_train, ds_test, dinov2_l)
    assert expected_d - 1e-2 < d < expected_d + 1e-2
