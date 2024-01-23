from pathlib import Path
from warnings import warn

import pytest
import torch

from refiners.fluxion.utils import load_tensors
from refiners.foundationals.latent_diffusion import StableDiffusion_1
from refiners.foundationals.latent_diffusion.lora import Lora, SDLoraManager


@pytest.fixture
def manager() -> SDLoraManager:
    target = StableDiffusion_1()
    return SDLoraManager(target)


@pytest.fixture
def weights(test_weights_path: Path) -> dict[str, torch.Tensor]:
    weights_path = test_weights_path / "loras" / "pokemon-lora" / "pytorch_lora_weights.bin"

    if not weights_path.is_file():
        warn(f"could not find weights at {weights_path}, skipping")
        pytest.skip(allow_module_level=True)

    return load_tensors(weights_path)


def test_add_loras(manager: SDLoraManager, weights: dict[str, torch.Tensor]) -> None:
    manager.add_loras("pokemon-lora", tensors=weights)
    assert "pokemon-lora" in manager.names

    with pytest.raises(AssertionError) as exc:
        manager.add_loras("pokemon-lora", tensors=weights)
        assert "already exists" in str(exc.value)


def test_add_multiple_loras(manager: SDLoraManager, weights: dict[str, torch.Tensor]) -> None:
    manager.add_multiple_loras({"pokemon-lora": weights, "pokemon-lora2": weights})
    assert "pokemon-lora" in manager.names
    assert "pokemon-lora2" in manager.names


def test_remove_loras(manager: SDLoraManager, weights: dict[str, torch.Tensor]) -> None:
    manager.add_multiple_loras({"pokemon-lora": weights, "pokemon-lora2": weights})
    manager.remove_loras("pokemon-lora")
    assert "pokemon-lora" not in manager.names
    assert "pokemon-lora2" in manager.names

    manager.remove_loras("pokemon-lora2")
    assert "pokemon-lora2" not in manager.names
    assert len(manager.names) == 0


def test_remove_all(manager: SDLoraManager, weights: dict[str, torch.Tensor]) -> None:
    manager.add_multiple_loras({"pokemon-lora": weights, "pokemon-lora2": weights})
    manager.remove_all()
    assert len(manager.names) == 0


def test_get_lora(manager: SDLoraManager, weights: dict[str, torch.Tensor]) -> None:
    manager.add_loras("pokemon-lora", tensors=weights)
    assert all(isinstance(lora, Lora) for lora in manager.get_loras_by_name("pokemon-lora"))


def test_get_scale(manager: SDLoraManager, weights: dict[str, torch.Tensor]) -> None:
    manager.add_loras("pokemon-lora", tensors=weights, scale=0.4)
    assert manager.get_scale("pokemon-lora") == 0.4


def test_names(manager: SDLoraManager, weights: dict[str, torch.Tensor]) -> None:
    assert manager.names == []

    manager.add_loras("pokemon-lora", tensors=weights)
    assert manager.names == ["pokemon-lora"]

    manager.add_loras("pokemon-lora2", tensors=weights)
    assert set(manager.names) == set(["pokemon-lora", "pokemon-lora2"])


def test_scales(manager: SDLoraManager, weights: dict[str, torch.Tensor]) -> None:
    assert manager.scales == {}

    manager.add_loras("pokemon-lora", tensors=weights, scale=0.4)
    assert manager.scales == {"pokemon-lora": 0.4}

    manager.add_loras("pokemon-lora2", tensors=weights, scale=0.5)
    assert manager.scales == {"pokemon-lora": 0.4, "pokemon-lora2": 0.5}
