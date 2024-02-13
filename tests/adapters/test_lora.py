import pytest
import torch

from refiners.fluxion import layers as fl
from refiners.fluxion.adapters.lora import Conv2dLora, LinearLora, Lora, LoraAdapter


@pytest.fixture
def lora() -> LinearLora:
    return LinearLora("test", in_features=320, out_features=128, rank=16)


@pytest.fixture
def conv_lora() -> Conv2dLora:
    return Conv2dLora("conv_test", in_channels=16, out_channels=8, kernel_size=(3, 1), rank=4)


def test_properties(lora: LinearLora, conv_lora: Conv2dLora) -> None:
    assert lora.name == "test"
    assert lora.rank == lora.down.out_features == lora.up.in_features == 16
    assert lora.scale == 1.0
    assert lora.in_features == lora.down.in_features == 320
    assert lora.out_features == lora.up.out_features == 128

    assert conv_lora.name == "conv_test"
    assert conv_lora.rank == conv_lora.down.out_channels == conv_lora.up.in_channels == 4
    assert conv_lora.scale == 1.0
    assert conv_lora.in_channels == conv_lora.down.in_channels == 16
    assert conv_lora.out_channels == conv_lora.up.out_channels == 8
    assert conv_lora.kernel_size == (conv_lora.down.kernel_size[0], conv_lora.up.kernel_size[0]) == (3, 1)
    # padding is set so the spatial dimensions are preserved
    assert conv_lora.padding == (conv_lora.down.padding[0], conv_lora.up.padding[0]) == (0, 1)


def test_scale_setter(lora: LinearLora) -> None:
    lora.scale = 2.0
    assert lora.scale == 2.0
    assert lora.ensure_find(fl.Multiply).scale == 2.0


def test_from_weights(lora: LinearLora, conv_lora: Conv2dLora) -> None:
    new_lora = LinearLora.from_weights("test", down=lora.down.weight, up=lora.up.weight)
    x = torch.randn(1, 320)
    assert torch.allclose(lora(x), new_lora(x))

    new_conv_lora = Conv2dLora.from_weights("conv_test", down=conv_lora.down.weight, up=conv_lora.up.weight)
    x = torch.randn(1, 16, 64, 64)
    assert torch.allclose(conv_lora(x), new_conv_lora(x))


def test_from_dict() -> None:
    state_dict = {
        "down.weight": torch.randn(320, 128),
        "up.weight": torch.randn(128, 320),
        "this.is_not_used.alpha": torch.randn(1, 320),
        "probably.a.conv.down.weight": torch.randn(4, 16, 3, 3),
        "probably.a.conv.up.weight": torch.randn(8, 4, 1, 1),
    }
    loras = Lora.from_dict("test", state_dict=state_dict)
    assert len(loras) == 2
    linear_lora, conv_lora = loras.values()
    assert isinstance(linear_lora, LinearLora)
    assert isinstance(conv_lora, Conv2dLora)
    assert linear_lora.name == "test"
    assert conv_lora.name == "test"


@pytest.fixture
def lora_adapter() -> LoraAdapter:
    target = fl.Linear(320, 128)
    lora1 = LinearLora("test1", in_features=320, out_features=128, rank=16, scale=2.0)
    lora2 = LinearLora("test2", in_features=320, out_features=128, rank=16, scale=-1.0)
    return LoraAdapter(target, lora1, lora2)


def test_names(lora_adapter: LoraAdapter) -> None:
    assert set(lora_adapter.names) == {"test1", "test2"}


def test_loras(lora_adapter: LoraAdapter) -> None:
    assert set(lora_adapter.loras.keys()) == {"test1", "test2"}


def test_scales(lora_adapter: LoraAdapter) -> None:
    assert set(lora_adapter.scales.keys()) == {"test1", "test2"}
    assert lora_adapter.scales["test1"] == 2.0
    assert lora_adapter.scales["test2"] == -1.0


def test_scale_setter_lora_adapter(lora_adapter: LoraAdapter) -> None:
    lora_adapter.scale = {"test1": 0.0, "test2": 3.0}
    assert lora_adapter.scales == {"test1": 0.0, "test2": 3.0}


def test_add_lora(lora_adapter: LoraAdapter) -> None:
    lora3 = LinearLora("test3", in_features=320, out_features=128, rank=16)
    lora_adapter.add_lora(lora3)
    assert "test3" in lora_adapter.names


def test_remove_lora(lora_adapter: LoraAdapter) -> None:
    lora_adapter.remove_lora("test1")
    assert "test1" not in lora_adapter.names


def test_add_existing_lora(lora_adapter: LoraAdapter) -> None:
    lora3 = LinearLora("test1", in_features=320, out_features=128, rank=16)
    with pytest.raises(AssertionError):
        lora_adapter.add_lora(lora3)


def test_remove_nonexistent_lora(lora_adapter: LoraAdapter) -> None:
    assert lora_adapter.remove_lora("test3") is None


def test_set_scale_for_nonexistent_lora(lora_adapter: LoraAdapter) -> None:
    with pytest.raises(KeyError):
        lora_adapter.scale = {"test3": 2.0}
