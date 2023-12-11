# pyright: reportPrivateUsage=false
import pytest
import torch
from torch import Tensor, nn

import refiners.fluxion.layers as fl
from refiners.fluxion.model_converter import ConversionStage, ModelConverter
from refiners.fluxion.utils import manual_seed


class CustomBasicLayer1(fl.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(data=torch.randn(out_features, in_features))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.t()


class CustomBasicLayer2(fl.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(data=torch.randn(out_features, in_features))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.t()


# Source Model
class SourceModel(fl.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = fl.Linear(in_features=10, out_features=2)
        self.activation = fl.ReLU()
        self.custom_layers = nn.ModuleList(modules=[CustomBasicLayer1(in_features=2, out_features=2) for _ in range(3)])
        self.flatten = fl.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.conv = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        for layer in self.custom_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = x.view(1, 1, -1)
        x = self.conv(x)
        x = self.pool(x)
        return x


# Target Model (Purposely obfuscated but functionally equivalent)
class TargetModel(fl.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = fl.ReLU()
        self.drop = nn.Dropout(0.5)
        self.layers1 = nn.ModuleList(modules=[CustomBasicLayer2(in_features=2, out_features=2) for _ in range(3)])
        self.flattenIt = fl.Flatten()
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.convolution = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.lin = fl.Linear(in_features=10, out_features=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        x = self.relu(x)
        for layer in self.layers1:
            x = layer(x)
        x = self.flattenIt(x)
        x = self.drop(x)
        x = x.view(1, 1, -1)
        x = self.convolution(x)
        x = self.max_pool(x)
        return x


@pytest.fixture
def source_model() -> SourceModel:
    manual_seed(seed=2)
    return SourceModel()


@pytest.fixture
def target_model() -> TargetModel:
    manual_seed(seed=2)
    return TargetModel()


@pytest.fixture
def model_converter(source_model: SourceModel, target_model: TargetModel) -> ModelConverter:
    custom_layer_mapping: dict[type[nn.Module], type[nn.Module]] = {CustomBasicLayer1: CustomBasicLayer2}
    return ModelConverter(
        source_model=source_model, target_model=target_model, custom_layer_mapping=custom_layer_mapping, verbose=True
    )


@pytest.fixture
def random_tensor() -> Tensor:
    return torch.randn(1, 10)


@pytest.fixture
def source_args(random_tensor: Tensor) -> tuple[Tensor]:
    return (random_tensor,)


@pytest.fixture
def target_args(random_tensor: Tensor) -> tuple[Tensor]:
    return (random_tensor,)


def test_converter_stages(
    model_converter: ModelConverter, source_args: tuple[Tensor], target_args: tuple[Tensor]
) -> None:
    assert model_converter.stage == ConversionStage.INIT
    assert model_converter._run_init_stage()
    model_converter._increment_stage()

    assert model_converter.stage == ConversionStage.BASIC_LAYERS_MATCH
    assert model_converter._run_basic_layers_match_stage(source_args=source_args, target_args=target_args)
    model_converter._increment_stage()

    assert model_converter.stage == ConversionStage.SHAPE_AND_LAYERS_MATCH
    assert model_converter._run_shape_and_layers_match_stage(source_args=source_args, target_args=target_args)
    model_converter._increment_stage()

    assert model_converter.stage == ConversionStage.MODELS_OUTPUT_AGREE


def test_run(model_converter: ModelConverter, source_args: tuple[Tensor], target_args: tuple[Tensor]) -> None:
    assert model_converter.run(source_args=source_args, target_args=target_args)
    assert model_converter.stage == ConversionStage.MODELS_OUTPUT_AGREE
