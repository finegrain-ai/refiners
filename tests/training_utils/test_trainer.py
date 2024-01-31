from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import cast
from warnings import warn

import pytest
import torch
from torch import Tensor, nn
from torch.optim import SGD
from torch.utils.data import Dataset

from refiners.fluxion import layers as fl
from refiners.fluxion.utils import norm
from refiners.training_utils.config import BaseConfig, TimeUnit
from refiners.training_utils.trainer import (
    Trainer,
    TrainingClock,
    WarmupScheduler,
    count_learnable_parameters,
    human_readable_number,
)


@dataclass
class MockBatch:
    inputs: torch.Tensor
    targets: torch.Tensor


class MockDataset(Dataset[MockBatch]):
    def __len__(self):
        return 20

    def __getitem__(self, _: int) -> MockBatch:
        return MockBatch(inputs=torch.randn(1, 10), targets=torch.randn(1, 10))

    def collate_fn(self, batch: list[MockBatch]) -> MockBatch:
        return MockBatch(
            inputs=torch.cat([b.inputs for b in batch]),
            targets=torch.cat([b.targets for b in batch]),
        )


class MockConfig(BaseConfig):
    pass


class MockModel(fl.Chain):
    def __init__(self):
        super().__init__(
            fl.Linear(10, 10),
            fl.Linear(10, 10),
            fl.Linear(10, 10),
        )


class MockTrainer(Trainer[MockConfig, MockBatch]):
    step_counter: int = 0

    @cached_property
    def mock_model(self) -> MockModel:
        return MockModel()

    def load_dataset(self) -> Dataset[MockBatch]:
        return MockDataset()

    def load_models(self) -> dict[str, fl.Module]:
        return {"mock_model": self.mock_model}

    def compute_loss(self, batch: MockBatch) -> Tensor:
        self.step_counter += 1
        inputs, targets = batch.inputs.to(self.device), batch.targets.to(self.device)
        outputs = self.mock_model(inputs)
        return norm(outputs - targets)


@pytest.fixture
def mock_config(test_device: torch.device) -> MockConfig:
    if not test_device.type == "cuda":
        warn("only running on CUDA, skipping")
        pytest.skip("Skipping test because test_device is not CUDA")
    config = MockConfig.load_from_toml(Path(__file__).parent / "mock_config.toml")
    config.training.gpu_index = test_device.index
    return config


@pytest.fixture
def mock_trainer(mock_config: MockConfig) -> MockTrainer:
    return MockTrainer(config=mock_config)


@pytest.fixture
def mock_model() -> fl.Chain:
    return MockModel()


def test_count_learnable_parameters_with_params() -> None:
    params = [
        nn.Parameter(torch.randn(2, 2), requires_grad=True),
        nn.Parameter(torch.randn(5), requires_grad=False),
        nn.Parameter(torch.randn(3, 3), requires_grad=True),
    ]
    # cast because of PyTorch 2.2, see https://github.com/pytorch/pytorch/issues/118736
    assert count_learnable_parameters(cast(list[nn.Parameter], params)) == 13


def test_count_learnable_parameters_with_model(mock_model: fl.Chain) -> None:
    assert count_learnable_parameters(mock_model.parameters()) == 330


def test_human_readable_number() -> None:
    assert human_readable_number(123) == "123.0"
    assert human_readable_number(1234) == "1.2K"
    assert human_readable_number(1234567) == "1.2M"


@pytest.fixture
def training_clock() -> TrainingClock:
    return TrainingClock(
        dataset_length=100,
        batch_size=10,
        training_duration={"number": 5, "unit": TimeUnit.EPOCH},
        gradient_accumulation={"number": 1, "unit": TimeUnit.EPOCH},
        evaluation_interval={"number": 1, "unit": TimeUnit.EPOCH},
        lr_scheduler_interval={"number": 1, "unit": TimeUnit.EPOCH},
        checkpointing_save_interval={"number": 1, "unit": TimeUnit.EPOCH},
    )


def test_time_unit_to_steps_conversion(training_clock: TrainingClock) -> None:
    assert training_clock.convert_time_unit_to_steps(1, TimeUnit.EPOCH) == 10
    assert training_clock.convert_time_unit_to_steps(2, TimeUnit.EPOCH) == 20
    assert training_clock.convert_time_unit_to_steps(1, TimeUnit.STEP) == 1


def test_steps_to_time_unit_conversion(training_clock: TrainingClock) -> None:
    assert training_clock.convert_steps_to_time_unit(10, TimeUnit.EPOCH) == 1
    assert training_clock.convert_steps_to_time_unit(20, TimeUnit.EPOCH) == 2
    assert training_clock.convert_steps_to_time_unit(1, TimeUnit.STEP) == 1


def test_clock_properties(training_clock: TrainingClock) -> None:
    assert training_clock.num_batches_per_epoch == 10
    assert training_clock.num_epochs == 5
    assert training_clock.num_iterations == 5
    assert training_clock.num_steps == 50


def test_timer_functionality(training_clock: TrainingClock) -> None:
    training_clock.start_timer()
    assert training_clock.start_time is not None
    training_clock.stop_timer()
    assert training_clock.end_time is not None
    assert training_clock.time_elapsed >= 0


def test_state_based_properties(training_clock: TrainingClock) -> None:
    training_clock.step = 5  # Halfway through the first epoch
    assert not training_clock.is_evaluation_step  # Assuming evaluation every epoch
    assert not training_clock.is_checkpointing_step
    training_clock.step = 10  # End of the first epoch
    assert training_clock.is_evaluation_step
    assert training_clock.is_checkpointing_step


def test_mock_trainer_initialization(mock_config: MockConfig, mock_trainer: MockTrainer) -> None:
    assert mock_trainer.config == mock_config
    assert isinstance(mock_trainer, MockTrainer)
    assert mock_trainer.optimizer is not None
    assert mock_trainer.lr_scheduler is not None


def test_training_cycle(mock_trainer: MockTrainer) -> None:
    clock = mock_trainer.clock
    config = mock_trainer.config

    assert clock.num_step_per_iteration == config.training.gradient_accumulation["number"]
    assert clock.num_batches_per_epoch == mock_trainer.dataset_length // config.training.batch_size

    assert mock_trainer.step_counter == 0
    assert mock_trainer.clock.epoch == 0

    mock_trainer.train()

    assert clock.epoch == config.training.duration["number"]
    assert clock.step == config.training.duration["number"] * clock.num_batches_per_epoch

    assert mock_trainer.step_counter == mock_trainer.clock.step


@pytest.fixture
def warmup_scheduler():
    optimizer = SGD([nn.Parameter(torch.randn(2, 2), requires_grad=True)], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, 1)
    return WarmupScheduler(optimizer, scheduler, warmup_scheduler_steps=100)


def test_initial_lr(warmup_scheduler: WarmupScheduler) -> None:
    optimizer = warmup_scheduler.optimizer
    for group in optimizer.param_groups:
        assert group["lr"] == 1e-3


def test_warmup_lr(warmup_scheduler: WarmupScheduler) -> None:
    for _ in range(102):
        warmup_scheduler.step()
    optimizer = warmup_scheduler.optimizer
    for group in optimizer.param_groups:
        assert group["lr"] == 0.1
