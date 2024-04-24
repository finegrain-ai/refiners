import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import torch
from pydantic import field_validator
from torch import Tensor, nn
from torch.optim import SGD

from refiners.fluxion import layers as fl
from refiners.fluxion.utils import norm
from refiners.training_utils.callback import Callback, CallbackConfig
from refiners.training_utils.common import (
    Epoch,
    Iteration,
    Step,
    TimeValue,
    TimeValueInput,
    count_learnable_parameters,
    human_readable_number,
    parse_number_unit_field,
    scoped_seed,
)
from refiners.training_utils.config import BaseConfig, ModelConfig
from refiners.training_utils.trainer import (
    Trainer,
    TrainingClock,
    WarmupScheduler,
    count_learnable_parameters,
    human_readable_number,
    register_callback,
    register_model,
)


@dataclass
class MockBatch:
    inputs: torch.Tensor
    targets: torch.Tensor


class MockModelConfig(ModelConfig):
    use_activation: bool


class MockCallbackConfig(CallbackConfig):
    on_batch_end_interval: Step | Iteration | Epoch
    on_batch_end_seed: int
    on_optimizer_step_interval: Iteration | Epoch

    @field_validator("on_batch_end_interval", "on_optimizer_step_interval", mode="before")
    def parse_field(cls, value: TimeValueInput) -> TimeValue:
        return parse_number_unit_field(value)


class MockConfig(BaseConfig):
    mock_model: MockModelConfig
    mock_callback: MockCallbackConfig


class MockModel(fl.Chain):
    def __init__(self):
        super().__init__(
            fl.Linear(10, 10),
            fl.Linear(10, 10),
            fl.Linear(10, 10),
        )

    def add_activation(self) -> None:
        self.insert(1, fl.SiLU())
        self.insert(3, fl.SiLU())


class MockCallback(Callback["MockTrainer"]):
    def __init__(self, config: MockCallbackConfig) -> None:
        self.config = config
        self.optimizer_step_count = 0
        self.step_end_count = 0
        self.optimizer_step_random_int: int | None = None
        self.step_end_random_int: int | None = None

    def on_init_begin(self, trainer: "MockTrainer") -> None:
        pass

    def on_optimizer_step_begin(self, trainer: "MockTrainer") -> None:
        if not trainer.clock.is_due(self.config.on_optimizer_step_interval):
            return
        self.optimizer_step_count += 1
        self.optimizer_step_random_int = random.randint(0, 100)

    def on_step_end(self, trainer: "MockTrainer") -> None:
        if not trainer.clock.is_due(self.config.on_batch_end_interval):
            return

        # We verify that the callback is always called before the clock is updated
        assert trainer.clock.step // 3 <= self.step_end_count

        self.step_end_count += 1
        with scoped_seed(self.config.on_batch_end_seed):
            self.step_end_random_int = random.randint(0, 100)


class MockTrainer(Trainer[MockConfig, MockBatch]):
    step_counter: int = 0
    model_registration_counter: int = 0

    @property
    def dataset_length(self) -> int:
        return 20

    def get_item(self, index: int) -> MockBatch:
        return MockBatch(inputs=torch.randn(1, 10), targets=torch.randn(1, 10))

    def collate_fn(self, batch: list[MockBatch]) -> MockBatch:
        return MockBatch(
            inputs=torch.cat([b.inputs for b in batch]),
            targets=torch.cat([b.targets for b in batch]),
        )

    @register_callback()
    def mock_callback(self, config: MockCallbackConfig) -> MockCallback:
        return MockCallback(config)

    @register_model()
    def mock_model(self, config: MockModelConfig) -> MockModel:
        model = MockModel()
        if config.use_activation:
            model.add_activation()

        self.model_registration_counter += 1
        return model

    def compute_loss(self, batch: MockBatch) -> Tensor:
        self.step_counter += 1
        inputs, targets = batch.inputs.to(self.device), batch.targets.to(self.device)
        outputs = self.mock_model(inputs)
        return norm(outputs - targets)


@pytest.fixture
def mock_config() -> MockConfig:
    config = MockConfig.load_from_toml(Path(__file__).parent / "mock_config.toml")
    return config


@pytest.fixture
def mock_trainer(mock_config: MockConfig) -> MockTrainer:
    return MockTrainer(config=mock_config)


@pytest.fixture
def mock_trainer_short(mock_config: MockConfig) -> MockTrainer:
    mock_config_short = mock_config.model_copy(deep=True)
    mock_config_short.training.duration = Step(3)
    return MockTrainer(config=mock_config_short)


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
        training_duration=Epoch(5),
        gradient_accumulation=Epoch(1),
        lr_scheduler_interval=Epoch(1),
    )


def test_small_dataset_error():
    with pytest.raises(AssertionError):
        TrainingClock(
            dataset_length=3,
            batch_size=10,
            training_duration=Epoch(5),
            gradient_accumulation=Epoch(1),
            lr_scheduler_interval=Epoch(1),
        )


def test_zero_batch_size_error():
    with pytest.raises(AssertionError):
        TrainingClock(
            dataset_length=3,
            batch_size=0,
            training_duration=Epoch(5),
            gradient_accumulation=Epoch(1),
            lr_scheduler_interval=Epoch(1),
        )


def test_time_unit_to_steps_conversion(training_clock: TrainingClock) -> None:
    assert training_clock.convert_time_value_to_steps(Epoch(1)) == 10
    assert training_clock.convert_time_value_to_steps(Epoch(2)) == 20
    assert training_clock.convert_time_value_to_steps(Step(1)) == 1
    assert training_clock.convert_time_value_to_steps(Iteration(1)) == 10


def test_steps_to_time_unit_conversion(training_clock: TrainingClock) -> None:
    assert training_clock.convert_steps_to_time_unit(10, Epoch) == 1
    assert training_clock.convert_steps_to_time_unit(20, Epoch) == 2
    assert training_clock.convert_steps_to_time_unit(1, Step) == 1
    assert training_clock.convert_steps_to_time_unit(10, Iteration) == 1


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


def test_mock_trainer_initialization(mock_config: MockConfig, mock_trainer: MockTrainer) -> None:
    assert mock_trainer.config == mock_config
    assert isinstance(mock_trainer, MockTrainer)
    assert mock_trainer.optimizer is not None
    assert mock_trainer.lr_scheduler is not None
    assert mock_trainer.model_registration_counter == 1


def test_training_cycle(mock_trainer: MockTrainer) -> None:
    clock = mock_trainer.clock
    config = mock_trainer.config

    assert clock.num_step_per_iteration == config.training.gradient_accumulation.number
    assert clock.num_batches_per_epoch == mock_trainer.dataset_length // config.training.batch_size

    assert mock_trainer.step_counter == 0
    assert clock.epoch == 0

    mock_trainer.train()

    assert clock.epoch == config.training.duration.number
    assert clock.step == config.training.duration.number * clock.num_batches_per_epoch

    assert mock_trainer.step_counter == mock_trainer.clock.step


def test_callback_registration(mock_trainer: MockTrainer) -> None:
    mock_trainer.train()

    # Check that the callback skips every other iteration
    assert mock_trainer.mock_callback.optimizer_step_count == mock_trainer.clock.iteration // 2
    assert mock_trainer.mock_callback.step_end_count == mock_trainer.clock.step // 3

    # Check that the random seed was set
    assert mock_trainer.mock_callback.optimizer_step_random_int == 93
    assert mock_trainer.mock_callback.step_end_random_int == 81


def test_training_short_cycle(mock_trainer_short: MockTrainer) -> None:
    clock = mock_trainer_short.clock
    config = mock_trainer_short.config

    assert mock_trainer_short.step_counter == 0
    assert mock_trainer_short.clock.epoch == 0

    mock_trainer_short.train()

    assert clock.step == config.training.duration.number


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
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r"Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`",
        )
        for _ in range(102):
            warmup_scheduler.step()
    optimizer = warmup_scheduler.optimizer
    for group in optimizer.param_groups:
        assert group["lr"] == 0.1


class MockTrainerWith2Models(MockTrainer):
    @register_model()
    def mock_model1(self, config: ModelConfig) -> MockModel:
        return MockModel()

    @register_model()
    def mock_model2(self, config: ModelConfig) -> MockModel:
        return MockModel()

    def compute_loss(self, batch: MockBatch) -> Tensor:
        self.step_counter += 1
        inputs, targets = batch.inputs.to(self.device), batch.targets.to(self.device)
        outputs = self.mock_model2(self.mock_model1(inputs))
        return norm(outputs - targets)


class MockConfig_2_Models(BaseConfig):
    mock_model1: ModelConfig
    mock_model2: ModelConfig


@pytest.fixture
def mock_config_2_models() -> MockConfig_2_Models:
    return MockConfig_2_Models.load_from_toml(Path(__file__).parent / "mock_config_2_models.toml")


@pytest.fixture
def mock_trainer_2_models(mock_config_2_models: MockConfig) -> MockTrainerWith2Models:
    return MockTrainerWith2Models(config=mock_config_2_models)


def test_optimizer_parameters(mock_trainer_2_models: MockTrainerWith2Models) -> None:
    assert len(mock_trainer_2_models.optimizer.param_groups) == 2
    assert mock_trainer_2_models.optimizer.param_groups[0]["lr"] == 1e-5
