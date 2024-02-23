from enum import Enum
from logging import warn
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Type, TypeVar

import tomli
from bitsandbytes.optim import AdamW8bit, Lion8bit  # type: ignore
from prodigyopt import Prodigy  # type: ignore
from pydantic import BaseModel, ConfigDict, field_validator
from torch import Tensor
from torch.optim import SGD, Adam, AdamW, Optimizer

from refiners.training_utils.clock import ClockConfig
from refiners.training_utils.common import TimeUnit, TimeValue, parse_number_unit_field
from refiners.training_utils.gradient_clipping import GradientClippingConfig

# PyTorch optimizer parameters type
# TODO: replace with `from torch.optim.optimizer import ParamsT` when PyTorch 2.2+ is enforced
# See https://github.com/pytorch/pytorch/pull/111114
ParamsT = Iterable[Tensor] | Iterable[dict[str, Any]]


class TrainingConfig(BaseModel):
    device: str = "cpu"
    amp: bool = False
    dtype: str = "float32"
    duration: TimeValue = {"number": 1, "unit": TimeUnit.ITERATION}
    seed: int = 0
    batch_size: int = 1
    gradient_accumulation: TimeValue = {"number": 1, "unit": TimeUnit.STEP}
    evaluation_interval: TimeValue = {"number": 1, "unit": TimeUnit.ITERATION}
    evaluation_seed: int = 0

    model_config = ConfigDict(extra="forbid")

    @field_validator("duration", "gradient_accumulation", "evaluation_interval", mode="before")
    def parse_field(cls, value: Any) -> TimeValue:
        return parse_number_unit_field(value)


class Optimizers(str, Enum):
    SGD = "SGD"
    Adam = "Adam"
    AdamW = "AdamW"
    AdamW8bit = "AdamW8bit"
    Lion8bit = "Lion8bit"
    Prodigy = "Prodigy"


class LRSchedulerType(str, Enum):
    STEP_LR = "StepLR"
    EXPONENTIAL_LR = "ExponentialLR"
    REDUCE_LR_ON_PLATEAU = "ReduceLROnPlateau"
    COSINE_ANNEALING_LR = "CosineAnnealingLR"
    CONSTANT_LR = "ConstantLR"  # not to be confused with PyTorch's ConstantLR
    LAMBDA_LR = "LambdaLR"
    ONE_CYCLE_LR = "OneCycleLR"
    MULTIPLICATIVE_LR = "MultiplicativeLR"
    COSINE_ANNEALING_WARM_RESTARTS = "CosineAnnealingWarmRestarts"
    CYCLIC_LR = "CyclicLR"
    MULTI_STEP_LR = "MultiStepLR"
    DEFAULT = "ConstantLR"


class LRSchedulerConfig(BaseModel):
    type: LRSchedulerType = LRSchedulerType.DEFAULT
    update_interval: TimeValue = {"number": 1, "unit": TimeUnit.ITERATION}
    warmup: TimeValue = {"number": 0, "unit": TimeUnit.ITERATION}
    gamma: float = 0.1
    lr_lambda: Callable[[int], float] | None = None
    mode: Literal["min", "max"] = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    cooldown: int = 0
    milestones: list[int] = []
    base_lr: float = 1e-7
    min_lr: float | list[float] = 0
    max_lr: float | list[float] = 0
    eta_min: float = 0

    model_config = ConfigDict(extra="forbid")

    @field_validator("update_interval", "warmup", mode="before")
    def parse_field(cls, value: Any) -> TimeValue:
        return parse_number_unit_field(value)


class OptimizerConfig(BaseModel):
    optimizer: Optimizers
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

    model_config = ConfigDict(extra="forbid")

    def get(self, params: ParamsT) -> Optimizer:
        match self.optimizer:
            case Optimizers.SGD:
                return SGD(
                    params=params,
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )
            case Optimizers.Adam:
                return Adam(
                    params=params,
                    lr=self.learning_rate,
                    betas=self.betas,
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                )
            case Optimizers.AdamW:
                return AdamW(
                    params=params,
                    lr=self.learning_rate,
                    betas=self.betas,
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                )
            case Optimizers.AdamW8bit:
                return AdamW8bit(
                    params=params,
                    lr=self.learning_rate,
                    betas=self.betas,
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                )
            case Optimizers.Lion8bit:
                return Lion8bit(
                    params=params,
                    lr=self.learning_rate,
                    betas=self.betas,
                    weight_decay=self.weight_decay,  # type: ignore
                )
            case Optimizers.Prodigy:
                if self.learning_rate != 1.0:
                    warn("Prodigy learning rate is not 1.0, this might cause instability.")
                return Prodigy(
                    lr=self.learning_rate,
                    params=params,
                    betas=self.betas,
                    weight_decay=self.weight_decay,  # type: ignore
                    safeguard_warmup=True,
                    use_bias_correction=True,  # recommended for diffusion models
                )


class ModelConfig(BaseModel):
    # If None, then requires_grad will NOT be changed when loading the model
    # this can be useful if you want to train only a part of the model
    requires_grad: bool | None = None
    # Optional, per-model optimizer parameters
    learning_rate: float | None = None
    betas: tuple[float, float] | None = None
    eps: float | None = None
    weight_decay: float | None = None

    model_config = ConfigDict(extra="forbid")


T = TypeVar("T", bound="BaseConfig")


class BaseConfig(BaseModel):
    training: TrainingConfig
    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig
    clock: ClockConfig = ClockConfig()
    gradient_clipping: GradientClippingConfig = GradientClippingConfig()

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def load_from_toml(cls: Type[T], toml_path: Path | str) -> T:
        with open(file=toml_path, mode="rb") as f:
            config_dict = tomli.load(f)

        return cls(**config_dict)
