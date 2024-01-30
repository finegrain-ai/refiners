from enum import Enum
from logging import warn
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Type, TypeVar

import tomli
from bitsandbytes.optim import AdamW8bit, Lion8bit  # type: ignore
from prodigyopt import Prodigy  # type: ignore
from pydantic import BaseModel, validator
from torch.nn import Parameter
from torch.optim import SGD, Adam, AdamW, Optimizer
from typing_extensions import TypedDict  # https://errors.pydantic.dev/2.0b3/u/typed-dict-version

import refiners.fluxion.layers as fl
from refiners.training_utils.dropout import apply_dropout, apply_gyro_dropout

__all__ = [
    "parse_number_unit_field",
    "TimeUnit",
    "TimeValue",
    "TrainingConfig",
    "OptimizerConfig",
    "Optimizers",
]


class TimeUnit(Enum):
    STEP = "step"
    EPOCH = "epoch"
    ITERATION = "iteration"
    DEFAULT = "step"


class TimeValue(TypedDict):
    number: int
    unit: TimeUnit


def parse_number_unit_field(value: str | int | dict[str, str | int]) -> TimeValue:
    match value:
        case str(value_str):
            number, unit = value_str.split(sep=":")
            return {"number": int(number.strip()), "unit": TimeUnit(value=unit.strip().lower())}
        case int(number):
            return {"number": number, "unit": TimeUnit.DEFAULT}
        case {"number": int(number), "unit": str(unit)}:
            return {"number": number, "unit": TimeUnit(value=unit.lower())}
        case _:
            raise ValueError(f"Unsupported value format: {value}")


class TrainingConfig(BaseModel):
    duration: TimeValue = {"number": 1, "unit": TimeUnit.ITERATION}
    seed: int = 0
    gpu_index: int = 0
    batch_size: int = 1
    gradient_accumulation: TimeValue = {"number": 1, "unit": TimeUnit.STEP}
    clip_grad_norm: float | None = None
    clip_grad_value: float | None = None
    evaluation_interval: TimeValue = {"number": 1, "unit": TimeUnit.ITERATION}
    evaluation_seed: int = 0

    @validator("duration", "gradient_accumulation", "evaluation_interval", pre=True)
    def parse_field(cls, value: Any) -> TimeValue:
        return parse_number_unit_field(value)


class Optimizers(str, Enum):
    SGD = "SGD"
    Adam = "Adam"
    AdamW = "AdamW"
    AdamW8bit = "AdamW8bit"
    Lion8bit = "Lion8bit"
    Prodigy = "Prodigy"


class SchedulerType(str, Enum):
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


class SchedulerConfig(BaseModel):
    scheduler_type: SchedulerType = SchedulerType.DEFAULT
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

    @validator("update_interval", "warmup", pre=True)
    def parse_field(cls, value: Any) -> TimeValue:
        return parse_number_unit_field(value)


class OptimizerConfig(BaseModel):
    optimizer: Optimizers
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

    def get(self, model_parameters: Iterable[Parameter]) -> Optimizer:
        match self.optimizer:
            case Optimizers.SGD:
                return SGD(
                    params=model_parameters,
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )
            case Optimizers.Adam:
                return Adam(
                    params=model_parameters,
                    lr=self.learning_rate,
                    betas=self.betas,
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                )
            case Optimizers.AdamW:
                return AdamW(
                    params=model_parameters,
                    lr=self.learning_rate,
                    betas=self.betas,
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                )
            case Optimizers.AdamW8bit:
                return AdamW8bit(
                    params=model_parameters,
                    lr=self.learning_rate,
                    betas=self.betas,
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                )
            case Optimizers.Lion8bit:
                return Lion8bit(
                    params=model_parameters,
                    lr=self.learning_rate,
                    betas=self.betas,
                    weight_decay=self.weight_decay,  # type: ignore
                )
            case Optimizers.Prodigy:
                if self.learning_rate != 1.0:
                    warn("Prodigy learning rate is not 1.0, this might cause instability.")
                return Prodigy(
                    lr=self.learning_rate,
                    params=model_parameters,
                    betas=self.betas,
                    weight_decay=self.weight_decay,  # type: ignore
                    safeguard_warmup=True,
                    use_bias_correction=True,  # recommended for diffusion models
                )


class ModelConfig(BaseModel):
    checkpoint: Path | None = None
    train: bool = True
    learning_rate: float | None = None  # TODO: Implement this


class GyroDropoutConfig(BaseModel):
    total_subnetworks: int = 512
    concurrent_subnetworks: int = 64
    iters_per_epoch: int = 512
    num_features_threshold: float = 5e5


class DropoutConfig(BaseModel):
    dropout_probability: float = 0.0
    gyro_dropout: GyroDropoutConfig | None = None

    def apply_dropout(self, model: fl.Chain) -> None:
        if self.dropout_probability > 0.0:
            if self.gyro_dropout is not None:
                apply_gyro_dropout(module=model, probability=self.dropout_probability, **self.gyro_dropout.model_dump())
            else:
                apply_dropout(module=model, probability=self.dropout_probability)


class WandbConfig(BaseModel):
    mode: Literal["online", "offline", "disabled"] = "online"
    project: str
    entity: str = "finegrain"
    name: str | None = None
    tags: list[str] = []
    group: str | None = None
    job_type: str | None = None
    notes: str | None = None


class CheckpointingConfig(BaseModel):
    save_folder: Path | None = None
    save_interval: TimeValue = {"number": 1, "unit": TimeUnit.EPOCH}

    @validator("save_interval", pre=True)
    def parse_field(cls, value: Any) -> TimeValue:
        return parse_number_unit_field(value)


T = TypeVar("T", bound="BaseConfig")


class BaseConfig(BaseModel):
    models: dict[str, ModelConfig]
    wandb: WandbConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    dropout: DropoutConfig
    checkpointing: CheckpointingConfig

    @classmethod
    def load_from_toml(cls: Type[T], toml_path: Path | str) -> T:
        with open(file=toml_path, mode="rb") as f:
            config_dict = tomli.load(f)

        return cls(**config_dict)
