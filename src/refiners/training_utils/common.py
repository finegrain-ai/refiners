import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

import numpy as np
import torch
from loguru import logger
from torch import cuda, nn

from refiners.fluxion.utils import manual_seed


def compute_grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    """
    Computes the gradient norm of the parameters in the given iterable.

    We use the `torch.nn.utils.clip_grad_norm_` function to process the gradients efficiently on the GPU or CPU.
    """
    return nn.utils.clip_grad.clip_grad_norm_(parameters, float("inf")).item()


def count_learnable_parameters(parameters: Iterable[nn.Parameter]) -> int:
    return sum(p.numel() for p in parameters if p.requires_grad)


def human_readable_number(number: int) -> str:
    float_number = float(number)
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(float_number) < 1000:
            return f"{float_number:.1f}{unit}"
        float_number /= 1000
    return f"{float_number:.1f}E"


def seed_everything(seed: int | None = None) -> None:
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    random.seed(a=seed)
    np.random.seed(seed=seed)
    manual_seed(seed=seed)
    cuda.manual_seed_all(seed=seed)


class scoped_seed:
    """
    Context manager and decorator to set a fixed seed within a specific scope.

    The seed can be provided directly or as a callable that takes the same arguments
    as the decorated function. Supports setting the seed for random, numpy, torch,
    and torch.cuda modules.
    """

    def __init__(self, seed: int | Callable[..., int] | None = None):
        self.seed = seed
        self.actual_seed: int | None = None

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        def inner_wrapper(*args: Any, **kwargs: Any) -> Any:
            self.actual_seed = self.seed(*args, **kwargs) if callable(self.seed) else self.seed
            with self:
                return func(*args, **kwargs)

        return inner_wrapper

    def __enter__(self) -> None:
        if self.actual_seed is None:
            seed = self.seed() if callable(self.seed) else self.seed
        else:
            seed = self.actual_seed
        self.random_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        self.cuda_torch_state = cuda.get_rng_state()
        seed_everything(seed)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        logger.trace(f"Restoring previous seed state")
        random.setstate(self.random_state)
        np.random.set_state(self.numpy_state)
        torch.set_rng_state(self.torch_state)
        cuda.set_rng_state(self.cuda_torch_state)


@dataclass
@runtime_checkable
class TimeValue(Protocol):
    number: int

    @property
    def unit(self) -> "TimeUnit":
        match self.__class__.__name__:
            case "Step":
                return Step
            case "Epoch":
                return Epoch
            case "Iteration":
                return Iteration
            case _:
                raise ValueError(f"Unsupported time unit: {self.__class__.__name__}")

    @classmethod
    def from_str(cls, value: str) -> "TimeValue":
        match cls.extract_number_unit(value):
            case number, "step":
                return Step(number)
            case number, "epoch":
                return Epoch(number)
            case number, "iteration":
                return Iteration(number)
            case _:
                raise ValueError(f"Incorrect time value format: {value}")

    @staticmethod
    def extract_number_unit(value: str) -> tuple[int, str]:
        number, unit = value.lower().split(":")
        return int(number.strip()), unit.strip()


@dataclass
class Step(TimeValue):
    number: int


@dataclass
class Epoch(TimeValue):
    number: int


@dataclass
class Iteration(TimeValue):
    number: int


TimeUnit = type[Step] | type[Epoch] | type[Iteration]
TimeValueInput = str | int | dict[str, str | int] | TimeValue


def parse_number_unit_field(value: TimeValueInput) -> TimeValue:
    match value:
        case str(value_str):
            return TimeValue.from_str(value_str)
        case int(number):
            return Step(number=number)
        case TimeValue(number):
            return value
        case _:
            raise ValueError(f"Unsupported value format: {value}")
