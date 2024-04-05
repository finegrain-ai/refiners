import random
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Iterable

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
    logger.info(f"Using random seed: {seed}")
    random.seed(a=seed)
    np.random.seed(seed=seed)
    manual_seed(seed=seed)
    cuda.manual_seed_all(seed=seed)


def scoped_seed(seed: int | Callable[..., int] | None = None) -> Callable[..., Callable[..., Any]]:
    """
    Decorator for setting a random seed within the scope of a function.

    This decorator sets the random seed for Python's built-in `random` module,
    `numpy`, and `torch` and `torch.cuda` at the beginning of the decorated function. After the
    function is executed, it restores the state of the random number generators
    to what it was before the function was called. This is useful for ensuring
    reproducibility for specific parts of the code without affecting randomness
    elsewhere.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def inner_wrapper(*args: Any, **kwargs: Any) -> Any:
            random_state = random.getstate()
            numpy_state = np.random.get_state()
            torch_state = torch.get_rng_state()
            cuda_torch_state = cuda.get_rng_state()
            actual_seed = seed(*args) if callable(seed) else seed
            seed_everything(seed=actual_seed)
            result = func(*args, **kwargs)
            logger.trace(f"Restoring previous seed state")
            random.setstate(random_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            cuda.set_rng_state(cuda_torch_state)
            return result

        return inner_wrapper

    return decorator


class TimeUnit(str, Enum):
    STEP = "step"
    EPOCH = "epoch"
    ITERATION = "iteration"
    DEFAULT = "step"


@dataclass
class TimeValue:
    number: int
    unit: TimeUnit


TimeValueInput = str | int | dict[str, str | int] | TimeValue


def parse_number_unit_field(value: TimeValueInput) -> TimeValue:
    match value:
        case str(value_str):
            number, unit = value_str.split(sep=":")
            return TimeValue(number=int(number.strip()), unit=TimeUnit(value=unit.strip().lower()))
        case int(number):
            return TimeValue(number=number, unit=TimeUnit.DEFAULT)
        case {"number": int(number), "unit": str(unit)}:
            return TimeValue(number=number, unit=TimeUnit(value=unit.lower()))
        case TimeValue(number, unit):
            return TimeValue(number=number, unit=unit)
        case _:
            raise ValueError(f"Unsupported value format: {value}")
