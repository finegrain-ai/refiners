import sys
from importlib import import_module
from importlib.metadata import requires

from packaging.requirements import Requirement

from refiners.training_utils.callback import Callback, CallbackConfig
from refiners.training_utils.clock import ClockConfig
from refiners.training_utils.config import (
    BaseConfig,
    LRSchedulerConfig,
    LRSchedulerType,
    ModelConfig,
    OptimizerConfig,
    Optimizers,
    TrainingConfig,
)
from refiners.training_utils.gradient_clipping import GradientClippingConfig
from refiners.training_utils.trainer import Trainer, register_callback, register_model
from refiners.training_utils.wandb import WandbConfig, WandbMixin

refiners_requires = requires("refiners")
assert refiners_requires is not None

for dep in refiners_requires:
    req = Requirement(dep)
    marker = req.marker
    if marker is None or not marker.evaluate({"extra": "training"}):
        continue
    try:
        import_module(req.name)
    except ImportError:
        print(
            "Some dependencies are missing. Please install refiners with the `training` extra, e.g. `pip install"
            " refiners[training]`",
            file=sys.stderr,
        )
        sys.exit(1)


__all__ = [
    "Trainer",
    "BaseConfig",
    "ModelConfig",
    "register_callback",
    "register_model",
    "Callback",
    "CallbackConfig",
    "WandbMixin",
    "WandbConfig",
    "LRSchedulerConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "ClockConfig",
    "GradientClippingConfig",
    "Optimizers",
    "LRSchedulerType",
]
