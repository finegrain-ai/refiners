import sys
from importlib import import_module
from importlib.metadata import requires

from packaging.requirements import Requirement

refiners_requires = requires("refiners")
assert refiners_requires is not None

# Some dependencies have different module names than their package names
req_to_module: dict[str, str] = {
    "gitpython": "git",
}

for dep in refiners_requires:
    req = Requirement(dep)
    marker = req.marker
    if marker is None or not marker.evaluate({"extra": "training"}):
        continue

    module_name = req_to_module.get(req.name, req.name)

    try:
        import_module(module_name)
    except ImportError:
        print(
            f"Some dependencies are missing: {req.name}. Please install refiners with the `training` extra, e.g. `pip install"
            " refiners[training]`",
            file=sys.stderr,
        )
        sys.exit(1)
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
from refiners.training_utils.trainer import Trainer, register_callback, register_model
from refiners.training_utils.wandb import WandbConfig, WandbMixin

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
    "Optimizers",
    "LRSchedulerType",
]
