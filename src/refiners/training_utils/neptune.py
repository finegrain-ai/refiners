from abc import ABC
from pathlib import Path
from typing import Any, Literal

from neptune import Run, init_run  # type: ignore
from neptune.internal.init.parameters import (  # type: ignore
    ASYNC_LAG_THRESHOLD,
    ASYNC_NO_PROGRESS_THRESHOLD,
    DEFAULT_FLUSH_PERIOD,
)
from neptune.metadata_containers.abstract import NeptuneObjectCallback  # type: ignore
from neptune.types.atoms.git_ref import GitRef, GitRefDisabled  # type: ignore

from refiners.training_utils.callback import Callback, CallbackConfig
from refiners.training_utils.config import BaseConfig
from refiners.training_utils.trainer import Trainer, register_callback

AnyTrainer = Trainer[BaseConfig, Any]


class NeptuneConfig(CallbackConfig):
    """Neptune.ai run configuration

    See https://docs.neptune.ai/api/neptune#init_run
    and https://github.com/neptune-ai/neptune-client/blob/1cd8452045e8524318f59216d151d73328f85bd1/src/neptune/objects/run.py#L131
    """

    project: str | None = None
    api_token: str | None = None
    with_id: str | None = None
    custom_run_id: str | None = None
    mode: Literal["async", "sync", "offline", "read-only", "debug"] | None = None
    name: str | None = None
    description: str | None = None
    tags: str | list[str] | None = None
    source_files: str | list[str] | None = None
    capture_stdout: bool | None = None
    capture_stderr: bool | None = None
    capture_hardware_metrics: bool | None = None
    fail_on_exception: bool = True
    monitoring_namespace: str | None = None
    flush_period: float = DEFAULT_FLUSH_PERIOD
    proxies: dict[str, str] | None = None
    capture_traceback: bool = True
    git_ref: GitRef | GitRefDisabled | None = None
    dependencies: Path | str | None = None
    async_lag_callback: NeptuneObjectCallback | None = None
    async_no_progress_callback: NeptuneObjectCallback | None = None
    async_lag_threshold: float = ASYNC_LAG_THRESHOLD
    async_no_progress_threshold: float = ASYNC_NO_PROGRESS_THRESHOLD


class NeptuneCallback(Callback[AnyTrainer]):
    """Neptune.ai callback for logging metrics"""

    run: Run

    def __init__(self, config: NeptuneConfig) -> None:
        """Initialize Neptune.ai callback

        Args:
            config: Neptune.ai run configuration
        """
        self.config = config
        self.epoch_losses: list[float] = []
        self.iteration_losses: list[float] = []

    def on_train_begin(self, trainer: AnyTrainer) -> None:
        # initialize Neptune `Run` (see https://docs.neptune.ai/api/run/)
        self.run = init_run(**self.config.model_dump())
        self.run["config"] = trainer.config.model_dump()

        # reset epoch and iteration losses
        self.epoch_losses = []
        self.iteration_losses = []

    def on_compute_loss_end(self, trainer: AnyTrainer) -> None:
        loss_value = trainer.loss.detach().cpu().item()
        self.epoch_losses.append(loss_value)
        self.iteration_losses.append(loss_value)
        self.run["train/step_loss"].append(loss_value, step=trainer.clock.step)  # type: ignore

    def on_optimizer_step_end(self, trainer: AnyTrainer) -> None:
        self.run["train/total_grad_norm"].append(trainer.grad_norm, step=trainer.clock.step)  # type: ignore
        avg_iteration_loss = sum(self.iteration_losses) / len(self.iteration_losses)
        self.run["train/average_iteration_loss"].append(avg_iteration_loss, step=trainer.clock.step)  # type: ignore
        self.iteration_losses = []

    def on_epoch_end(self, trainer: AnyTrainer) -> None:
        avg_epoch_loss = sum(self.epoch_losses) / len(self.epoch_losses)
        self.run["train/average_epoch_loss"].append(avg_epoch_loss, step=trainer.clock.step)  # type: ignore
        self.run["train/epoch"].append(trainer.clock.epoch, step=trainer.clock.step)  # type: ignore
        self.epoch_losses = []

    def on_lr_scheduler_step_end(self, trainer: AnyTrainer) -> None:
        self.run["train/learning_rate"].append(trainer.optimizer.param_groups[0]["lr"], step=trainer.clock.step)  # type: ignore

    def on_train_end(self, trainer: AnyTrainer) -> None:
        self.run.stop()


class NeptuneMixin(ABC):
    @register_callback()
    def neptune(self, config: NeptuneConfig) -> NeptuneCallback:
        return NeptuneCallback(config)
