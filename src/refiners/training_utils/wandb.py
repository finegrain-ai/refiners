from abc import ABC
from pathlib import Path
from typing import Any, Literal

import wandb
from PIL import Image

from refiners.training_utils.callback import Callback, CallbackConfig
from refiners.training_utils.config import BaseConfig
from refiners.training_utils.trainer import Trainer, register_callback

number = float | int
WandbLoggable = number | Image.Image | list[number] | dict[str, list[number]]


def convert_to_wandb(value: WandbLoggable) -> Any:
    match value:
        case Image.Image():
            return convert_to_wandb_image(value=value)
        case list():
            return convert_to_wandb_histogram(value=value)
        case dict():
            return convert_to_wandb_table(value=value)
        case _:
            return value


def convert_to_wandb_image(value: Image.Image) -> wandb.Image:
    return wandb.Image(data_or_path=value)


def convert_to_wandb_histogram(value: list[number]) -> wandb.Histogram:
    return wandb.Histogram(sequence=value)


def convert_to_wandb_table(value: dict[str, list[number]]) -> wandb.Table:
    assert all(
        isinstance(v, list) and len(v) == len(next(iter(value.values()))) for v in value.values()
    ), "Expected a dictionary of lists of the same size"
    columns = list(value.keys())
    data_rows = list(zip(*value.values()))
    return wandb.Table(columns=columns, data=data_rows)


class WandbLogger:
    def __init__(self, init_config: dict[str, Any] = {}) -> None:
        self.wandb_run = wandb.init(**init_config)  # type: ignore

    def log(self, data: dict[str, WandbLoggable], step: int) -> None:
        converted_data = {key: convert_to_wandb(value=value) for key, value in data.items()}
        self.wandb_run.log(converted_data, step=step)  # type: ignore

    def update_summary(self, key: str, value: Any) -> None:
        self.wandb_run.summary[key] = value  # type: ignore

    @property
    def project_name(self) -> str:
        return self.wandb_run.project_name()  # type: ignore

    @property
    def run_name(self) -> str:
        return self.wandb_run.name or ""  # type: ignore


class WandbConfig(CallbackConfig):
    """
    Wandb configuration.

    See https://docs.wandb.ai/ref/python/init for more details.
    """

    mode: Literal["online", "offline", "disabled"] = "disabled"
    project: str
    entity: str | None = None
    save_code: bool | None = None
    name: str | None = None
    tags: list[str] = []
    group: str | None = None
    job_type: str | None = None
    notes: str | None = None
    dir: Path | None = None
    resume: bool | Literal["allow", "must", "never", "auto"] | None = None
    reinit: bool | None = None
    magic: bool | None = None
    anonymous: Literal["never", "allow", "must"] | None = None
    id: str | None = None


AnyTrainer = Trainer[BaseConfig, Any]


class WandbCallback(Callback["TrainerWithWandb"]):
    def __init__(self, config: WandbConfig, /, trainer_config: dict[str, Any]) -> None:
        self.config = config
        self.epoch_losses: list[float] = []
        self.iteration_losses: list[float] = []
        self.logger = WandbLogger({**config.model_dump(), "config": trainer_config})

    def on_train_begin(self, trainer: "TrainerWithWandb") -> None:
        self.epoch_losses = []
        self.iteration_losses = []

    def on_compute_loss_end(self, trainer: "TrainerWithWandb") -> None:
        loss_value = trainer.loss.detach().cpu().item()
        self.epoch_losses.append(loss_value)
        self.iteration_losses.append(loss_value)
        if trainer.clock.is_evaluation_step:
            trainer.wandb_log(data={"step_loss": loss_value})

    def on_optimizer_step_end(self, trainer: "TrainerWithWandb") -> None:
        if trainer.clock.is_evaluation_step:
            avg_iteration_loss = sum(self.iteration_losses) / len(self.iteration_losses)
            trainer.wandb_log(data={"average_iteration_loss": avg_iteration_loss})
        self.iteration_losses = []
    def on_batch_end(self, trainer: "TrainerWithWandb") -> None:
        if trainer.clock.is_evaluation_step:
            batch_time, forward_time, backprop_time, data_time = (
                trainer.batch_time_m.avg,
                trainer.forward_time_m.avg,
                trainer.backprop_time_m.avg,
                trainer.data_time_m.avg,
            )
            batch_time_curr, forward_time_curr, backprop_time_curr, data_time_curr = (
                trainer.batch_time_m.val,
                trainer.forward_time_m.val,
                trainer.backprop_time_m.val,
                trainer.data_time_m.val,
            )
            effective_batch_size = trainer.clock.batch_size*trainer.clock.num_step_per_iteration
            trainer.wandb_log(
                data={
                    "batch_time": batch_time / effective_batch_size,
                    "forward_time": forward_time / effective_batch_size,
                    "backprop_time": backprop_time / effective_batch_size,
                    "data_time": data_time / effective_batch_size,
                    "batch_time_current": batch_time_curr / effective_batch_size,
                    "forward_time_current": forward_time_curr / effective_batch_size,
                    "backprop_time_current": backprop_time_curr / effective_batch_size,
                    "data_time_current": data_time_curr / effective_batch_size,
                }
            )
    def on_epoch_end(self, trainer: "TrainerWithWandb") -> None:
        if trainer.clock.is_evaluation_step:
            avg_epoch_loss = sum(self.epoch_losses) / len(self.epoch_losses)
            trainer.wandb_log(data={"average_epoch_loss": avg_epoch_loss, "epoch": trainer.clock.epoch})
            self.epoch_losses = []

    def on_lr_scheduler_step_end(self, trainer: "TrainerWithWandb") -> None:
        if trainer.clock.is_evaluation_step:
            trainer.wandb_log(data={"learning_rate": trainer.optimizer.param_groups[0]["lr"]})

    def on_backward_end(self, trainer: "TrainerWithWandb") -> None:
        if trainer.clock.is_evaluation_step:
            trainer.wandb_log(data={"total_grad_norm": trainer.total_gradient_norm})


class WandbMixin(ABC):
    config: Any
    wandb_logger: WandbLogger

    @register_callback()
    def wandb(self, config: WandbConfig) -> WandbCallback:
        return WandbCallback(config, trainer_config=self.config.model_dump())

    def wandb_log(self, data: dict[str, WandbLoggable]) -> None:
        assert isinstance(self, Trainer), "WandbMixin must be mixed with a Trainer"
        self.wandb.logger.log(data=data, step=self.clock.step)


class TrainerWithWandb(AnyTrainer, WandbMixin, ABC):
    pass
