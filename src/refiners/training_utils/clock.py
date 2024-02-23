import time
from functools import cached_property
from typing import TYPE_CHECKING, Any

from refiners.training_utils.callback import Callback, CallbackConfig
from refiners.training_utils.common import TimeUnit, TimeValue

if TYPE_CHECKING:
    from refiners.training_utils.config import BaseConfig
    from refiners.training_utils.trainer import Trainer


from loguru import logger
from torch import Tensor


# Ported from open-muse
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def update(self, val: float):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


class ClockConfig(CallbackConfig):
    verbose: bool = True


class TrainingClock(Callback["Trainer[BaseConfig, Any]"]):
    def __init__(
        self,
        dataset_length: int,
        batch_size: int,
        training_duration: TimeValue,
        gradient_accumulation: TimeValue,
        evaluation_interval: TimeValue,
        lr_scheduler_interval: TimeValue,
        verbose: bool = True,
    ) -> None:
        self.dataset_length = dataset_length
        self.batch_size = batch_size
        self.training_duration = training_duration
        self.gradient_accumulation = gradient_accumulation
        self.evaluation_interval = evaluation_interval
        self.lr_scheduler_interval = lr_scheduler_interval
        self.verbose = verbose
        self.num_batches_per_epoch = dataset_length // batch_size
        self.start_time = None
        self.end_time = None
        self.step = 0
        self.epoch = 0
        self.iteration = 0
        self.num_batches_processed = 0
        self.num_minibatches_processed = 0
        self.loss: Tensor | None = None
        self.meter_start_time: float = 0
        self.batch_time_meter = AverageMeter()
        self.forward_time_meter = AverageMeter()
        self.backprop_time_meter = AverageMeter()
        self.data_time_meter = AverageMeter()

    @cached_property
    def unit_to_steps(self) -> dict[TimeUnit, int]:
        iteration_factor = self.num_batches_per_epoch if self.gradient_accumulation["unit"] == TimeUnit.EPOCH else 1
        return {
            TimeUnit.STEP: 1,
            TimeUnit.EPOCH: self.num_batches_per_epoch,
            TimeUnit.ITERATION: self.gradient_accumulation["number"] * iteration_factor,
        }

    def convert_time_unit_to_steps(self, number: int, unit: TimeUnit) -> int:
        return number * self.unit_to_steps[unit]

    def convert_steps_to_time_unit(self, steps: int, unit: TimeUnit) -> int:
        return steps // self.unit_to_steps[unit]

    def convert_time_value(self, time_value: TimeValue, target_unit: TimeUnit) -> int:
        number, unit = time_value["number"], time_value["unit"]
        steps = self.convert_time_unit_to_steps(number=number, unit=unit)
        return self.convert_steps_to_time_unit(steps=steps, unit=target_unit)

    @cached_property
    def num_epochs(self) -> int:
        return self.convert_time_value(time_value=self.training_duration, target_unit=TimeUnit.EPOCH)

    @cached_property
    def num_iterations(self) -> int:
        return self.convert_time_value(time_value=self.training_duration, target_unit=TimeUnit.ITERATION)

    @cached_property
    def num_steps(self) -> int:
        return self.convert_time_value(time_value=self.training_duration, target_unit=TimeUnit.STEP)

    @cached_property
    def num_step_per_iteration(self) -> int:
        return self.convert_time_unit_to_steps(
            number=self.gradient_accumulation["number"], unit=self.gradient_accumulation["unit"]
        )

    @cached_property
    def num_step_per_evaluation(self) -> int:
        return self.convert_time_unit_to_steps(
            number=self.evaluation_interval["number"], unit=self.evaluation_interval["unit"]
        )

    def reset(self) -> None:
        self.start_time = None
        self.end_time = None
        self.step = 0
        self.epoch = 0
        self.iteration = 0
        self.num_batches_processed = 0
        self.num_minibatches_processed = 0

    def start_timer(self) -> None:
        self.start_time = time.time()

    def stop_timer(self) -> None:
        self.end_time = time.time()

    @property
    def time_elapsed(self) -> int:
        assert self.start_time is not None, "Timer has not been started yet."
        return int(time.time() - self.start_time)

    @cached_property
    def evaluation_interval_steps(self) -> int:
        return self.convert_time_unit_to_steps(
            number=self.evaluation_interval["number"], unit=self.evaluation_interval["unit"]
        )

    @cached_property
    def lr_scheduler_interval_steps(self) -> int:
        return self.convert_time_unit_to_steps(
            number=self.lr_scheduler_interval["number"], unit=self.lr_scheduler_interval["unit"]
        )

    @property
    def is_optimizer_step(self) -> bool:
        return self.num_minibatches_processed == self.num_step_per_iteration

    @property
    def is_lr_scheduler_step(self) -> bool:
        return self.step % self.lr_scheduler_interval_steps == 0

    @property
    def done(self) -> bool:
        return self.step >= self.num_steps

    @property
    def is_evaluation_step(self) -> bool:
        return self.step % self.evaluation_interval_steps == 0

    def log(self, message: str, /) -> None:
        if self.verbose:
            logger.info(message)

    def on_train_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        trainer.clock.reset()
        self.log(
            (
                "Starting training for a total of: "
                f"{trainer.clock.num_steps} steps, "
                f"{trainer.clock.num_epochs} epochs, "
                f"{trainer.clock.num_iterations} iterations."
            )
        )
        trainer.clock.start_timer()

    def on_train_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        trainer.clock.stop_timer()
        self.log(
            (
                "Training took: "
                f"{trainer.clock.time_elapsed} seconds, "
                f"{trainer.clock.iteration} iterations, "
                f"{trainer.clock.epoch} epochs, "
                f"{trainer.clock.step} steps."
            )
        )

    def on_epoch_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.log(f"Epoch {trainer.clock.epoch} started.")
        self.meter_start_time = time.time()

    def on_batch_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.log(f"Step {trainer.clock.step} started.")
        self.data_time_meter.update(time.time() - self.meter_start_time)

    def on_compute_loss_begin(self, trainer: Trainer[BaseConfig, Any]) -> None:
        self.meter_start_time = time.time()

    def on_compute_loss_end(self, trainer: Trainer[BaseConfig, Any]) -> None:
        self.forward_time_meter.update(time.time() - self.meter_start_time)

    def on_backward_begin(self, trainer: Trainer[BaseConfig, Any]) -> None:
        self.meter_start_time = time.time()

    def on_backward_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        trainer.clock.step += 1
        trainer.clock.num_batches_processed += 1
        trainer.clock.num_minibatches_processed += 1
        if (not trainer.clock.is_optimizer_step) and (not trainer.clock.is_lr_scheduler_step):
            self.backprop_time_meter.update(time.time() - self.meter_start_time)

    def on_optimizer_step_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.log(f"Iteration {trainer.clock.iteration} ended.")
        trainer.clock.iteration += 1
        trainer.clock.num_minibatches_processed = 0
        if not trainer.clock.is_lr_scheduler_step:
            self.backprop_time_meter.update(time.time() - self.meter_start_time)

    def on_lr_scheduler_step_end(self, trainer: Trainer[BaseConfig, Any]) -> None:
        self.backprop_time_meter.update(time.time() - self.meter_start_time)

    def on_batch_end(self, trainer: Trainer[BaseConfig, Any]) -> None:
        data_time = self.data_time_meter.val
        forward_time = self.forward_time_meter.val
        backprop_time = self.backprop_time_meter.val
        self.batch_time_meter.update(data_time + forward_time + backprop_time)
        self.meter_start_time = time.time()

    def on_evaluate_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.log("Evaluation started.")

    def on_evaluate_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.log("Evaluation ended.")

    def on_epoch_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        trainer.clock.epoch += 1
        trainer.clock.num_batches_processed = 0
