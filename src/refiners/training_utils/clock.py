import time
from functools import cached_property
from typing import TYPE_CHECKING, Any

from refiners.training_utils.callback import Callback, CallbackConfig
from refiners.training_utils.common import Epoch, Iteration, Step, TimeUnit, TimeValue

if TYPE_CHECKING:
    from refiners.training_utils.config import BaseConfig
    from refiners.training_utils.trainer import Trainer


from loguru import logger
from torch import Tensor


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
        assert batch_size > 0, "Batch size must be greater than 0."
        assert (
            dataset_length >= batch_size
        ), f"Dataset length ({dataset_length}) must be greater than batch_size ({batch_size})."
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

    @cached_property
    def unit_to_steps(self) -> dict[TimeUnit, int]:
        iteration_factor = self.num_batches_per_epoch if isinstance(self.gradient_accumulation, Epoch) else 1
        return {
            Step: 1,
            Epoch: self.num_batches_per_epoch,
            Iteration: self.gradient_accumulation.number * iteration_factor,
        }

    def convert_time_value_to_steps(self, time_value: TimeValue) -> int:
        return time_value.number * self.unit_to_steps[time_value.unit]

    def convert_steps_to_time_unit(self, steps: int, unit: TimeUnit) -> int:
        return steps // self.unit_to_steps[unit]

    def convert_time_value(self, time_value: TimeValue, target_unit: TimeUnit) -> int:
        steps = self.convert_time_value_to_steps(time_value=time_value)
        return self.convert_steps_to_time_unit(steps=steps, unit=target_unit)

    @cached_property
    def num_epochs(self) -> int:
        return self.convert_time_value(time_value=self.training_duration, target_unit=Epoch)

    @cached_property
    def num_iterations(self) -> int:
        return self.convert_time_value(time_value=self.training_duration, target_unit=Iteration)

    @cached_property
    def num_steps(self) -> int:
        return self.convert_time_value(time_value=self.training_duration, target_unit=Step)

    @cached_property
    def num_step_per_iteration(self) -> int:
        return self.convert_time_value_to_steps(self.gradient_accumulation)

    @cached_property
    def num_step_per_evaluation(self) -> int:
        return self.convert_time_value_to_steps(self.evaluation_interval)

    def is_due(self, interval: TimeValue) -> bool:
        return self.step % self.convert_time_value_to_steps(interval) == 0

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

    @property
    def is_optimizer_step(self) -> bool:
        return self.num_minibatches_processed == self.num_step_per_iteration

    @property
    def done(self) -> bool:
        return self.step >= self.num_steps

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

    def on_epoch_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.log(f"Epoch {trainer.clock.epoch} ended.")
        trainer.clock.epoch += 1
        trainer.clock.num_batches_processed = 0

    def on_step_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        if self.num_minibatches_processed == 0:
            self.log(f"Iteration {trainer.clock.iteration} started.")
        self.log(f"Step {trainer.clock.step} started.")

    def on_backward_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.log(f"Step {trainer.clock.step} ended.")
        trainer.clock.step += 1
        trainer.clock.num_batches_processed += 1
        trainer.clock.num_minibatches_processed += 1

    def on_optimizer_step_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.log(f"Iteration {trainer.clock.iteration} ended.")
        trainer.clock.iteration += 1
        trainer.clock.num_minibatches_processed = 0

    def on_evaluate_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.log("Evaluation started.")

    def on_evaluate_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.log("Evaluation ended.")
