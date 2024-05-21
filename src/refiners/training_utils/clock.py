import time
from typing import TYPE_CHECKING, Any

from refiners.training_utils.callback import Callback, CallbackConfig
from refiners.training_utils.common import Epoch, Iteration, Step, TimeValue

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
        training_duration: TimeValue,
        gradient_accumulation: Step,
        lr_scheduler_interval: TimeValue,
        verbose: bool = True,
    ) -> None:
        self.training_duration = training_duration
        self.gradient_accumulation = gradient_accumulation
        self.lr_scheduler_interval = lr_scheduler_interval
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.step = 0
        self.epoch = 0
        self.iteration = 0
        self.num_batches_processed = 0
        self.num_minibatches_processed = 0
        self.loss: Tensor | None = None

    def is_due(self, interval: TimeValue) -> bool:
        match interval:
            case Step(number):
                return self.step % number == 0
            case Iteration(number):
                return self.iteration % number == 0
            case Epoch(number):
                return self.epoch % number == 0
            case _:
                raise ValueError(f"Unsupported TimeValue: {interval}")

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
        return self.num_minibatches_processed == self.gradient_accumulation.number

    @property
    def done(self) -> bool:
        match self.training_duration:
            case Step(number):
                return self.step >= number
            case Iteration(number):
                return self.iteration >= number
            case Epoch(number):
                return self.epoch >= number
            case _:
                raise ValueError(f"Unsupported TimeValue: {self.training_duration}")

    def log(self, message: str, /) -> None:
        if self.verbose:
            logger.info(message)

    def on_train_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.log(f"Starting training for {self.training_duration}.")
        self.reset()
        self.start_timer()

    def on_train_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.stop_timer()
        self.log(
            (
                "Training took: "
                f"{self.time_elapsed} seconds, "
                f"{self.iteration} iterations, "
                f"{self.epoch} epochs, "
                f"{self.step} steps."
            )
        )

    def on_epoch_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.log(f"Epoch {self.epoch} started.")

    def on_epoch_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.log(f"Epoch {self.epoch} ended.")
        self.epoch += 1
        self.num_batches_processed = 0

    def on_step_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        if self.num_minibatches_processed == 0:
            if self.iteration > 0:
                self.log(f"Iteration {self.iteration - 1} ended.")
            self.log(f"Iteration {self.iteration} started.")
        self.log(f"Step {self.step} started.")

    def on_step_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.log(f"Step {self.step} ended.")
        self.step += 1

    def on_backward_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.num_minibatches_processed += 1
        self.num_batches_processed += 1

    def on_optimizer_step_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.iteration += 1
        self.num_minibatches_processed = 0
