from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from pydantic import BaseModel, ConfigDict, field_validator

from refiners.training_utils.common import (
    Epoch,
    Iteration,
    Step,
    TimeValue,
    TimeValueInput,
    parse_number_unit_field,
    scoped_seed,
)

if TYPE_CHECKING:
    from refiners.training_utils.trainer import Trainer

T = TypeVar("T", bound="Trainer[Any, Any]")


class StepEventConfig(BaseModel):
    """
    Base configuration for an event that is triggered at every step.

    - `seed`: Seed to use for the event. If `None`, the seed will not be set. The random state will be saved and
    restored after the event.
    - `interval`: Interval at which the event should be triggered. The interval is defined by either a `Step` object,
    an `Iteration` object, or an `Epoch` object.
    """

    model_config = ConfigDict(extra="forbid")
    seed: int | None = None
    interval: Step | Iteration | Epoch = Step(1)

    @field_validator("interval", mode="before")
    def parse_field(cls, value: TimeValueInput) -> TimeValue:
        return parse_number_unit_field(value)


class IterationEventConfig(BaseModel):
    """
    Base configuration for an event that is triggered only once per iteration.

    - `seed`: Seed to use for the event. If `None`, the seed will not be set. The random state will be saved and
    restored after the event.
    - `interval`: Interval at which the event should be triggered. The interval is defined by an `Iteration` object or
    a `Epoch` object.
    """

    model_config = ConfigDict(extra="forbid")
    seed: int | None = None
    interval: Iteration | Epoch = Iteration(1)

    @field_validator("interval", mode="before")
    def parse_field(cls, value: TimeValueInput) -> TimeValue:
        return parse_number_unit_field(value)


class EpochEventConfig(BaseModel):
    """
    Base configuration for an event that is triggered only once per epoch.

    - `seed`: Seed to use for the event. If `None`, the seed will not be set. The random state will be saved and
    restored after the event.
    - `interval`: Interval at which the event should be triggered. The interval is defined by a `Epoch` object.
    """

    model_config = ConfigDict(extra="forbid")
    seed: int | None = None
    interval: Epoch = Epoch(1)

    @field_validator("interval", mode="before")
    def parse_field(cls, value: TimeValueInput) -> TimeValue:
        return parse_number_unit_field(value)


EventConfig = StepEventConfig | IterationEventConfig | EpochEventConfig


class CallbackConfig(BaseModel):
    """
    Base configuration for a callback.

    For your callback to be properly configured, you should inherit from this class and add your own configuration.
    """

    model_config = ConfigDict(extra="forbid")
    on_epoch_begin: EpochEventConfig = EpochEventConfig()
    on_epoch_end: EpochEventConfig = EpochEventConfig()
    on_batch_begin: StepEventConfig = StepEventConfig()
    on_batch_end: StepEventConfig = StepEventConfig()
    on_backward_begin: StepEventConfig = StepEventConfig()
    on_backward_end: StepEventConfig = StepEventConfig()
    on_optimizer_step_begin: IterationEventConfig = IterationEventConfig()
    on_optimizer_step_end: IterationEventConfig = IterationEventConfig()
    on_compute_loss_begin: StepEventConfig = StepEventConfig()
    on_compute_loss_end: StepEventConfig = StepEventConfig()
    on_evaluate_begin: IterationEventConfig = IterationEventConfig()
    on_evaluate_end: IterationEventConfig = IterationEventConfig()
    on_lr_scheduler_step_begin: IterationEventConfig = IterationEventConfig()
    on_lr_scheduler_step_end: IterationEventConfig = IterationEventConfig()


class Callback(Generic[T]):
    def run_event(self, trainer: T, callback_name: str, event_name: str) -> None:
        if not hasattr(self, event_name):
            return
        callback_config = getattr(trainer.config, callback_name)
        # For event that run once, there is no configuration to check, e.g. on_train_begin
        if not hasattr(callback_config, event_name):
            getattr(self, event_name)(trainer)
            return
        event_config = cast(EventConfig, getattr(callback_config, event_name))
        if not trainer.clock.is_due(event_config.interval):
            return
        with scoped_seed(event_config.seed):
            getattr(self, event_name)(trainer)

    def on_init_begin(self, trainer: T) -> None: ...

    def on_init_end(self, trainer: T) -> None: ...

    def on_train_begin(self, trainer: T) -> None: ...

    def on_train_end(self, trainer: T) -> None: ...

    def on_epoch_begin(self, trainer: T) -> None: ...

    def on_epoch_end(self, trainer: T) -> None: ...

    def on_batch_begin(self, trainer: T) -> None: ...

    def on_batch_end(self, trainer: T) -> None: ...

    def on_backward_begin(self, trainer: T) -> None: ...

    def on_backward_end(self, trainer: T) -> None: ...

    def on_optimizer_step_begin(self, trainer: T) -> None: ...

    def on_optimizer_step_end(self, trainer: T) -> None: ...

    def on_compute_loss_begin(self, trainer: T) -> None: ...

    def on_compute_loss_end(self, trainer: T) -> None: ...

    def on_evaluate_begin(self, trainer: T) -> None: ...

    def on_evaluate_end(self, trainer: T) -> None: ...

    def on_lr_scheduler_step_begin(self, trainer: T) -> None: ...

    def on_lr_scheduler_step_end(self, trainer: T) -> None: ...
