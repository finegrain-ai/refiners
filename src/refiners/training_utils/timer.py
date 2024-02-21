import time
from typing import TYPE_CHECKING, Any

from refiners.training_utils.callback import Callback
from refiners.training_utils.config import BaseConfig
from refiners.training_utils.trainer import Trainer

if TYPE_CHECKING:
    from refiners.training_utils.config import BaseConfig
    from refiners.training_utils.trainer import Trainer


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


class TrainingTimer(Callback["Trainer[BaseConfig, Any]"]):
    def __init__(
        self,
    ) -> None:
        self.start_time: float = 0
        self.batch_time_meter = AverageMeter()
        self.forward_time_meter = AverageMeter()
        self.backprop_time_meter = AverageMeter()
        self.data_time_meter = AverageMeter()

    def on_epoch_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.start_time = time.time()

    def on_batch_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.data_time_meter.update(time.time() - self.start_time)

    def on_compute_loss_begin(self, trainer: Trainer[BaseConfig, Any]) -> None:
        self.start_time = time.time()

    def on_compute_loss_end(self, trainer: Trainer[BaseConfig, Any]) -> None:
        self.forward_time_meter.update(time.time() - self.start_time)

    def on_backward_begin(self, trainer: Trainer[BaseConfig, Any]) -> None:
        self.start_time = time.time()

    def on_backward_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        if (not trainer.clock.is_optimizer_step) and (not trainer.clock.is_lr_scheduler_step):
            self.backprop_time_meter.update(time.time() - self.start_time)

    def on_optimizer_step_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        if not trainer.clock.is_lr_scheduler_step:
            self.backprop_time_meter.update(time.time() - self.start_time)

    def on_lr_scheduler_step_end(self, trainer: Trainer[BaseConfig, Any]) -> None:
        self.backprop_time_meter.update(time.time() - self.start_time)

    def on_batch_end(self, trainer: Trainer[BaseConfig, Any]) -> None:
        data_time = self.data_time_meter.val
        forward_time = self.forward_time_meter.val
        backprop_time = self.backprop_time_meter.val
        self.batch_time_meter.update(data_time + forward_time + backprop_time)
        self.start_time = time.time()
