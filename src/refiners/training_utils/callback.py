from typing import TYPE_CHECKING, Any, Generic, Iterable, TypeVar

from loguru import logger
from torch import tensor
from torch.nn import Parameter
import time

if TYPE_CHECKING:
    from refiners.training_utils.config import BaseConfig
    from refiners.training_utils.trainers.trainer import Trainer

__all__ = [
    "Callback",
    "GradientNormClipping",
    "GradientValueClipping",
    "ClockCallback",
    "GradientNormLogging",
    "MonitorLoss",
]


def clip_gradient_norm(parameters: Iterable[Parameter], total_norm: float, clip_norm: float = 1.0) -> None:
    """
    Clips the gradient norm of the parameters of a given model similar to `clip_grad_norm_`.
    """
    gradients = [p.grad.detach() for p in parameters if p.grad is not None]
    assert gradients, "The model has no gradients to clip."
    clip_coefficient = tensor(data=clip_norm / (total_norm + 1e-6)).clamp(max=1)
    for gradient in gradients:
        gradient.mul_(other=clip_coefficient)  # type: ignore


def clip_gradient_value(parameters: Iterable[Parameter], clip_value: float) -> None:
    """
    Clips the gradients of the parameters of a given model at an individual level similar to `clip_grad_value_`.
    """
    gradients = [p.grad.detach() for p in parameters if p.grad is not None]
    assert gradients, "The model has no gradients to clip."
    for gradient in gradients:
        gradient.clamp_(min=-clip_value, max=clip_value)


T = TypeVar("T")


class Callback(Generic[T]):
    def on_init_begin(self, trainer: T) -> None:
        ...

    def on_init_end(self, trainer: T) -> None:
        ...

    def on_train_begin(self, trainer: T) -> None:
        ...

    def on_train_end(self, trainer: T) -> None:
        ...

    def on_epoch_begin(self, trainer: T) -> None:
        ...

    def on_epoch_end(self, trainer: T) -> None:
        ...

    def on_batch_begin(self, trainer: T) -> None:
        ...

    def on_batch_end(self, trainer: T) -> None:
        ...

    def on_backward_begin(self, trainer: T) -> None:
        ...

    def on_backward_end(self, trainer: T) -> None:
        ...

    def on_optimizer_step_begin(self, trainer: T) -> None:
        ...

    def on_optimizer_step_end(self, trainer: T) -> None:
        ...

    def on_compute_loss_begin(self, trainer: T) -> None:
        ...

    def on_compute_loss_end(self, trainer: T) -> None:
        ...

    def on_evaluate_begin(self, trainer: T) -> None:
        ...

    def on_evaluate_end(self, trainer: T) -> None:
        ...

    def on_lr_scheduler_step_begin(self, trainer: T) -> None:
        ...

    def on_lr_scheduler_step_end(self, trainer: T) -> None:
        ...

    def on_checkpoint_save(self, trainer: T) -> None:
        ...


class ClockCallback(Callback["Trainer[BaseConfig, Any]"]):
    def on_train_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        trainer.clock.reset()
        logger.info(
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
        logger.info(
            (
                "Training took: "
                f"{trainer.clock.time_elapsed} seconds, "
                f"{trainer.clock.iteration} iterations, "
                f"{trainer.clock.epoch} epochs, "
                f"{trainer.clock.step} steps."
            )
        )

    def on_epoch_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        logger.info(f"Epoch {trainer.clock.epoch} started.")

    def on_epoch_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        trainer.clock.epoch += 1
        trainer.clock.num_batches_processed = 0

    def on_batch_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        logger.info(f"Step {trainer.clock.step} started.")
        
    def on_backward_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        trainer.clock.step += 1
        trainer.clock.num_batches_processed += 1
        trainer.clock.num_minibatches_processed += 1

    def on_optimizer_step_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        logger.info(f"Iteration {trainer.clock.iteration} ended.")
        trainer.clock.iteration += 1
        trainer.clock.num_minibatches_processed = 0

    def on_evaluate_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        logger.info("Evaluation started.")

    def on_evaluate_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        logger.info("Evaluation ended.")


class MonitorLoss(Callback["Trainer[BaseConfig, Any]"]):
    def on_train_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.epoch_losses: list[float] = []
        self.iteration_losses: list[float] = []

    def on_compute_loss_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        loss_value = trainer.loss.detach().cpu().item()
        self.epoch_losses.append(loss_value)
        self.iteration_losses.append(loss_value)
        trainer.log(data={"step_loss": loss_value})

    def on_optimizer_step_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        avg_iteration_loss = sum(self.iteration_losses) / len(self.iteration_losses)
        trainer.log(data={"average_iteration_loss": avg_iteration_loss})
        self.iteration_losses = []

    def on_epoch_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        avg_epoch_loss = sum(self.epoch_losses) / len(self.epoch_losses)
        trainer.log(data={"average_epoch_loss": avg_epoch_loss, "epoch": trainer.clock.epoch})
        self.epoch_losses = []

    def on_lr_scheduler_step_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        trainer.log(data={"learning_rate": trainer.optimizer.param_groups[0]["lr"]})

# Ported from open-muse
class AverageTimeMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def update(self, val: float):
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
    
    def start(self):
        self.start_time = time.time()
    
    def cancel(self):
        self.start_time = None
    
    def stop(self):
        if self.start_time is None:
            return
        spent = time.time() - self.start_time
        self.update(spent)
        self.start_time = None


class MonitorTime(Callback["Trainer[BaseConfig, Any]"]):
    def __init__(self) -> None:
        self.forward = AverageTimeMeter()
        self.data = AverageTimeMeter()
        self.backward = AverageTimeMeter()
        
        super().__init__()
    
    def on_compute_loss_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.forward.start()
    
    def on_compute_loss_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.forward.stop()
    
    def on_backward_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.backward.start()
    
    def on_backward_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.backward.stop()
    
    def on_batch_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.data.stop()
    
    def on_batch_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        trainer.log(data={
            "timing/forward_time per item": self.forward.avg / trainer.clock.batch_size, 
            "timing/backward_time per item": self.backward.avg / trainer.clock.batch_size, 
            "timing/data_time per item": self.data.avg / trainer.clock.batch_size
        })
        self.data.start()
       
    def on_epoch_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.data.start()
        
    def on_epoch_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        self.data.cancel()


class GradientNormClipping(Callback["Trainer[BaseConfig, Any]"]):
    def on_backward_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        clip_norm = trainer.config.training.clip_grad_norm
        if clip_norm is not None:
            clip_gradient_norm(
                parameters=trainer.learnable_parameters, total_norm=trainer.total_gradient_norm, clip_norm=clip_norm
            )


class GradientValueClipping(Callback["Trainer[BaseConfig, Any]"]):
    def on_backward_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        clip_value = trainer.config.training.clip_grad_value
        if clip_value is not None:
            clip_gradient_value(parameters=trainer.learnable_parameters, clip_value=clip_value)


class GradientNormLogging(Callback["Trainer[BaseConfig, Any]"]):
    def on_backward_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        trainer.log(data={"total_grad_norm": trainer.total_gradient_norm})

from fnmatch import fnmatch
        
class GradientNormLayerLogging(Callback["Trainer[BaseConfig, Any]"]):
    def __init__(self, names: list[str]) -> None:
        self.names = names
        super().__init__()
    def on_backward_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        named_gradient_norm = trainer.named_gradient_norm
        for layer_name, norm in named_gradient_norm:
            for pattern in self.names:
                if fnmatch(layer_name, pattern):
                    trainer.log(data={f"layer_grad_norm/{layer_name}": norm})