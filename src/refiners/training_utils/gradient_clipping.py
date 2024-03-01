from typing import TYPE_CHECKING, Any, Iterable

import torch
from torch import nn

from refiners.training_utils.callback import Callback, CallbackConfig

if TYPE_CHECKING:
    from refiners.training_utils.config import BaseConfig
    from refiners.training_utils.trainer import Trainer


def clip_gradient_norm(parameters: Iterable[nn.Parameter], total_norm: float, clip_norm: float = 1.0) -> None:
    """
    Clips the gradient norm of the parameters of a given model similar to `clip_grad_norm_`.
    """
    gradients = [p.grad.detach() for p in parameters if p.grad is not None]
    assert gradients, "The model has no gradients to clip."
    clip_coefficient = torch.tensor(data=clip_norm / (total_norm + 1e-6)).clamp(max=1)
    for gradient in gradients:
        gradient.mul_(other=clip_coefficient)  # type: ignore


def clip_gradient_value(parameters: Iterable[nn.Parameter], clip_value: float) -> None:
    """
    Clips the gradients of the parameters of a given model at an individual level similar to `clip_grad_value_`.
    """
    gradients = [p.grad.detach() for p in parameters if p.grad is not None]
    assert gradients, "The model has no gradients to clip."
    for gradient in gradients:
        gradient.clamp_(min=-clip_value, max=clip_value)


class GradientClippingConfig(CallbackConfig):
    clip_grad_norm: float | None = None
    clip_grad_value: float | None = None


class GradientClipping(Callback["Trainer[BaseConfig, Any]"]):
    def __init__(self, config: GradientClippingConfig) -> None:
        self.config = config

    def on_optimizer_step_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        clip_norm = self.config.clip_grad_norm
        if trainer.scaler is not None:
            trainer.scaler.unscale_(trainer.optimizer)
        if clip_norm is not None:
            clip_gradient_norm(
                parameters=trainer.learnable_parameters, total_norm=trainer.total_gradient_norm, clip_norm=clip_norm
            )

        clip_value = self.config.clip_grad_value
        if clip_value is not None:
            clip_gradient_value(parameters=trainer.learnable_parameters, clip_value=clip_value)
