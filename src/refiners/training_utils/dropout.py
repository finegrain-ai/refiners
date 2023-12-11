from typing import TYPE_CHECKING, Any, TypeVar

from torch import Tensor, cat, rand, randint
from torch.nn import Dropout as TorchDropout

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.adapter import Adapter
from refiners.training_utils.callback import Callback

if TYPE_CHECKING:
    from refiners.training_utils.config import BaseConfig
    from refiners.training_utils.trainer import Trainer


__all__ = ["Dropout", "GyroDropout", "DropoutCallback"]


class Dropout(TorchDropout, fl.Module):
    def __init__(self, probability: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p=probability, inplace=inplace)


class GyroDropout(fl.Module):
    """
    GyroDropout is a variant of dropout that maximizes the ensemble effect during neural network training.
    It pre-selects a fixed number of dropout masks and periodically selects a subset of them for training.
    This leads to increased robustness and diversity among the subnetworks, improving accuracy compared to conventional
    dropout.

    Parameters:
    -----------
    total_subnetworks:
        The total number of pre-selected subnetworks ('Sigma'). These subnetworks are dropout masks
        that are precomputed and stored.

    concurrent_subnetworks:
        The number of subnetworks to use concurrently in each forward pass ('Tau'). A random selection of
        masks from the precomputed set is used to dropout different portions of the input.

    dropout_probability: float, optional (default=0.5)
        The probability that an element will be zeroed by the dropout.

    iters_per_epoch:
        Number of iterations per epoch, used to determine how often the masks should be updated.

    num_features_threshold:
        If the number of features in the input is greater than this threshold, dropout is skipped. This is because
        gyro dropout mask size vram usage is proportional to the number of features in the input.
    """

    def __init__(
        self,
        total_subnetworks: int,
        concurrent_subnetworks: int,
        dropout_probability: float = 0.5,
        iters_per_epoch: int = 1,
        num_features_threshold: float = 5e5,
    ) -> None:
        super().__init__()
        assert (
            iters_per_epoch >= total_subnetworks
        ), "The number of iterations per epoch must be greater than the number of masks"
        self.dropout_probability = dropout_probability
        self.iters_per_epoch = iters_per_epoch
        self.total_subnetworks = total_subnetworks
        self.concurrent_subnetworks = concurrent_subnetworks
        self.scale = 1 / (1 - self.dropout_probability)
        self.mask_update_interval = int(self.iters_per_epoch / self.total_subnetworks) * self.concurrent_subnetworks
        self.preselected_masks: Tensor | None = None
        self.dropout_mask = None
        self.training_step = 0
        self.num_features_threshold = num_features_threshold
        self.skip_high_num_features = False

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        if self.skip_high_num_features:
            return self.basic_dropout(x)
        if self.training_step == 0:
            num_features = x.shape[1] * x.shape[2] if x.dim() == 3 else x.shape[1]
            if num_features > self.num_features_threshold:
                self.skip_high_num_features = True
                self.basic_dropout = Dropout(probability=self.dropout_probability)
                return self.basic_dropout(x)
            self.init_masks(x=x)

        if self.training_step % self.mask_update_interval == 0:
            self.update_dropout_mask(x=x)

        self.training_step += 1

        return x * self.dropout_mask * self.scale

    def init_masks(self, x: Tensor) -> None:
        if x.dim() == 2:
            self.preselected_masks = (
                rand(self.total_subnetworks, x.shape[1], device=x.device) > self.dropout_probability
            )
        if x.dim() == 3:
            self.preselected_masks = (
                rand(self.total_subnetworks, x.shape[1], x.shape[2], device=x.device) > self.dropout_probability
            )

        assert self.preselected_masks is not None, "The input tensor must have 2 or 3 dimensions"
        self.preselected_masks = self.preselected_masks.float()

    def update_dropout_mask(self, x: Tensor) -> None:
        assert self.preselected_masks is not None
        indices = randint(low=0, high=self.total_subnetworks, size=(self.concurrent_subnetworks,), device=x.device)
        selected_masks = self.preselected_masks[indices]

        repeat_factor = x.shape[0] // self.concurrent_subnetworks
        remaining = x.shape[0] % self.concurrent_subnetworks
        repeated_masks = [selected_masks] * repeat_factor
        if remaining > 0:
            repeated_masks.append(selected_masks[:remaining])
        final_masks = cat(tensors=repeated_masks, dim=0)

        if x.dim() == 2:
            self.dropout_mask = final_masks
        if x.dim() == 3:
            self.dropout_mask = final_masks.expand_as(x)


class DropoutAdapter(fl.Chain, Adapter[fl.Linear]):
    def __init__(self, target: fl.Linear, probability: float = 0.5):
        with self.setup_adapter(target):
            super().__init__(target, Dropout(probability=probability))


class GyroDropoutAdapter(fl.Chain, Adapter[fl.Linear]):
    def __init__(
        self,
        target: fl.Linear,
        probability: float = 0.5,
        total_subnetworks: int = 512,
        concurrent_subnetworks: int = 64,
        iters_per_epoch: int = 512,
        num_features_threshold: float = 5e5,
    ) -> None:
        self.probability = probability
        self.total_subnetworks = total_subnetworks
        self.concurrent_subnetworks = concurrent_subnetworks
        self.iters_per_epoch = iters_per_epoch

        with self.setup_adapter(target):
            super().__init__(
                target,
                GyroDropout(
                    total_subnetworks=total_subnetworks,
                    concurrent_subnetworks=concurrent_subnetworks,
                    dropout_probability=probability,
                    iters_per_epoch=iters_per_epoch,
                    num_features_threshold=num_features_threshold,
                ),
            )


def apply_dropout(module: fl.Chain, probability: float = 0.5) -> None:
    for linear, parent in module.walk(fl.Linear):
        if not linear.weight.requires_grad:
            continue
        assert not (
            isinstance(parent, Dropout) or isinstance(parent, GyroDropout)
        ), f"{linear} already has a dropout layer"
        DropoutAdapter(target=linear, probability=probability).inject(parent)


def apply_gyro_dropout(
    module: fl.Chain,
    probability: float = 0.5,
    total_subnetworks: int = 32,
    concurrent_subnetworks: int = 16,
    iters_per_epoch: int = 32,
) -> None:
    for linear, parent in module.walk(fl.Linear):
        if not linear.weight.requires_grad:
            continue
        assert not (
            isinstance(parent, Dropout) or isinstance(parent, GyroDropout)
        ), f"{linear} already has a dropout layer"
        GyroDropoutAdapter(
            target=linear,
            probability=probability,
            total_subnetworks=total_subnetworks,
            concurrent_subnetworks=concurrent_subnetworks,
            iters_per_epoch=iters_per_epoch,
        ).inject(parent)


ConfigType = TypeVar("ConfigType", bound="BaseConfig")


class DropoutCallback(Callback["Trainer[ConfigType, Any]"]):
    def on_train_begin(self, trainer: "Trainer[ConfigType, Any]") -> None:
        dropout_config = trainer.config.dropout
        chain_models = [model for model in trainer.models.values() if isinstance(model, fl.Chain)]
        for model in chain_models:
            dropout_config.apply_dropout(model=model)
