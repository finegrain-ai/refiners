from functools import cached_property
from typing import Generic, TypeVar

import torch
from jaxtyping import Float
from torch import Tensor

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.adapter import Adapter
from refiners.foundationals.latent_diffusion import SD1UNet, SDXLUNet

T = TypeVar("T", bound="SD1UNet | SDXLUNet")


class ExtractReferenceFeatures(fl.Module):
    """Extract the reference features from the input features.

    Note:
        This layer expects the input features to be a concatenation of conditional and unconditional features,
        as done when using Classifier-free guidance (CFG).

    The reference features are the first features of the conditional and unconditional input features.
    They are extracted, and repeated to match the batch size of the input features.

    Receives:
        features (Float[Tensor, "cfg_batch_size sequence_length embedding_dim"]): The input features.

    Returns:
        reference (Float[Tensor, "cfg_batch_size sequence_length embedding_dim"]): The reference features.
    """

    def forward(
        self,
        features: Float[Tensor, "cfg_batch_size sequence_length embedding_dim"],
    ) -> Float[Tensor, "cfg_batch_size sequence_length embedding_dim"]:
        cfg_batch_size = features.shape[0]
        batch_size = cfg_batch_size // 2

        # split the cfg
        features_cond, features_uncond = torch.chunk(features, 2, dim=0)
        # -> 2 x (batch_size, sequence_length, embedding_dim)

        # extract the reference features
        features_ref = torch.stack(
            (
                features_cond[0],  # (sequence_length, embedding_dim)
                features_uncond[0],  # (sequence_length, embedding_dim)
            ),
        )  # -> (2, sequence_length, embedding_dim)

        # repeat the reference features to match the batch size
        features_ref = features_ref.repeat_interleave(batch_size, dim=0)
        # -> (cfg_batch_size, sequence_length, embedding_dim)

        return features_ref


class AdaIN(fl.Module):
    """Apply Adaptive Instance Normalization (AdaIN) to the target features.

    See [[arXiv:1703.06868] Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868) for more details.

    Receives:
        reference (Float[Tensor, "cfg_batch_size sequence_length embedding_dim"]): The reference features.
        targets (Float[Tensor, "cfg_batch_size sequence_length embedding_dim"]): The target features.

    Returns:
        reference (Float[Tensor, "cfg_batch_size sequence_length embedding_dim"]): The reference features (unchanged).
        targets (Float[Tensor, "cfg_batch_size sequence_length embedding_dim"]): The target features, renormalized.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        """Initialize the AdaIN module.

        Args:
            epsilon: A small value to avoid division by zero.
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        targets: Float[Tensor, "cfg_batch_size sequence_length embedding_dim"],
        reference: Float[Tensor, "cfg_batch_size sequence_length embedding_dim"],
    ) -> tuple[
        Float[Tensor, "cfg_batch_size sequence_length embedding_dim"],  # targets (renormalized)
        Float[Tensor, "cfg_batch_size sequence_length embedding_dim"],  # reference (unchanged)
    ]:
        targets_mean = torch.mean(targets, dim=-2, keepdim=True)
        targets_std = torch.std(targets, dim=-2, keepdim=True)
        targets_normalized = (targets - targets_mean) / (targets_std + self.epsilon)

        reference_mean = torch.mean(reference, dim=-2, keepdim=True)
        reference_std = torch.std(reference, dim=-2, keepdim=True)
        targets_renormalized = targets_normalized * reference_std + reference_mean

        return (
            targets_renormalized,
            reference,
        )


class ScaleReferenceFeatures(fl.Module):
    """Scale the reference features.

    Note:
        This layer expects the input features to be a concatenation of conditional and unconditional features,
        as done when using Classifier-free guidance (CFG).

    This layer scales the reference features which will later be used (in the attention dot product) with the target features.

    Receives:
        features (Float[Tensor, "cfg_batch_size sequence_length embedding_dim"]): The input reference features.

    Returns:
        features (Float[Tensor, "cfg_batch_size sequence_length embedding_dim"]): The rescaled reference features.
    """

    def __init__(
        self,
        scale: float = 1.0,
    ) -> None:
        """Initialize the ScaleReferenceFeatures module.

        Args:
            scale: The scaling factor.
        """
        super().__init__()
        self.scale = scale

    def forward(
        self,
        features: Float[Tensor, "cfg_batch_size sequence_length embedding_dim"],
    ) -> Float[Tensor, "cfg_batch_size sequence_length embedding_dim"]:
        cfg_batch_size = features.shape[0]
        batch_size = cfg_batch_size // 2

        # clone the features
        # needed because all the following operations are in-place
        features = features.clone()

        # "stack" the cfg
        features_cfg_stack = features.reshape(2, batch_size, *features.shape[1:])

        # scale the reference features which will later be used (in the attention dot product) with the target features
        features_cfg_stack[:, 1:] *= self.scale

        # "unstack" the cfg
        features = features_cfg_stack.reshape(features.shape)

        return features


class StyleAligned(fl.Chain):
    """StyleAligned module.

    This layer encapsulates the logic of the StyleAligned method,
    as described in [[arXiv:2312.02133] Style Aligned Image Generation via Shared Attention](https://arxiv.org/abs/2312.02133).

    See also <https://blog.finegrain.ai/posts/implementing-style-aligned/>.

    Receives:
        features (Float[Tensor, "cfg_batch_size sequence_length_in embedding_dim"]): The input features.

    Returns:
        shared_features (Float[Tensor, "cfg_batch_size sequence_length_out embedding_dim"]): The transformed features.
    """

    def __init__(
        self,
        adain: bool,
        concatenate: bool,
        scale: float = 1.0,
    ) -> None:
        """Initialize the StyleAligned module.

        Args:
            adain: Whether to apply Adaptive Instance Normalization to the target features.
            scale: The scaling factor for the reference features.
            concatenate: Whether to concatenate the reference and target features.
        """
        super().__init__(
            # (features): (cfg_batch_size sequence_length embedding_dim)
            fl.Parallel(
                fl.Identity(),
                ExtractReferenceFeatures(),
            ),
            # (targets, reference)
            AdaIN(),
            # (targets_renormalized, reference)
            fl.Distribute(
                fl.Identity(),
                ScaleReferenceFeatures(scale=scale),
            ),
            # (targets_renormalized, reference_scaled)
            fl.Concatenate(
                fl.GetArg(index=0),  # targets
                fl.GetArg(index=1),  # reference
                dim=-2,  # sequence_length
            ),
            # (features_with_shared_reference)
        )

        if not adain:
            adain_module = self.ensure_find(AdaIN)
            self.remove(adain_module)

        if not concatenate:
            concatenate_module = self.ensure_find(fl.Concatenate)
            self.replace(
                old_module=concatenate_module,
                new_module=fl.GetArg(index=0),  # targets
            )

    @property
    def scale(self) -> float:
        """The scaling factor for the reference features."""
        scale_reference = self.ensure_find(ScaleReferenceFeatures)
        return scale_reference.scale

    @scale.setter
    def scale(self, scale: float) -> None:
        scale_reference = self.ensure_find(ScaleReferenceFeatures)
        scale_reference.scale = scale


class SharedSelfAttentionAdapter(fl.Chain, Adapter[fl.SelfAttention]):
    """Upgrades a `SelfAttention` layer into a `SharedSelfAttention` layer.

    This adapter inserts 3 `StyleAligned` modules right after
    the original Q, K, V `Linear`-s (wrapped inside a `fl.Distribute`).
    """

    def __init__(
        self,
        target: fl.SelfAttention,
        scale: float = 1.0,
    ) -> None:
        with self.setup_adapter(target):
            super().__init__(target)

        self._style_aligned_layers = [
            StyleAligned(  # Query
                adain=True,
                concatenate=False,
                scale=scale,
            ),
            StyleAligned(  # Key
                adain=True,
                concatenate=True,
                scale=scale,
            ),
            StyleAligned(  # Value
                adain=False,
                concatenate=True,
                scale=scale,
            ),
        ]

    @cached_property
    def style_aligned_layers(self) -> fl.Distribute:
        return fl.Distribute(*self._style_aligned_layers)

    def inject(self, parent: fl.Chain | None = None) -> "SharedSelfAttentionAdapter":
        self.target.insert_before_type(
            module_type=fl.ScaledDotProductAttention,
            new_module=self.style_aligned_layers,
        )
        return super().inject(parent)

    def eject(self) -> None:
        self.target.remove(self.style_aligned_layers)
        super().eject()

    @property
    def scale(self) -> float:
        return self.style_aligned_layers.layer(0, StyleAligned).scale

    @scale.setter
    def scale(self, scale: float) -> None:
        for style_aligned_module in self.style_aligned_layers:
            style_aligned_module.scale = scale


class StyleAlignedAdapter(Generic[T], fl.Chain, Adapter[T]):
    """Upgrade each `SelfAttention` layer of a UNet into a `SharedSelfAttention` layer."""

    def __init__(
        self,
        target: T,
        scale: float = 1.0,
    ) -> None:
        """Initialize the StyleAlignedAdapter.

        Args:
            target: The target module.
            scale: The scaling factor for the reference features.
        """
        with self.setup_adapter(target):
            super().__init__(target)

        # create a SharedSelfAttentionAdapter for each SelfAttention module
        self.shared_self_attention_adapters = tuple(
            SharedSelfAttentionAdapter(
                target=self_attention,
                scale=scale,
            )
            for self_attention in self.target.layers(fl.SelfAttention)
        )

    def inject(self, parent: fl.Chain | None = None) -> "StyleAlignedAdapter[T]":
        for shared_self_attention_adapter in self.shared_self_attention_adapters:
            shared_self_attention_adapter.inject()
        return super().inject(parent)

    def eject(self) -> None:
        for shared_self_attention_adapter in self.shared_self_attention_adapters:
            shared_self_attention_adapter.eject()
        super().eject()

    @property
    def scale(self) -> float:
        """The scaling factor for the reference features."""
        return self.shared_self_attention_adapters[0].scale

    @scale.setter
    def scale(self, scale: float) -> None:
        for shared_self_attention_adapter in self.shared_self_attention_adapters:
            shared_self_attention_adapter.scale = scale
