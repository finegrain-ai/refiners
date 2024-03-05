from typing import Any, Iterator, cast

from torch import Tensor

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.lora import Lora, LoraAdapter, auto_attach_loras
from refiners.foundationals.latent_diffusion.model import LatentDiffusionModel


class SDLoraManager:
    """Manage LoRAs for a Stable Diffusion model.

    Note:
        In the context of SDLoraManager, a "LoRA" is a set of ["LoRA layers"][refiners.fluxion.adapters.lora.Lora]
        that can be attached to a target model.
    """

    def __init__(
        self,
        target: LatentDiffusionModel,
    ) -> None:
        """Initialize the LoRA manager.

        Args:
            target: The target model to manage the LoRAs for.
        """
        self.target = target

    @property
    def unet(self) -> fl.Chain:
        """The Stable Diffusion's U-Net model."""
        unet = self.target.unet
        assert isinstance(unet, fl.Chain)
        return unet

    @property
    def clip_text_encoder(self) -> fl.Chain:
        """The Stable Diffusion's text encoder."""
        clip_text_encoder = self.target.clip_text_encoder
        assert isinstance(clip_text_encoder, fl.Chain)
        return clip_text_encoder

    def add_loras(
        self,
        name: str,
        /,
        tensors: dict[str, Tensor],
        scale: float = 1.0,
        unet_inclusions: list[str] | None = None,
        unet_exclusions: list[str] | None = None,
        unet_preprocess: dict[str, str] | None = None,
        text_encoder_inclusions: list[str] | None = None,
        text_encoder_exclusions: list[str] | None = None,
    ) -> None:
        """Load a single LoRA from a `state_dict`.

        Warning:
            This method expects the keys of the `state_dict` to be in the commonly found formats on CivitAI's hub.

        Args:
            name: The name of the LoRA.
            tensors: The `state_dict` of the LoRA to load.
            scale: The scale to use for the LoRA.

        Raises:
            AssertionError: If the Manager already has a LoRA with the same name.
        """
        assert name not in self.names, f"LoRA {name} already exists"

        # load LoRA the state_dict
        loras = Lora.from_dict(
            name,
            state_dict={
                key: value.to(
                    device=self.target.device,
                    dtype=self.target.dtype,
                )
                for key, value in tensors.items()
            },
        )
        # sort all the LoRA's keys using the `sort_keys` method
        loras = {key: loras[key] for key in sorted(loras.keys(), key=SDLoraManager.sort_keys)}

        # if no key contains "unet" or "text", assume all keys are for the unet
        if all("unet" not in key and "text" not in key for key in loras.keys()):
            loras = {f"unet_{key}": value for key, value in loras.items()}

        # attach the LoRA to the target
        self.add_loras_to_unet(loras, include=unet_inclusions, exclude=unet_exclusions, preprocess=unet_preprocess)
        self.add_loras_to_text_encoder(loras, include=text_encoder_inclusions, exclude=text_encoder_exclusions)

        # set the scale of the LoRA
        self.set_scale(name, scale)

    def add_loras_to_text_encoder(
        self,
        loras: dict[str, Lora[Any]],
        /,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        """Add multiple LoRAs to the text encoder.

        Args:
            loras: The dictionary of LoRAs to add to the text encoder.
                (keys are the names of the LoRAs, values are the LoRAs to add to the text encoder)
        """
        text_encoder_loras = {key: loras[key] for key in loras.keys() if "text" in key}
        auto_attach_loras(text_encoder_loras, self.clip_text_encoder, exclude=exclude, include=include)

    def add_loras_to_unet(
        self,
        loras: dict[str, Lora[Any]],
        /,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        preprocess: dict[str, str] | None = None,
        debug_map: list[tuple[str, str]] | None = None,
    ) -> None:
        """Add multiple LoRAs to the U-Net.

        Args:
            loras: The dictionary of LoRAs to add to the U-Net.
                (keys are the names of the LoRAs, values are the LoRAs to add to the U-Net)
        """
        unet_loras = {key: loras[key] for key in loras.keys() if "unet" in key}

        if exclude is None:
            exclude = ["TimestepEncoder"]

        if preprocess is None:
            preprocess = {
                "res": "ResidualBlock",
                "downsample": "Downsample",
                "upsample": "Upsample",
            }

        if include is not None:
            preprocess = {k: v for k, v in preprocess.items() if v in include}

        preprocess = {k: v for k, v in preprocess.items() if v not in exclude}

        loras_excluded = {k: v for k, v in unet_loras.items() if any(x in k for x in preprocess.keys())}
        loras_remaining = {k: v for k, v in unet_loras.items() if k not in loras_excluded}

        for exc, v in preprocess.items():
            ls = {k: v for k, v in loras_excluded.items() if exc in k}
            auto_attach_loras(ls, self.unet, include=[v], debug_map=debug_map)

        auto_attach_loras(loras_remaining, self.unet, exclude=exclude, include=include, debug_map=debug_map)

    def remove_loras(self, *names: str) -> None:
        """Remove multiple LoRAs from the target.

        Args:
            names: The names of the LoRAs to remove.
        """
        for lora_adapter in self.lora_adapters:
            for name in names:
                lora_adapter.remove_lora(name)

            if len(lora_adapter.loras) == 0:
                lora_adapter.eject()

    def remove_all(self) -> None:
        """Remove all the LoRAs from the target."""
        for lora_adapter in self.lora_adapters:
            lora_adapter.eject()

    def get_loras_by_name(self, name: str, /) -> list[Lora[Any]]:
        """Get the LoRA layers with the given name.

        Args:
            name: The name of the LoRA.
        """
        return [lora for lora in self.loras if lora.name == name]

    def get_scale(self, name: str, /) -> float:
        """Get the scale of the LoRA with the given name.

        Args:
            name: The name of the LoRA.

        Returns:
            The scale of the LoRA layers with the given name.
        """
        loras = self.get_loras_by_name(name)
        assert all([lora.scale == loras[0].scale for lora in loras]), "lora scales are not all the same"
        return loras[0].scale

    def set_scale(self, name: str, scale: float, /) -> None:
        """Set the scale of the LoRA with the given name.

        Args:
            name: The name of the LoRA.
            scale: The new scale to set.
        """
        self.update_scales({name: scale})

    def update_scales(self, scales: dict[str, float], /) -> None:
        """Update the scales of multiple LoRAs.

        Args:
            scales: The scales to update.
                (keys are the names of the LoRAs, values are the new scales to set)
        """
        assert all([name in self.names for name in scales]), f"Scales keys must be a subset of {self.names}"
        for name, scale in scales.items():
            for lora in self.get_loras_by_name(name):
                lora.scale = scale

    @property
    def loras(self) -> list[Lora[Any]]:
        """List of all the LoRA layers managed by the SDLoraManager."""
        unet_layers = cast(Iterator[Lora[Any]], self.unet.layers(Lora))
        text_encoder_layers = cast(Iterator[Lora[Any]], self.clip_text_encoder.layers(Lora))
        return [*unet_layers, *text_encoder_layers]

    @property
    def names(self) -> list[str]:
        """List of all the LoRA names managed the SDLoraManager"""
        return list(set(lora.name for lora in self.loras))

    @property
    def lora_adapters(self) -> list[LoraAdapter]:
        """List of all the LoraAdapters managed by the SDLoraManager."""
        return list(self.unet.layers(LoraAdapter)) + list(self.clip_text_encoder.layers(LoraAdapter))

    @property
    def scales(self) -> dict[str, float]:
        """The scales of all the LoRAs managed by the SDLoraManager."""
        return {name: self.get_scale(name) for name in self.names}

    @staticmethod
    def _pad(input: str, /, padding_length: int = 2) -> str:
        """Make all numbers the same length so they sort correctly.

        e.g. foo.3.bar -> foo.03.bar

        Args:
            input: The string to pad.
            padding_length: The length to pad the numbers to.

        Returns:
            The padded string.
        """

        new_split: list[str] = []
        for s in input.split("_"):
            if s.isdigit():
                new_split.append(s.zfill(padding_length))
            else:
                new_split.append(s)
        return "_".join(new_split)

    @staticmethod
    def sort_keys(key: str, /) -> tuple[str, int]:
        """Compute the score of a key, relatively to its suffix.

        When used by [`sorted`][sorted], the keys will only be sorted "at the suffix level".
        The idea is that sometimes closely related keys in the state dict are not in the
        same order as the one we expect, for instance `q -> k -> v` or `in -> out`. This
        attempts to fix that issue, not cases where distant layers are called in a different
        order.

        Args:
            key: The key to sort.

        Returns:
            The padded prefix of the key.
            A score depending on the key's suffix.
        """

        # this dict might not be exhaustive
        suffix_scores = {"q": 1, "k": 2, "v": 3, "in": 3, "out": 4, "out0": 4, "out_0": 4}
        patterns = ["_{}", "_{}_lora"]

        # apply patterns to the keys of suffix_scores
        key_char_order = {f.format(k): v for k, v in suffix_scores.items() for f in patterns}

        # get the suffix and score for `key` (default: no suffix, highest score = 5)
        (sfx, score) = next(((k, v) for k, v in key_char_order.items() if key.endswith(k)), ("", 5))

        padded_key_prefix = SDLoraManager._pad(key.removesuffix(sfx))
        return (padded_key_prefix, score)
