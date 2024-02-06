from typing import Any, Iterator, cast
from warnings import warn

from torch import Tensor

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.lora import Lora, LoraAdapter
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
        self.add_loras_to_unet(loras)
        self.add_loras_to_text_encoder(loras)

        # set the scale of the LoRA
        self.set_scale(name, scale)

    def add_multiple_loras(
        self,
        /,
        tensors: dict[str, dict[str, Tensor]],
        scale: dict[str, float] | None = None,
    ) -> None:
        """Load multiple LoRAs from a dictionary of `state_dict`.

        Args:
            tensors: The dictionary of `state_dict` of the LoRAs to load
                (keys are the names of the LoRAs, values are the `state_dict` of the LoRAs).
            scale: The scales to use for the LoRAs.

        Raises:
            AssertionError: If the manager already has a LoRA with the same name.
        """
        for name, lora_tensors in tensors.items():
            self.add_loras(name, tensors=lora_tensors, scale=scale[name] if scale else 1.0)

    def add_loras_to_text_encoder(self, loras: dict[str, Lora[Any]], /) -> None:
        """Add multiple LoRAs to the text encoder.

        Args:
            loras: The dictionary of LoRAs to add to the text encoder.
                (keys are the names of the LoRAs, values are the LoRAs to add to the text encoder)
        """
        text_encoder_loras = {key: loras[key] for key in loras.keys() if "text" in key}
        SDLoraManager.auto_attach(text_encoder_loras, self.clip_text_encoder)

    def add_loras_to_unet(self, loras: dict[str, Lora[Any]], /) -> None:
        """Add multiple LoRAs to the U-Net.

        Args:
            loras: The dictionary of LoRAs to add to the U-Net.
                (keys are the names of the LoRAs, values are the LoRAs to add to the U-Net)
        """
        unet_loras = {key: loras[key] for key in loras.keys() if "unet" in key}
        exclude = [
            block for s, block in self.unet_exclusions.items() if all([s not in key for key in unet_loras.keys()])
        ]
        SDLoraManager.auto_attach(unet_loras, self.unet, exclude=exclude)

    def remove_loras(self, *names: str) -> None:
        """Remove mulitple LoRAs from the target.

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
        """Update the scales of mulitple LoRAs.

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
    def unet_exclusions(self) -> dict[str, str]:
        return {
            "time": "TimestepEncoder",
            "res": "ResidualBlock",
            "downsample": "Downsample",
            "upsample": "Upsample",
        }

    @property
    def scales(self) -> dict[str, float]:
        """The scales of all the LoRAs managed by the SDLoraManager."""
        return {name: self.get_scale(name) for name in self.names}

    @staticmethod
    def pad(input: str, /, padding_length: int = 2) -> str:
        new_split: list[str] = []
        for s in input.split("_"):
            if s.isdigit():
                new_split.append(s.zfill(padding_length))
            else:
                new_split.append(s)
        return "_".join(new_split)

    @staticmethod
    def sort_keys(key: str, /) -> tuple[str, int]:
        # this dict might not be exhaustive
        suffix_scores = {"q": 1, "k": 2, "v": 3, "in": 3, "out": 4, "out0": 4, "out_0": 4}
        patterns = ["_{}", "_{}_lora"]
        key_char_order = {f.format(k): v for k, v in suffix_scores.items() for f in patterns}
        (sfx, score) = next(((k, v) for k, v in key_char_order.items() if key.endswith(k)), ("", 5))
        return (SDLoraManager.pad(key.removesuffix(sfx)), score)

    @staticmethod
    def auto_attach(
        loras: dict[str, Lora[Any]],
        target: fl.Chain,
        /,
        exclude: list[str] | None = None,
    ) -> None:
        failed_loras: dict[str, Lora[Any]] = {}
        for key, lora in loras.items():
            if attach := lora.auto_attach(target, exclude=exclude):
                adapter, parent = attach
                # if parent is None, `adapter` is already attached and `lora` has been added to it
                if parent is not None:
                    adapter.inject(parent)
            else:
                failed_loras[key] = lora

        if failed_loras:
            warn(f"failed to attach {len(failed_loras)}/{len(loras)} loras to {target.__class__.__name__}")

        # TODO: add a stronger sanity check to make sure loras are attached correctly
