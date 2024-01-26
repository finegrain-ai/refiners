from warnings import warn

from torch import Tensor

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.lora import Lora, LoraAdapter
from refiners.foundationals.latent_diffusion.model import LatentDiffusionModel


class SDLoraManager:
    def __init__(
        self,
        target: LatentDiffusionModel,
    ) -> None:
        self.target = target

    @property
    def unet(self) -> fl.Chain:
        unet = self.target.unet
        assert isinstance(unet, fl.Chain)
        return unet

    @property
    def clip_text_encoder(self) -> fl.Chain:
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
        """Load the LoRA weights from a dictionary of tensors.

        Expects the keys to be in the commonly found formats on CivitAI's hub.
        """
        assert name not in self.names, f"LoRA {name} already exists"
        loras = Lora.from_dict(
            name, {key: value.to(device=self.target.device, dtype=self.target.dtype) for key, value in tensors.items()}
        )
        loras = {key: loras[key] for key in sorted(loras.keys(), key=SDLoraManager.sort_keys)}

        # if no key contains "unet" or "text", assume all keys are for the unet
        if all(["unet" not in key and "text" not in key for key in loras.keys()]):
            loras = {f"unet_{key}": value for key, value in loras.items()}

        self.add_loras_to_unet(loras)
        self.add_loras_to_text_encoder(loras)

        self.set_scale(name, scale)

    def add_multiple_loras(
        self,
        /,
        tensors: dict[str, dict[str, Tensor]],
        scale: dict[str, float] | None = None,
    ) -> None:
        for name, lora_tensors in tensors.items():
            self.add_loras(name, tensors=lora_tensors, scale=scale[name] if scale else 1.0)

    def add_loras_to_text_encoder(self, loras: dict[str, Lora], /) -> None:
        text_encoder_loras = {key: loras[key] for key in loras.keys() if "text" in key}
        SDLoraManager.auto_attach(text_encoder_loras, self.clip_text_encoder)

    def add_loras_to_unet(self, loras: dict[str, Lora], /) -> None:
        unet_loras = {key: loras[key] for key in loras.keys() if "unet" in key}
        exclude: list[str] = []
        exclude = [
            self.unet_exclusions[exclusion]
            for exclusion in self.unet_exclusions
            if all([exclusion not in key for key in unet_loras.keys()])
        ]
        SDLoraManager.auto_attach(unet_loras, self.unet, exclude=exclude)

    def remove_loras(self, *names: str) -> None:
        for lora_adapter in self.lora_adapters:
            for name in names:
                lora_adapter.remove_lora(name)

            if len(lora_adapter.loras) == 0:
                lora_adapter.eject()

    def remove_all(self) -> None:
        for lora_adapter in self.lora_adapters:
            lora_adapter.eject()

    def get_loras_by_name(self, name: str, /) -> list[Lora]:
        return [lora for lora in self.loras if lora.name == name]

    def get_scale(self, name: str, /) -> float:
        loras = self.get_loras_by_name(name)
        assert all([lora.scale == loras[0].scale for lora in loras]), "lora scales are not all the same"
        return loras[0].scale

    def set_scale(self, name: str, scale: float, /) -> None:
        self.update_scales({name: scale})

    def update_scales(self, scales: dict[str, float], /) -> None:
        assert all([name in self.names for name in scales]), f"Scales keys must be a subset of {self.names}"
        for name, scale in scales.items():
            for lora in self.get_loras_by_name(name):
                lora.scale = scale

    @property
    def loras(self) -> list[Lora]:
        return list(self.unet.layers(Lora)) + list(self.clip_text_encoder.layers(Lora))

    @property
    def names(self) -> list[str]:
        return list(set(lora.name for lora in self.loras))

    @property
    def lora_adapters(self) -> list[LoraAdapter]:
        return list(self.unet.layers(LoraAdapter)) + list(self.clip_text_encoder.layers(LoraAdapter))

    @property
    def unet_exclusions(self) -> dict[str, str]:
        return {
            "time": "TimestepEncoder",
            "res": "ResidualBlock",
            "downsample": "DownsampleBlock",
            "upsample": "UpsampleBlock",
        }

    @property
    def scales(self) -> dict[str, float]:
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
        # out0 happens sometimes as an alias for out ; this dict might not be exhaustive
        key_char_order = {"q": 1, "k": 2, "v": 3, "out": 4, "out0": 4}

        for i, s in enumerate(key.split("_")):
            if s in key_char_order:
                prefix = SDLoraManager.pad("_".join(key.split("_")[:i]))
                return (prefix, key_char_order[s])

        return (SDLoraManager.pad(key), 5)

    @staticmethod
    def auto_attach(
        loras: dict[str, Lora],
        target: fl.Chain,
        /,
        exclude: list[str] | None = None,
    ) -> None:
        failed_loras: dict[str, Lora] = {}
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
