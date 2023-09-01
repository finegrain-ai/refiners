from enum import Enum
from pathlib import Path
from typing import Iterator

from torch import Tensor

import refiners.fluxion.layers as fl
from refiners.fluxion.utils import load_from_safetensors, load_metadata_from_safetensors

from refiners.adapters.adapter import Adapter
from refiners.adapters.lora import SingleLoraAdapter, LoraAdapter

from refiners.foundationals.clip.text_encoder import FeedForward, TransformerLayer
from refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock2d
from refiners.foundationals.latent_diffusion import (
    StableDiffusion_1,
    SD1UNet,
    CLIPTextEncoderL,
    LatentDiffusionAutoencoder,
)
from refiners.foundationals.latent_diffusion.stable_diffusion_1.controlnet import Controlnet

MODELS = ["unet", "text_encoder", "lda"]


class LoraTarget(str, Enum):
    Self = "self"
    Attention = "Attention"
    SelfAttention = "SelfAttention"
    CrossAttention = "CrossAttentionBlock2d"
    FeedForward = "FeedForward"
    TransformerLayer = "TransformerLayer"

    def get_class(self) -> type[fl.Chain]:
        match self:
            case LoraTarget.Self:
                return fl.Chain
            case LoraTarget.Attention:
                return fl.Attention
            case LoraTarget.SelfAttention:
                return fl.SelfAttention
            case LoraTarget.CrossAttention:
                return CrossAttentionBlock2d
            case LoraTarget.FeedForward:
                return FeedForward
            case LoraTarget.TransformerLayer:
                return TransformerLayer


def lora_targets(
    module: fl.Chain,
    target: LoraTarget | list[LoraTarget],
) -> Iterator[tuple[fl.Linear, fl.Chain]]:
    if isinstance(target, list):
        for t in target:
            yield from lora_targets(module, t)
        return

    lookup_class = fl.Linear if target == LoraTarget.Self else target.get_class()

    if isinstance(module, SD1UNet):

        def predicate(m: fl.Module, p: fl.Chain) -> bool:
            if isinstance(m, Controlnet):  # do not adapt Controlnet linears
                raise StopIteration
            return isinstance(m, lookup_class)

    else:

        def predicate(m: fl.Module, p: fl.Chain) -> bool:
            return isinstance(m, lookup_class)

    if target == LoraTarget.Self:
        for m, p in module.walk(predicate):
            assert isinstance(m, fl.Linear)
            yield (m, p)
        return

    for layer, _ in module.walk(predicate):
        for t in layer.walk(fl.Linear):
            yield t


class SD1LoraAdapter(fl.Chain, Adapter[StableDiffusion_1]):
    metadata: dict[str, str] | None
    tensors: dict[str, Tensor]

    def __init__(
        self,
        target: StableDiffusion_1,
        sub_targets: dict[str, list[LoraTarget]],
        scale: float = 1.0,
        weights: dict[str, Tensor] | None = None,
    ):
        with self.setup_adapter(target):
            super().__init__(target)

        self.sub_adapters: list[LoraAdapter[SD1UNet | CLIPTextEncoderL | LatentDiffusionAutoencoder]] = []

        for model_name in MODELS:
            if not (model_targets := sub_targets.get(model_name, [])):
                continue
            model = getattr(target, model_name)
            if model.find(SingleLoraAdapter):
                raise NotImplementedError(f"{model} already contains LoRA layers")

            lora_weights = [weights[k] for k in sorted(weights) if k.startswith(model_name)] if weights else None
            self.sub_adapters.append(
                LoraAdapter[type(model)](
                    model,
                    sub_targets=lora_targets(model, model_targets),
                    scale=scale,
                    weights=lora_weights,
                )
            )

    @classmethod
    def from_safetensors(
        cls,
        target: StableDiffusion_1,
        checkpoint_path: Path | str,
        scale: float = 1.0,
    ):
        metadata = load_metadata_from_safetensors(checkpoint_path)
        assert metadata is not None, "Invalid safetensors checkpoint: missing metadata"
        tensors = load_from_safetensors(checkpoint_path, device=target.device)

        sub_targets = {}
        for model_name in MODELS:
            if not (v := metadata.get(f"{model_name}_targets", "")):
                continue
            sub_targets[model_name] = [LoraTarget(x) for x in v.split(",")]

        return cls(
            target,
            sub_targets,
            scale=scale,
            weights=tensors,
        )

    def inject(self: "SD1LoraAdapter", parent: fl.Chain | None = None) -> "SD1LoraAdapter":
        for adapter in self.sub_adapters:
            adapter.inject()
        return super().inject(parent)

    def eject(self) -> None:
        for adapter in self.sub_adapters:
            adapter.eject()
        super().eject()
