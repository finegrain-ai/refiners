from enum import Enum
from pathlib import Path
from typing import Callable, Iterator

from torch import Tensor

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.adapter import Adapter
from refiners.fluxion.adapters.lora import Lora, LoraAdapter
from refiners.fluxion.utils import load_from_safetensors, load_metadata_from_safetensors
from refiners.foundationals.segment_anything.image_encoder import FusedSelfAttention, SAMViTH, TransformerLayer
from refiners.foundationals.segment_anything.mask_decoder import MaskDecoder
from refiners.foundationals.segment_anything.model import SegmentAnything
from refiners.foundationals.segment_anything.prompt_encoder import CoordinateEncoder, MaskEncoder, PointEncoder
from refiners.foundationals.segment_anything.transformer import (
    SparseCrossDenseAttention,
    TwoWayTranformerLayer,
)

MODELS = ["image_encoder", "point_encoder", "mask_encoder", "mask_decoder"]


class LoraTarget(str, Enum):
    Self = "self"
    Attention = "Attention"
    FusedSelfAttention = "FusedSelfAttention"
    SparseCrossDenseAttention = "SparseCrossDenseAttention"
    TwoWayTranformerLayer = "TwoWayTransformerLayer"
    CoordinateEncoder = "CoordinateEncoder"
    TransformerLayer = "TransformerLayer"

    def get_class(self) -> type[fl.Chain]:
        match self:
            case LoraTarget.Self:
                return fl.Chain
            case LoraTarget.Attention:
                return fl.Attention
            case LoraTarget.FusedSelfAttention:
                return FusedSelfAttention
            case LoraTarget.SparseCrossDenseAttention:
                return SparseCrossDenseAttention
            case LoraTarget.TwoWayTranformerLayer:
                return TwoWayTranformerLayer
            case LoraTarget.CoordinateEncoder:
                return CoordinateEncoder
            case LoraTarget.TransformerLayer:
                return TransformerLayer


def _predicate(k: type[fl.Module]) -> Callable[[fl.Module, fl.Chain], bool]:
    def f(m: fl.Module, _: fl.Chain) -> bool:
        if isinstance(m, Lora):  # do not adapt other LoRAs
            raise StopIteration
        return isinstance(m, k)

    return f


def _iter_linears(module: fl.Chain) -> Iterator[tuple[fl.Linear, fl.Chain]]:
    for m, p in module.walk(_predicate(fl.Linear)):
        assert isinstance(m, fl.Linear)
        yield (m, p)


def lora_targets(
    module: fl.Chain,
    target: LoraTarget | list[LoraTarget],
) -> Iterator[tuple[fl.Linear, fl.Chain]]:
    if isinstance(target, list):
        for t in target:
            yield from lora_targets(module, t)
        return

    if target == LoraTarget.Self:
        yield from _iter_linears(module)
        return
    
    for layer, _ in module.walk(_predicate(target.get_class())):
        assert isinstance(layer, fl.Chain)
        yield from _iter_linears(layer)


class SAMLoraAdapter(fl.Chain, Adapter[SegmentAnything]):
    metadata: dict[str, str] | None
    tensors: dict[str, Tensor]

    def __init__(
        self,
        target: SegmentAnything,
        sub_targets: dict[str, list[LoraTarget]],
        scale: float = 1.0,
        weights: dict[str, Tensor] | None = None,
        rank: int | None = None,
    ):
        with self.setup_adapter(target):
            super().__init__(target)
        
        self.sub_adapters: list[LoraAdapter[SAMViTH | MaskEncoder | PointEncoder | MaskDecoder]] = []
            
        for model_name in MODELS:

            if not (model_targets := sub_targets.get(model_name, [])):
                continue

            model = getattr(target, model_name)

            lora_weights = [weights[k] for k in sorted(weights) if k.startswith(model_name)] if weights else None
            self.sub_adapters.append(
                LoraAdapter[type(model)](
                    model,
                    sub_targets=lora_targets(model, model_targets),
                    scale=scale,
                    weights=lora_weights,
                    rank=rank,
                )
            )

    @classmethod
    def from_safetensors(
        cls,
        target: SegmentAnything,
        checkpoint_path: Path | str,
        scale: float = 1.0,
        rank: int | None = None,
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
            rank=rank,
        )

    def inject(self: "SAMLoraAdapter", parent: fl.Chain | None = None) -> "SAMLoraAdapter":
        for adapter in self.sub_adapters:
            adapter.inject()
        return super().inject(parent)

    def eject(self) -> None:
        for adapter in self.sub_adapters:
            adapter.eject()
        super().eject()
