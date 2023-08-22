from enum import Enum
from pathlib import Path
from typing import Iterator


from torch import Tensor, device as Device
from torch.nn import Parameter as TorchParameter
from refiners.adapters.lora import LoraAdapter, load_lora_weights
from refiners.foundationals.clip.text_encoder import FeedForward, TransformerLayer
from refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock2d
from refiners.foundationals.latent_diffusion import StableDiffusion_1
from refiners.foundationals.latent_diffusion.controlnet import Controlnet
import refiners.fluxion.layers as fl
from refiners.fluxion.utils import load_from_safetensors, load_metadata_from_safetensors


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


def get_lora_rank(weights: list[Tensor]) -> int:
    ranks: set[int] = {w.shape[1] for w in weights[0::2]}
    assert len(ranks) == 1
    return ranks.pop()


def lora_targets(module: fl.Chain, target: LoraTarget) -> Iterator[tuple[fl.Linear, fl.Chain]]:
    it = [module] if target == LoraTarget.Self else module.layers(layer_type=target.get_class())
    for layer in it:
        for t in layer.walk(fl.Linear):
            yield t


def apply_loras_to_target(module: fl.Chain, target: LoraTarget, rank: int, scale: float) -> None:
    for linear, parent in lora_targets(module, target):
        adapter = LoraAdapter(target=linear, rank=rank, scale=scale)
        adapter.inject(parent)


class LoraWeights:
    """A single LoRA weights training checkpoint used to patch a Stable Diffusion 1.5 model."""

    metadata: dict[str, str] | None
    tensors: dict[str, Tensor]

    def __init__(self, checkpoint_path: Path | str, device: Device | str):
        self.metadata = load_metadata_from_safetensors(checkpoint_path)
        self.tensors = load_from_safetensors(checkpoint_path, device=device)

    def patch(self, sd: StableDiffusion_1, scale: float = 1.0) -> None:
        assert self.metadata is not None, "Invalid safetensors checkpoint: missing metadata"

        for meta_key, meta_value in self.metadata.items():
            match meta_key:
                case "unet_targets":
                    # TODO: support this transparently
                    if any([isinstance(module, Controlnet) for module in sd.unet]):
                        raise NotImplementedError("Cannot patch a UNet which already contains a Controlnet adapter")
                    model = sd.unet
                    key_prefix = "unet."
                case "text_encoder_targets":
                    model = sd.clip_text_encoder
                    key_prefix = "text_encoder."
                case "lda_targets":
                    model = sd.lda
                    key_prefix = "lda."
                case _:
                    raise ValueError(f"Unexpected key in checkpoint metadata: {meta_key}")

            # TODO(FG-487): support loading multiple LoRA-s
            if any(model.layers(LoraAdapter)):
                raise NotImplementedError(f"{model.__class__.__name__} already contains LoRA layers")

            lora_weights = [w for w in [self.tensors[k] for k in sorted(self.tensors) if k.startswith(key_prefix)]]
            assert len(lora_weights) % 2 == 0

            rank = get_lora_rank(lora_weights)
            for target in meta_value.split(","):
                apply_loras_to_target(model, target=LoraTarget(target), rank=rank, scale=scale)

            assert len(list(model.layers(LoraAdapter))) == (len(lora_weights) // 2)

            load_lora_weights(model, [TorchParameter(w) for w in lora_weights])
