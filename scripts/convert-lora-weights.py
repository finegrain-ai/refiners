# Note: this conversion script currently only support simple LoRAs which adapt
# the UNet's attentions such as https://huggingface.co/pcuenq/pokemon-lora

from typing import cast
import torch
from torch.nn.init import zeros_
from torch.nn import Parameter as TorchParameter

import refiners.fluxion.layers as fl

from refiners.fluxion.utils import save_to_safetensors
from refiners.foundationals.latent_diffusion.unet import UNet
from refiners.foundationals.latent_diffusion.lora import LoraTarget, apply_loras_to_target
from refiners.adapters.lora import Lora
from refiners.fluxion.utils import create_state_dict_mapping

from diffusers import DiffusionPipeline  # type: ignore


def get_weight(linear: fl.Linear) -> torch.Tensor:
    assert linear.bias is None
    return linear.state_dict()["weight"]


def build_loras_safetensors(module: fl.Chain, key_prefix: str) -> dict[str, torch.Tensor]:
    weights: list[torch.Tensor] = []
    for lora in module.layers(layer_type=Lora):
        linears = list(lora.layers(layer_type=fl.Linear))
        assert len(linears) == 2
        weights.extend((get_weight(linear=linears[1]), get_weight(linear=linears[0])))  # aka (up_weight, down_weight)
    return {f"{key_prefix}{i:03d}": w for i, w in enumerate(iterable=weights)}


@torch.no_grad()
def process(source: str, base_model: str, output_file: str) -> None:
    diffusers_state_dict = torch.load(source, map_location="cpu")  # type: ignore
    diffusers_sd = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path=base_model)  # type: ignore
    diffusers_model = cast(fl.Module, diffusers_sd.unet)  # type: ignore

    refiners_model = UNet(in_channels=4, clip_embedding_dim=768)
    target = LoraTarget.CrossAttention
    metadata = {"unet_targets": "CrossAttentionBlock2d"}
    rank = diffusers_state_dict[
        "mid_block.attentions.0.transformer_blocks.0.attn1.processor.to_q_lora.down.weight"
    ].shape[0]

    x = torch.randn(1, 4, 32, 32)
    timestep = torch.tensor(data=[0])
    clip_text_embeddings = torch.randn(1, 77, 768)

    refiners_model.set_timestep(timestep=timestep)
    refiners_model.set_clip_text_embedding(clip_text_embedding=clip_text_embeddings)
    refiners_args = (x,)

    diffusers_args = (x, timestep, clip_text_embeddings)

    diffusers_to_refiners = create_state_dict_mapping(
        source_model=refiners_model, target_model=diffusers_model, source_args=refiners_args, target_args=diffusers_args
    )
    assert diffusers_to_refiners is not None

    apply_loras_to_target(module=refiners_model, target=LoraTarget(target), rank=rank, scale=1.0)
    for layer in refiners_model.layers(layer_type=Lora):
        zeros_(tensor=layer.Linear_1.weight)

    targets = {k.split("_lora.")[0] for k in diffusers_state_dict.keys()}
    for target_k in targets:
        k_p, k_s = target_k.split(".processor.")
        r = [v for k, v in diffusers_to_refiners.items() if k.startswith(f"{k_p}.{k_s}")]
        assert len(r) == 1
        orig_k = r[0]
        orig_path = orig_k.split(sep=".")
        p = refiners_model
        for seg in orig_path[:-1]:
            p = p[seg]
            assert isinstance(p, fl.Chain)
        last_seg = (
            "LoraAdapter" if orig_path[-1] == "Linear" else f"LoraAdapter_{orig_path[-1].removeprefix('Linear_')}"
        )
        p_down = TorchParameter(data=diffusers_state_dict[f"{target_k}_lora.down.weight"])
        p_up = TorchParameter(data=diffusers_state_dict[f"{target_k}_lora.up.weight"])
        p[last_seg].Lora.load_weights(p_down, p_up)

    state_dict = build_loras_safetensors(module=refiners_model, key_prefix="unet.")
    assert len(state_dict) == 320
    save_to_safetensors(path=output_file, tensors=state_dict, metadata=metadata)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from",
        type=str,
        dest="source",
        required=True,
        help="Source file path (.bin)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=False,
        default="runwayml/stable-diffusion-v1-5",
        help="Base model",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        default="output.safetensors",
        help="Path for the output file",
    )
    args = parser.parse_args()
    process(source=args.source, base_model=args.base_model, output_file=args.output_file)


if __name__ == "__main__":
    main()
