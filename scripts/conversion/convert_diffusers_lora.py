import argparse
from pathlib import Path
from typing import cast
import torch
from torch import Tensor
from torch.nn.init import zeros_
from torch.nn import Parameter as TorchParameter
from diffusers import DiffusionPipeline  # type: ignore
import refiners.fluxion.layers as fl
from refiners.fluxion.model_converter import ModelConverter
from refiners.fluxion.utils import save_to_safetensors
from refiners.foundationals.latent_diffusion import SD1UNet
from refiners.foundationals.latent_diffusion.lora import LoraTarget, apply_loras_to_target
from refiners.adapters.lora import Lora


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


class Args(argparse.Namespace):
    source_path: str
    base_model: str
    output_file: str
    verbose: bool


@torch.no_grad()
def process(args: Args) -> None:
    diffusers_state_dict = cast(dict[str, Tensor], torch.load(args.source_path, map_location="cpu"))  # type: ignore
    diffusers_sd = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path=args.base_model)  # type: ignore
    diffusers_model = cast(fl.Module, diffusers_sd.unet)  # type: ignore

    refiners_model = SD1UNet(in_channels=4, clip_embedding_dim=768)
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

    converter = ModelConverter(
        source_model=refiners_model, target_model=diffusers_model, skip_output_check=True, verbose=args.verbose
    )
    if not converter.run(
        source_args=refiners_args,
        target_args=diffusers_args,
    ):
        raise RuntimeError("Model conversion failed")

    diffusers_to_refiners = converter.get_mapping()

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
    save_to_safetensors(path=args.output_path, tensors=state_dict, metadata=metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LoRAs saved using the diffusers library to refiners format.")
    parser.add_argument(
        "--from",
        type=str,
        dest="source_path",
        required=True,
        help="Source file path (.bin|safetensors) containing the LoRAs.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=False,
        default="runwayml/stable-diffusion-v1-5",
        help="Base model, used for the UNet structure. Default: runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument(
        "--to",
        type=str,
        dest="output_path",
        required=False,
        default=None,
        help=(
            "Output file path (.safetensors) for converted LoRAs. If not provided, the output path will be the same as"
            " the source path."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        dest="verbose",
        default=False,
        help="Use this flag to print verbose output during conversion.",
    )
    args = parser.parse_args(namespace=Args())
    if args.output_path is None:
        args.output_path = f"{Path(args.source_path).stem}-refiners.safetensors"
    process(args=args)


if __name__ == "__main__":
    main()
