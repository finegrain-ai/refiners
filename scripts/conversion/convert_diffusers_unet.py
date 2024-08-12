import argparse
from pathlib import Path
from typing import Any

import torch
from diffusers import UNet2DConditionModel  # type: ignore
from torch import nn

from refiners.fluxion.model_converter import ModelConverter
from refiners.fluxion.utils import load_from_safetensors, load_tensors
from refiners.foundationals.latent_diffusion import SD1UNet, SDXLUNet
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.lcm import SDXLLcmAdapter


class Args(argparse.Namespace):
    source_path: str
    output_path: str | None
    subfolder: str
    half: bool
    verbose: bool
    skip_init_check: bool
    override_weights: str | None


def setup_converter(args: Args) -> ModelConverter:
    # low_cpu_mem_usage=False stops some annoying console messages us to `pip install accelerate`
    source: nn.Module = UNet2DConditionModel.from_pretrained(  # type: ignore
        pretrained_model_name_or_path=args.source_path,
        subfolder=args.subfolder,
        low_cpu_mem_usage=False,
    )
    if args.override_weights is not None:
        if args.override_weights.endswith(".pth"):
            sd = load_tensors(args.override_weights)
        elif args.override_weights.endswith(".safetensors"):
            sd = load_from_safetensors(args.override_weights)
        else:
            raise ValueError(f"Unsupported file format: {args.override_weights}")
        source.load_state_dict(sd)
    source_in_channels: int = source.config.in_channels  # type: ignore
    source_clip_embedding_dim: int = source.config.cross_attention_dim  # type: ignore
    source_has_time_ids: bool = source.config.addition_embed_type == "text_time"  # type: ignore
    source_is_lcm: bool = source.config.time_cond_proj_dim is not None

    if source_has_time_ids:
        target = SDXLUNet(in_channels=source_in_channels)
    else:
        target = SD1UNet(in_channels=source_in_channels)

    if source_is_lcm:
        assert isinstance(target, SDXLUNet)
        SDXLLcmAdapter(target=target).inject()

    x = torch.randn(1, source_in_channels, 32, 32)
    timestep = torch.tensor(data=[0])
    clip_text_embeddings = torch.randn(1, 77, source_clip_embedding_dim)

    target.set_timestep(timestep=timestep)
    target.set_clip_text_embedding(clip_text_embedding=clip_text_embeddings)
    added_cond_kwargs = {}
    if isinstance(target, SDXLUNet):
        added_cond_kwargs = {"text_embeds": torch.randn(1, 1280), "time_ids": torch.randn(1, 6)}
        target.set_time_ids(time_ids=added_cond_kwargs["time_ids"])
        target.set_pooled_text_embedding(pooled_text_embedding=added_cond_kwargs["text_embeds"])

    target_args = (x,)

    source_kwargs: dict[str, Any] = {}
    if source_has_time_ids:
        source_kwargs["added_cond_kwargs"] = added_cond_kwargs
    if source_is_lcm:
        source_kwargs["timestep_cond"] = torch.randn(1, source.config.time_cond_proj_dim)

    source_args = {
        "positional": (x, timestep, clip_text_embeddings),
        "keyword": source_kwargs,
    }

    converter = ModelConverter(
        source_model=source,
        target_model=target,
        skip_init_check=args.skip_init_check,
        skip_output_check=True,
        verbose=args.verbose,
    )
    if not converter.run(
        source_args=source_args,
        target_args=target_args,
    ):
        raise RuntimeError("Model conversion failed")
    return converter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converts a Diffusion UNet model to a Refiners SD1UNet or SDXLUNet model"
    )
    parser.add_argument(
        "--from",
        type=str,
        dest="source_path",
        default="runwayml/stable-diffusion-v1-5",
        help=(
            "Can be a path to a .bin file, a .safetensors file or a model name from the HuggingFace Hub. Default:"
            " runwayml/stable-diffusion-v1-5"
        ),
    )
    parser.add_argument(
        "--override-weights",
        type=str,
        default=None,
        help=(
            "Path to a weights file to override the source model (keeping its config). "
            "This is useful for models distributed as .pth files."
        ),
    )
    parser.add_argument(
        "--to",
        type=str,
        dest="output_path",
        default=None,
        help=(
            "Output path (.safetensors) for converted model. If not provided, the output path will be the same as the"
            " source path."
        ),
    )
    parser.add_argument("--subfolder", type=str, default="unet", help="Subfolder. Default: unet.")
    parser.add_argument(
        "--skip-init-check",
        action="store_true",
        help="Skip check that source and target have the same layers count.",
    )
    parser.add_argument("--half", action="store_true", help="Convert to half precision.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Prints additional information during conversion. Default: False",
    )
    args = parser.parse_args(namespace=Args())
    if args.output_path is None:
        args.output_path = f"{Path(args.source_path).stem}-unet.safetensors"
    converter = setup_converter(args=args)
    converter.save_to_safetensors(path=args.output_path, half=args.half)


if __name__ == "__main__":
    main()
