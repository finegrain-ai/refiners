import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download  # type: ignore

from refiners.fluxion.utils import load_from_safetensors, save_to_safetensors


class Args(argparse.Namespace):
    source_path: str
    output_path: str | None
    use_half: bool


def convert(args: Args) -> dict[str, torch.Tensor]:
    if Path(args.source_path).suffix != ".safetensors":
        args.source_path = hf_hub_download(
            repo_id=args.source_path, filename="ella-sd1.5-tsc-t5xl.safetensors", local_dir="tests/weights/ELLA-Adapter"
        )
    weights = load_from_safetensors(args.source_path)

    for key in list(weights.keys()):
        if "latents" in key:
            new_key = "PerceiverResampler.Latents.ParameterInitialized.weight"
            weights[new_key] = weights.pop(key)
        elif "time_embedding" in key:
            new_key = key.replace("time_embedding", "TimestepEncoder.RangeEncoder").replace("linear", "Linear")
            weights[new_key] = weights.pop(key)
        elif "proj_in" in key:
            new_key = f"PerceiverResampler.Linear.{key.split('.')[-1]}"
            weights[new_key] = weights.pop(key)
        elif "time_aware" in key:
            new_key = f"PerceiverResampler.Residual.Linear.{key.split('.')[-1]}"
            weights[new_key] = weights.pop(key)
        elif "attn.in_proj" in key:
            layer_num = int(key.split(".")[2])
            query_param, key_param, value_param = weights.pop(key).chunk(3, dim=0)
            param_type = "weight" if "weight" in key else "bias"
            for i, param in enumerate([query_param, key_param, value_param]):
                new_key = f"PerceiverResampler.Transformer.TransformerLayer_{layer_num+1}.Residual_1.PerceiverAttention.Attention.Distribute.Linear_{i+1}.{param_type}"
                weights[new_key] = param
        elif "attn.out_proj" in key:
            layer_num = int(key.split(".")[2])
            new_key = f"PerceiverResampler.Transformer.TransformerLayer_{layer_num+1}.Residual_1.PerceiverAttention.Attention.Linear.{key.split('.')[-1]}"
            weights[new_key] = weights.pop(key)
        elif "ln_ff" in key:
            layer_num = int(key.split(".")[2])
            new_key = f"PerceiverResampler.Transformer.TransformerLayer_{layer_num+1}.Residual_2.AdaLayerNorm.Parallel.Chain.Linear.{key.split('.')[-1]}"
            weights[new_key] = weights.pop(key)
        elif "ln_1" in key or "ln_2" in key:
            layer_num = int(key.split(".")[2])
            n = 1 if int(key.split(".")[3].split("_")[-1]) == 2 else 2
            new_key = f"PerceiverResampler.Transformer.TransformerLayer_{layer_num+1}.Residual_1.PerceiverAttention.Distribute.AdaLayerNorm_{n}.Parallel.Chain.Linear.{key.split('.')[-1]}"
            weights[new_key] = weights.pop(key)
        elif "mlp" in key:
            layer_num = int(key.split(".")[2])
            n = 1 if "c_fc" in key else 2
            new_key = f"PerceiverResampler.Transformer.TransformerLayer_{layer_num+1}.Residual_2.FeedForward.Linear_{n}.{key.split('.')[-1]}"
            weights[new_key] = weights.pop(key)

    if args.use_half:
        weights = {key: value.half() for key, value in weights.items()}

    return weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a pretrained Ella Adapter to refiners implementation")
    parser.add_argument(
        "--from",
        type=str,
        dest="source_path",
        default="QQGYLab/ELLA",
        help=(
            "A path to a local .safetensors weights. If not provided, a repo from Hugging Face Hub will be used"
            "Default to QQGYLab/ELLA"
        ),
    )

    parser.add_argument(
        "--to",
        type=str,
        dest="output_path",
        default=None,
        help=(
            "Path to save the converted model (extension will be .safetensors). If not specified, the output path will"
            " be the source path with the prefix set to refiners"
        ),
    )
    parser.add_argument(
        "--half",
        action="store_true",
        dest="use_half",
        default=True,
        help="Use this flag to save the output file as half precision (default: full precision).",
    )
    args = parser.parse_args(namespace=Args())
    weights = convert(args)
    if args.output_path is None:
        args.output_path = f"{Path(args.source_path).stem}-refiners.safetensors"
    save_to_safetensors(path=args.output_path, tensors=weights)
