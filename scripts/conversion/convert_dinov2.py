import argparse
from pathlib import Path

import torch

from refiners.fluxion.utils import load_tensors, save_to_safetensors


def convert_dinov2_facebook(weights: dict[str, torch.Tensor]) -> None:
    """Convert a DINOv2 weights from facebook to refiners."""
    # get depth from "blocks" keys
    depth = max([int(k.split(".")[1]) for k in weights.keys() if k.startswith("blocks.")]) + 1

    # only needed when pre-training
    del weights["mask_token"]

    # squeeze cls_token and position_embeddings
    weights["cls_token"] = weights["cls_token"].squeeze(0)
    weights["pos_embed"] = weights["pos_embed"].squeeze(0)

    rename_keys: list[tuple[str, str]] = [
        ("cls_token", "Concatenate.ClassToken.Parameter.weight"),
        ("pos_embed", "PositionalEncoder.Parameter.weight"),
        ("patch_embed.proj.weight", "Concatenate.PatchEncoder.Conv2d.weight"),
        ("patch_embed.proj.bias", "Concatenate.PatchEncoder.Conv2d.bias"),
        ("norm.weight", "LayerNorm.weight"),
        ("norm.bias", "LayerNorm.bias"),
    ]
    for i in range(depth):
        rename_keys.append(
            (
                f"blocks.{i}.norm1.weight",
                f"Transformer.TransformerLayer_{i+1}.Residual_1.LayerNorm.weight",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.norm1.bias",
                f"Transformer.TransformerLayer_{i+1}.Residual_1.LayerNorm.bias",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.attn.proj.weight",
                f"Transformer.TransformerLayer_{i+1}.Residual_1.SelfAttention.Linear.weight",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.attn.proj.bias",
                f"Transformer.TransformerLayer_{i+1}.Residual_1.SelfAttention.Linear.bias",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.ls1.gamma",
                f"Transformer.TransformerLayer_{i+1}.Residual_1.LayerScale.weight",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.norm2.weight",
                f"Transformer.TransformerLayer_{i+1}.Residual_2.LayerNorm.weight",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.norm2.bias",
                f"Transformer.TransformerLayer_{i+1}.Residual_2.LayerNorm.bias",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.mlp.fc1.weight",
                f"Transformer.TransformerLayer_{i+1}.Residual_2.FeedForward.Linear_1.weight",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.mlp.fc1.bias",
                f"Transformer.TransformerLayer_{i+1}.Residual_2.FeedForward.Linear_1.bias",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.mlp.fc2.weight",
                f"Transformer.TransformerLayer_{i+1}.Residual_2.FeedForward.Linear_2.weight",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.mlp.fc2.bias",
                f"Transformer.TransformerLayer_{i+1}.Residual_2.FeedForward.Linear_2.bias",
            ),
        )
        rename_keys.append(
            (
                f"blocks.{i}.ls2.gamma",
                f"Transformer.TransformerLayer_{i+1}.Residual_2.LayerScale.weight",
            ),
        )

    if "register_tokens" in weights:
        weights["register_tokens"] = weights["register_tokens"].squeeze(0)
        rename_keys.append(("register_tokens", "Registers.Parameter.weight"))

    # rename keys
    for old_key, new_key in rename_keys:
        weights[new_key] = weights.pop(old_key)

    # split the qkv weights and biases
    for i in range(depth):
        qkv_weight = weights.pop(f"blocks.{i}.attn.qkv.weight")
        q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
        weights[f"Transformer.TransformerLayer_{i+1}.Residual_1.SelfAttention.Distribute.Linear_1.weight"] = q_weight
        weights[f"Transformer.TransformerLayer_{i+1}.Residual_1.SelfAttention.Distribute.Linear_2.weight"] = k_weight
        weights[f"Transformer.TransformerLayer_{i+1}.Residual_1.SelfAttention.Distribute.Linear_3.weight"] = v_weight

        qkv_bias = weights.pop(f"blocks.{i}.attn.qkv.bias")
        q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)
        weights[f"Transformer.TransformerLayer_{i+1}.Residual_1.SelfAttention.Distribute.Linear_1.bias"] = q_bias
        weights[f"Transformer.TransformerLayer_{i+1}.Residual_1.SelfAttention.Distribute.Linear_2.bias"] = k_bias
        weights[f"Transformer.TransformerLayer_{i+1}.Residual_1.SelfAttention.Distribute.Linear_3.bias"] = v_bias


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from",
        type=str,
        required=True,
        dest="source_path",
        help=(
            "Official checkpoint from https://github.com/facebookresearch/dinov2"
            " e.g. /path/to/dinov2_vits14_pretrain.pth"
        ),
    )
    parser.add_argument(
        "--to",
        type=str,
        dest="output_path",
        default=None,
        help=(
            "Path to save the converted model. If not specified, the output path will be the source path with the"
            " extension changed to .safetensors."
        ),
    )
    parser.add_argument("--half", action="store_true", dest="half")
    args = parser.parse_args()

    weights = load_tensors(args.source_path)
    convert_dinov2_facebook(weights)
    if args.half:
        weights = {key: value.half() for key, value in weights.items()}
    if args.output_path is None:
        args.output_path = f"{Path(args.source_path).stem}.safetensors"
    save_to_safetensors(path=args.output_path, tensors=weights)


if __name__ == "__main__":
    main()
