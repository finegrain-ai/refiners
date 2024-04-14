import argparse
import gc
import os

import torch
from tqdm import tqdm
from transformers import FuyuForCausalLM

from refiners.fluxion.utils import load_tensors, save_to_safetensors


def convert_fuyu_huggingface(weights: dict[str, torch.Tensor]) -> None:
    """Convert a fuyu8b weights from HuggingFace to refiners."""
    # get depth from "blocks" keys
    depth = max([int(k.split(".")[3]) for k in weights.keys() if k.startswith("language_model.model.layers.")]) + 1

    rename_keys: list[tuple[str, str]] = [
        ("vision_embed_tokens.bias", "InputEncoder.image_encoder.Linear.bias"),
        ("vision_embed_tokens.weight", "InputEncoder.image_encoder.Linear.weight"),
        ("language_model.model.embed_tokens.weight", "InputEncoder.token_encoder.weight"),
        ("language_model.model.final_layernorm.weight", "LayerNorm.weight"),
        ("language_model.model.final_layernorm.bias", "LayerNorm.bias"),
        ("language_model.lm_head.weight", "Linear.weight")
    ]

    for i in tqdm(range(depth), desc="Modifying state dict"):
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.input_layernorm.weight", 
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_1.LayerNorm.weight"
            )
        )
        rename_keys.append(
            (   f"language_model.model.layers.{i}.input_layernorm.bias", 
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_1.LayerNorm.bias"
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.mlp.dense_h_to_4h.weight", 
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_2.FeedForward.Linear_1.weight"
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.mlp.dense_h_to_4h.bias", 
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_2.FeedForward.Linear_1.bias"
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.mlp.dense_4h_to_h.weight", 
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_2.FeedForward.Linear_2.weight"
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.mlp.dense_4h_to_h.bias", 
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_2.FeedForward.Linear_2.bias"
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.post_attention_layernorm.weight", 
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_2.LayerNorm.weight"
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.post_attention_layernorm.bias", 
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_2.LayerNorm.bias"
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.self_attn.dense.weight", 
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_1.FuyuSelfAttention.Linear.weight"
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.self_attn.dense.bias", 
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_1.FuyuSelfAttention.Linear.bias"
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.self_attn.q_layernorm.weight", 
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_1.FuyuSelfAttention.QKVProjection.Parallel.Chain_1.LayerNorm.weight"
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.self_attn.q_layernorm.bias", 
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_1.FuyuSelfAttention.QKVProjection.Parallel.Chain_1.LayerNorm.bias"
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.self_attn.k_layernorm.weight", 
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_1.FuyuSelfAttention.QKVProjection.Parallel.Chain_2.LayerNorm.weight"
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.self_attn.k_layernorm.bias", 
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_1.FuyuSelfAttention.QKVProjection.Parallel.Chain_2.LayerNorm.bias"
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.self_attn.query_key_value.weight", 
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_1.FuyuSelfAttention.QKVProjection.Linear.weight"
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.self_attn.query_key_value.bias", 
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_1.FuyuSelfAttention.QKVProjection.Linear.bias"
            )
        )

    # rename keys
    for old_key, new_key in rename_keys:
        weights[new_key] = weights.pop(old_key)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from",
        type=str,
        dest="source_path",
        help=(
            "Path to the Hugging Face model.state_dict. If not specified"
            "The model is downloaded and the state_dict extracted"
        ),
    )
    parser.add_argument(
        "--to",
        type=str,
        dest="output_path",
        default=None,
        required=True,
        help=(
            "Path to save the converted model"
        ),
    )
    args = parser.parse_args()
    
    if args.source_path is not None:
        weights = load_tensors(args.source_path)
    else:
        model_id = "adept/fuyu-8b"
        source = FuyuForCausalLM.from_pretrained(model_id)
        weights = source.state_dict()
        del source
        gc.collect()
        

    convert_fuyu_huggingface(weights)
    weights = {key: value for key, value in weights.items()}
    output_dir = os.path.split(args.output_path)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # torch.save(weights, args.output_path)
    save_to_safetensors(path=args.output_path, tensors=weights)
    
    

if __name__ == "__main__":
    main()
