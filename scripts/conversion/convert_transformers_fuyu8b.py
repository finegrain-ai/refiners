import argparse
import gc
import gzip
import shutil
from pathlib import Path

from torch import Tensor
from tqdm import tqdm
from transformers import FuyuForCausalLM, FuyuProcessor  # type: ignore[reportMissingTypeStubs]

from refiners.fluxion.utils import save_to_safetensors


def convert_fuyu_huggingface(weights: dict[str, Tensor]) -> None:
    """Convert a fuyu8b weights from HuggingFace to refiners."""
    # get depth from "blocks" keys
    depth = max([int(k.split(".")[3]) for k in weights.keys() if k.startswith("language_model.model.layers.")]) + 1

    rename_keys: list[tuple[str, str]] = [
        ("vision_embed_tokens.bias", "InputEncoder.image_encoder.Linear.bias"),
        ("vision_embed_tokens.weight", "InputEncoder.image_encoder.Linear.weight"),
        ("language_model.model.embed_tokens.weight", "InputEncoder.token_encoder.weight"),
        ("language_model.model.final_layernorm.weight", "LayerNorm.weight"),
        ("language_model.model.final_layernorm.bias", "LayerNorm.bias"),
        ("language_model.lm_head.weight", "Linear.weight"),
    ]

    for i in tqdm(range(depth), desc="Modifying state dict"):
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.input_layernorm.weight",
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_1.LayerNorm.weight",
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.input_layernorm.bias",
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_1.LayerNorm.bias",
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.mlp.dense_h_to_4h.weight",
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_2.FeedForward.Linear_1.weight",
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.mlp.dense_h_to_4h.bias",
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_2.FeedForward.Linear_1.bias",
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.mlp.dense_4h_to_h.weight",
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_2.FeedForward.Linear_2.weight",
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.mlp.dense_4h_to_h.bias",
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_2.FeedForward.Linear_2.bias",
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.post_attention_layernorm.weight",
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_2.LayerNorm.weight",
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.post_attention_layernorm.bias",
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_2.LayerNorm.bias",
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.self_attn.dense.weight",
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_1.FuyuSelfAttention.Linear.weight",
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.self_attn.dense.bias",
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_1.FuyuSelfAttention.Linear.bias",
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.self_attn.q_layernorm.weight",
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_1.FuyuSelfAttention.QKVProjection.Parallel.Chain_1.LayerNorm.weight",
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.self_attn.q_layernorm.bias",
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_1.FuyuSelfAttention.QKVProjection.Parallel.Chain_1.LayerNorm.bias",
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.self_attn.k_layernorm.weight",
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_1.FuyuSelfAttention.QKVProjection.Parallel.Chain_2.LayerNorm.weight",
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.self_attn.k_layernorm.bias",
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_1.FuyuSelfAttention.QKVProjection.Parallel.Chain_2.LayerNorm.bias",
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.self_attn.query_key_value.weight",
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_1.FuyuSelfAttention.QKVProjection.Linear.weight",
            )
        )
        rename_keys.append(
            (
                f"language_model.model.layers.{i}.self_attn.query_key_value.bias",
                f"FuyuTransformer.FuyuTransformerLayer_{i+1}.Residual_1.FuyuSelfAttention.QKVProjection.Linear.bias",
            )
        )

    # rename keys
    for old_key, new_key in rename_keys:
        weights[new_key] = weights.pop(old_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--to",
        type=str,
        dest="output_path",
        default=None,
        help=("Path to save the converted model"),
    )
    parser.add_argument("--half", action="store_true", dest="half")
    args = parser.parse_args()

    # create refiners .cache folder
    refiners_cache_path = Path.home() / ".cache/refiners/fuyu-8b/"
    refiners_cache_path.mkdir(parents=True, exist_ok=True)

    model_id = "adept/fuyu-8b"

    # extract the json vocabulary file from the hf .cache
    processor: FuyuProcessor = FuyuProcessor.from_pretrained(pretrained_model_name_or_path=model_id)  # type: ignore[reportUnknownMemberType, reportAssignmentType]
    vocabulary_file_path: str = Path(processor.tokenizer.vocab_file).absolute().parent / "tokenizer.json"  # type: ignore[reportUnknownMemberType, reportUnknownVariableType]
    vocabulary_save_path = refiners_cache_path / "tokenizer.json.gz"

    # compress the json vocabulary file
    with open(vocabulary_file_path, "rb") as original_file:
        with gzip.open(vocabulary_save_path, "wb") as compressed_file:
            shutil.copyfileobj(original_file, compressed_file)

    # get hf fuyu weights
    source: FuyuForCausalLM = FuyuForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id)  # type: ignore[reportUnknownMemberType, reportAssignmentType]
    weights = source.state_dict()
    del source
    gc.collect()

    # converts weights
    convert_fuyu_huggingface(weights)
    if args.half:
        weights = {key: value.half() for key, value in weights.items()}

    # save to output path if specified, otherwise save to refiners .cache folder
    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = refiners_cache_path / "fuyu.safetensors"

    save_to_safetensors(path=output_path, tensors=weights)


if __name__ == "__main__":
    main()
