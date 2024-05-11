import argparse

from pathlib import Path

import torch
from torch import nn
from refiners.fluxion.utils import no_grad, save_to_safetensors
from refiners.foundationals.mistral import Mistral7b
from transformers import MistralForCausalLM  # type: ignore


def get_tranformer_layer_mapping(index: int):
    """Maps Refiners Mistral Transformer layer components to HuggingFace Mistral components.
    """
    return {
        f"MistralTransformer.MistralTranformerLayer_{index+1}.Residual_1.MistralRMSNorm.weight": f"model.layers.{index}.input_layernorm.weight",
        f"MistralTransformer.MistralTranformerLayer_{index+1}.Residual_1.MistralAttention.Parallel.Linear_1.weight": f"model.layers.{index}.self_attn.q_proj.weight",
        f"MistralTransformer.MistralTranformerLayer_{index+1}.Residual_1.MistralAttention.Parallel.Linear_2.weight": f"model.layers.{index}.self_attn.k_proj.weight",
        f"MistralTransformer.MistralTranformerLayer_{index+1}.Residual_1.MistralAttention.Parallel.Linear_3.weight": f"model.layers.{index}.self_attn.v_proj.weight",
        f"MistralTransformer.MistralTranformerLayer_{index+1}.Residual_1.MistralAttention.Linear.weight": f"model.layers.{index}.self_attn.o_proj.weight",
        # MLP
        f"MistralTransformer.MistralTranformerLayer_{index+1}.Residual_2.MistralRMSNorm.weight": f"model.layers.{index}.post_attention_layernorm.weight",
        f"MistralTransformer.MistralTranformerLayer_{index+1}.Residual_2.FeedForward.Parallel.Chain.Linear.weight": f"model.layers.{index}.mlp.gate_proj.weight",
        f"MistralTransformer.MistralTranformerLayer_{index+1}.Residual_2.FeedForward.Parallel.Linear.weight": f"model.layers.{index}.mlp.up_proj.weight",
        f"MistralTransformer.MistralTranformerLayer_{index+1}.Residual_2.FeedForward.Linear.weight": f"model.layers.{index}.mlp.down_proj.weight",
    }


def get_mistral_mapping(n_layers: int = 32):
    """Generate a mapping dictionary for transferring parameters from a Mistral HuggingFace model.
    """
    mapping: dict[str, str] = {}

    mapping["Embedding.weight"] = "model.embed_tokens.weight"

    for index in range(n_layers):
        mapping.update(get_tranformer_layer_mapping(index))

    mapping["MistralRMSNorm.weight"] = "model.norm.weight"
    mapping["Linear.weight"] = "lm_head.weight"

    return mapping


def check_shapes(
    source_model_state_dict: dict[str, torch.Tensor],
    target_model_state_dict: dict[str, torch.Tensor],
    mapping: dict[str, str],
):

    for target_layer_name, source_layer_name in mapping.items():
        assert (
            target_model_state_dict[target_layer_name].shape
            == source_model_state_dict[source_layer_name].shape
        )

    print("Shapes aligned !")


@no_grad()
def convert(source_path: str, hf_token: str) -> dict[str, torch.Tensor]:
    def _convert_state_dict(
        source_state_dict: dict[str, torch.Tensor],
        target_state_dict: dict[str, torch.Tensor],
        state_dict_mapping: dict[str, str],
    ):
        converted_state_dict: dict[str, torch.Tensor] = {}
        for target_key in target_state_dict:
            source_key = state_dict_mapping[target_key]
            converted_state_dict[target_key] = source_state_dict[source_key]
        return converted_state_dict

    mistral_src: nn.Module = MistralForCausalLM.from_pretrained(  # type: ignore
        pretrained_model_name_or_path=source_path,
        low_cpu_mem_usage=False,
        token=hf_token
    )

    mistral_refiners = Mistral7b(device="meta")
    mapping = get_mistral_mapping()

    check_shapes(mistral_src.state_dict(), mistral_refiners.state_dict(), mapping)

    state_dict = _convert_state_dict(
        source_state_dict=mistral_src.state_dict(),
        target_state_dict=mistral_refiners.state_dict(),
        state_dict_mapping=mapping,
    )

    return state_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Mistral 7b 0.1v model to a Refiners Mistral 7b 0.1v model"
    )
    parser.add_argument(
        "--from",
        type=str,
        dest="source_path",
        default="mistralai/Mistral-7B-v0.1",
        help=(
            "Can be a path to a .bin, a .safetensors file, or a model identifier from Hugging Face Hub. Defaults to"
            " mistralai/Mistral-7B-v0.1"
        ),
    )
    parser.add_argument(
        "--to",
        type=str,
        dest="output_path",
        required=False,
        default=None,
        help=(
            "Output path (.safetensors) for converted model. If not provided, the output path will be the same as the"
            " source path."
        ),
    )

    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help=(
            "The authentication token required for accessing the HuggingFace model."
        ),
    )

    parser.add_argument("--half", action="store_true", dest="half")
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = f"{Path(args.source_path).stem}-mistral_7b_v0.1.safetensors"
    state_dict = convert(source_path=args.source_path, hf_token=args.token)

    if args.half:
        state_dict = {k: v.half() for k, v in state_dict.items()}

    save_to_safetensors(path=args.output_path, tensors=state_dict)


if __name__ == "__main__":
    main()
