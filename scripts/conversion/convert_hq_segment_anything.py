import argparse

from torch import Tensor

from refiners.fluxion.utils import load_tensors, save_to_safetensors


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HQ SAM model to Refiners state_dict format")
    parser.add_argument(
        "--from",
        type=str,
        dest="source_path",
        required=True,
        default="sam_hq_vit_h.pth",
        help="Path to the source model checkpoint.",
    )
    parser.add_argument(
        "--to",
        type=str,
        dest="output_path",
        required=True,
        default="refiners_sam_hq_vit_h.safetensors",
        help="Path to save the converted model in Refiners format.",
    )
    args = parser.parse_args()

    source_state_dict = load_tensors(args.source_path)

    state_dict: dict[str, Tensor] = {}

    for suffix in ["weight", "bias"]:
        state_dict[f"HQFeatures.CompressViTFeat.ConvTranspose2d_1.{suffix}"] = source_state_dict[
            f"mask_decoder.compress_vit_feat.0.{suffix}"
        ]
        state_dict[f"HQFeatures.EmbeddingEncoder.ConvTranspose2d_1.{suffix}"] = source_state_dict[
            f"mask_decoder.embedding_encoder.0.{suffix}"
        ]
        state_dict[f"EmbeddingMaskfeature.Conv2d_1.{suffix}"] = source_state_dict[
            f"mask_decoder.embedding_maskfeature.0.{suffix}"
        ]

        state_dict[f"HQFeatures.CompressViTFeat.LayerNorm2d.{suffix}"] = source_state_dict[
            f"mask_decoder.compress_vit_feat.1.{suffix}"
        ]
        state_dict[f"HQFeatures.EmbeddingEncoder.LayerNorm2d.{suffix}"] = source_state_dict[
            f"mask_decoder.embedding_encoder.1.{suffix}"
        ]
        state_dict[f"EmbeddingMaskfeature.LayerNorm2d.{suffix}"] = source_state_dict[
            f"mask_decoder.embedding_maskfeature.1.{suffix}"
        ]

        state_dict[f"HQFeatures.CompressViTFeat.ConvTranspose2d_2.{suffix}"] = source_state_dict[
            f"mask_decoder.compress_vit_feat.3.{suffix}"
        ]
        state_dict[f"HQFeatures.EmbeddingEncoder.ConvTranspose2d_2.{suffix}"] = source_state_dict[
            f"mask_decoder.embedding_encoder.3.{suffix}"
        ]
        state_dict[f"EmbeddingMaskfeature.Conv2d_2.{suffix}"] = source_state_dict[
            f"mask_decoder.embedding_maskfeature.3.{suffix}"
        ]

    state_dict = {f"Chain.HQSAMMaskPrediction.Chain.DenseEmbeddingUpscalingHQ.{k}": v for k, v in state_dict.items()}

    # HQ Token
    state_dict["MaskDecoderTokensExtender.hq_token.weight"] = source_state_dict["mask_decoder.hf_token.weight"]

    # HQ MLP
    for i in range(3):
        state_dict[f"Chain.HQSAMMaskPrediction.HQTokenMLP.MultiLinear.Linear_{i+1}.weight"] = source_state_dict[
            f"mask_decoder.hf_mlp.layers.{i}.weight"
        ]
        state_dict[f"Chain.HQSAMMaskPrediction.HQTokenMLP.MultiLinear.Linear_{i+1}.bias"] = source_state_dict[
            f"mask_decoder.hf_mlp.layers.{i}.bias"
        ]

    save_to_safetensors(path=args.output_path, tensors=state_dict)


if __name__ == "__main__":
    main()
