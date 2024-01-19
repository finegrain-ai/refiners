import argparse
from pathlib import Path
from typing import Any

import torch

from refiners.fluxion.utils import load_tensors, save_to_safetensors
from refiners.foundationals.latent_diffusion import SD1IPAdapter, SD1UNet, SDXLIPAdapter, SDXLUNet

# Running:
#
#     from diffusers import UNet2DConditionModel
#     unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
#     for k in unet.attn_processors.keys():
#         print(k)
#
# Gives:
#
#     down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor
#     down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor
#     ...
#     down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor
#     up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor
#     up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor
#     ...
#     up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor
#     mid_block.attentions.0.transformer_blocks.0.attn1.processor
#     mid_block.attentions.0.transformer_blocks.0.attn2.processor
#
# With attn1=self-attention and attn2=cross-attention, and middle block in last position. So in terms of increasing
# indices:
#
#     DownBlocks  -> [1, 3, 5, 7, 9, 11]
#     MiddleBlock -> [31]
#     UpBlocks    -> [13, 15, 17, 19, 21, 23, 25, 27, 29]
#
# Same for SDXL with more layers (70 cross-attentions vs. 16)
CROSS_ATTN_MAPPING: dict[str, list[int]] = {
    "sd15": list(range(1, 12, 2)) + [31] + list(range(13, 30, 2)),
    "sdxl": list(range(1, 48, 2)) + list(range(121, 140, 2)) + list(range(49, 120, 2)),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Converts a IP-Adapter diffusers model to refiners.")
    parser.add_argument(
        "--from",
        type=str,
        required=True,
        dest="source_path",
        help="Path to the source model. (e.g.: 'ip-adapter_sd15.bin').",
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
    parser.add_argument("--verbose", action="store_true", dest="verbose")
    parser.add_argument("--half", action="store_true", dest="half")
    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = f"{Path(args.source_path).stem}.safetensors"

    weights: dict[str, Any] = load_tensors(args.source_path, device="cpu")
    assert isinstance(weights, dict)
    assert sorted(weights.keys()) == ["image_proj", "ip_adapter"]

    fine_grained = "latents" in weights["image_proj"]  # aka IP-Adapter plus

    match len(weights["ip_adapter"]):
        case 32:
            ip_adapter = SD1IPAdapter(target=SD1UNet(in_channels=4), fine_grained=fine_grained)
            cross_attn_mapping = CROSS_ATTN_MAPPING["sd15"]
        case 140:
            ip_adapter = SDXLIPAdapter(target=SDXLUNet(in_channels=4), fine_grained=fine_grained)
            cross_attn_mapping = CROSS_ATTN_MAPPING["sdxl"]
        case _:
            raise ValueError("Unexpected number of keys in input checkpoint")

    # Manual conversion to avoid any runtime dependency on IP-Adapter[1] custom classes
    # [1]: https://github.com/tencent-ailab/IP-Adapter

    state_dict: dict[str, torch.Tensor] = {}

    image_proj_weights = weights["image_proj"]
    image_proj_state_dict: dict[str, torch.Tensor]

    if fine_grained:
        w = image_proj_weights
        image_proj_state_dict = {
            "LatentsToken.Parameter.weight": w["latents"].squeeze(0),  # drop batch dim = 1
            "Linear_1.weight": w["proj_in.weight"],
            "Linear_1.bias": w["proj_in.bias"],
            "Linear_2.weight": w["proj_out.weight"],
            "Linear_2.bias": w["proj_out.bias"],
            "LayerNorm.weight": w["norm_out.weight"],
            "LayerNorm.bias": w["norm_out.bias"],
        }
        for i in range(4):
            t_pfx, s_pfx = f"Transformer.TransformerLayer_{i+1}.Residual_", f"layers.{i}."
            image_proj_state_dict.update(
                {
                    f"{t_pfx}1.PerceiverAttention.Distribute.LayerNorm_1.weight": w[f"{s_pfx}0.norm1.weight"],
                    f"{t_pfx}1.PerceiverAttention.Distribute.LayerNorm_1.bias": w[f"{s_pfx}0.norm1.bias"],
                    f"{t_pfx}1.PerceiverAttention.Distribute.LayerNorm_2.weight": w[f"{s_pfx}0.norm2.weight"],
                    f"{t_pfx}1.PerceiverAttention.Distribute.LayerNorm_2.bias": w[f"{s_pfx}0.norm2.bias"],
                    f"{t_pfx}1.PerceiverAttention.Parallel.Chain_2.Linear.weight": w[f"{s_pfx}0.to_q.weight"],
                    f"{t_pfx}1.PerceiverAttention.Parallel.Chain_1.Linear.weight": w[f"{s_pfx}0.to_kv.weight"],
                    f"{t_pfx}1.PerceiverAttention.Linear.weight": w[f"{s_pfx}0.to_out.weight"],
                    f"{t_pfx}2.LayerNorm.weight": w[f"{s_pfx}1.0.weight"],
                    f"{t_pfx}2.LayerNorm.bias": w[f"{s_pfx}1.0.bias"],
                    f"{t_pfx}2.FeedForward.Linear_1.weight": w[f"{s_pfx}1.1.weight"],
                    f"{t_pfx}2.FeedForward.Linear_2.weight": w[f"{s_pfx}1.3.weight"],
                }
            )
    else:
        image_proj_state_dict = {
            "Linear.weight": image_proj_weights["proj.weight"],
            "Linear.bias": image_proj_weights["proj.bias"],
            "LayerNorm.weight": image_proj_weights["norm.weight"],
            "LayerNorm.bias": image_proj_weights["norm.bias"],
        }
    ip_adapter.image_proj.load_state_dict(state_dict=image_proj_state_dict)

    for k, v in image_proj_state_dict.items():
        state_dict[f"image_proj.{k}"] = v

    ip_adapter_weights: dict[str, torch.Tensor] = weights["ip_adapter"]
    assert len(ip_adapter.sub_adapters) == len(ip_adapter_weights.keys()) // 2

    for i, _ in enumerate(ip_adapter.sub_adapters):
        cross_attn_index = cross_attn_mapping[i]
        k_ip = f"{cross_attn_index}.to_k_ip.weight"
        v_ip = f"{cross_attn_index}.to_v_ip.weight"

        # the name of the key is not checked at runtime, so we keep the original name
        state_dict[f"ip_adapter.{i:03d}.to_k_ip.weight"] = ip_adapter_weights[k_ip]
        state_dict[f"ip_adapter.{i:03d}.to_v_ip.weight"] = ip_adapter_weights[v_ip]

    if args.half:
        state_dict = {key: value.half() for key, value in state_dict.items()}
    if args.output_path is None:
        args.output_path = f"{Path(args.source_path).stem}.safetensors"
    save_to_safetensors(path=args.output_path, tensors=state_dict)


if __name__ == "__main__":
    main()
