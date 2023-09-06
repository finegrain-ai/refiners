from pathlib import Path
from typing import Any
import argparse

import torch

from refiners.foundationals.latent_diffusion import SD1UNet
from refiners.foundationals.latent_diffusion.stable_diffusion_1 import SD1IPAdapter
from refiners.fluxion.utils import save_to_safetensors


def main() -> None:
    parser = argparse.ArgumentParser(description="Converts a IP-Adapter diffusers model to refiners.")
    parser.add_argument(
        "--from",
        type=str,
        dest="source_path",
        default="ip-adapter_sd15.bin",
        help="Path to the source model. (default: 'ip-adapter_sd15.bin').",
    )
    parser.add_argument(
        "--to",
        type=str,
        dest="output_path",
        default="ip-adapter_sd15.safetensors",
        help="Path to save the converted model. (default: 'ip-adapter_sd15.safetensors').",
    )
    parser.add_argument("--verbose", action="store_true", dest="verbose")
    parser.add_argument("--half", action="store_true", dest="half")
    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = f"{Path(args.source_path).stem}.safetensors"

    weights: dict[str, Any] = torch.load(f=args.source_path, map_location="cpu")  # type: ignore
    assert isinstance(weights, dict)
    assert sorted(weights.keys()) == ["image_proj", "ip_adapter"]

    unet = SD1UNet(in_channels=4)

    ip_adapter = SD1IPAdapter(target=unet)

    # Manual conversion to avoid any runtime dependency on IP-Adapter[1] custom classes
    # [1]: https://github.com/tencent-ailab/IP-Adapter

    state_dict: dict[str, torch.Tensor] = {}

    image_proj_weights = weights["image_proj"]
    image_proj_state_dict: dict[str, torch.Tensor] = {
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
    cross_attn_mapping: list[int] = [1, 3, 5, 7, 9, 11, 31, 13, 15, 17, 19, 21, 23, 25, 27, 29]

    for i, cross_attn in enumerate(ip_adapter.sub_adapters):
        cross_attn_index = cross_attn_mapping[i]
        k_ip = f"{cross_attn_index}.to_k_ip.weight"
        v_ip = f"{cross_attn_index}.to_v_ip.weight"

        # Ignore Wq, Wk, Wv and Proj (hence strict=False): at runtime, they will be part of the UNet original weights
        cross_attn_state_dict: dict[str, Any] = {
            cross_attn.get_parameter_name("wk_prime"): ip_adapter_weights[k_ip],
            cross_attn.get_parameter_name("wv_prime"): ip_adapter_weights[v_ip],
        }
        cross_attn.load_state_dict(state_dict=cross_attn_state_dict, strict=False)

        for k, v in cross_attn_state_dict.items():
            state_dict[f"ip_adapter.{i:03d}.{k}"] = v

    if args.half:
        state_dict = {key: value.half() for key, value in state_dict.items()}
    save_to_safetensors(path=args.output_path, tensors=state_dict)


if __name__ == "__main__":
    main()
