import argparse
from functools import partial
from torch import Tensor
from refiners.fluxion.utils import (
    load_from_safetensors,
    load_metadata_from_safetensors,
    save_to_safetensors,
)
from convert_diffusers_unet import setup_converter as convert_unet, Args as UnetConversionArgs
from convert_transformers_clip_text_model import (
    setup_converter as convert_text_encoder,
    Args as TextEncoderConversionArgs,
)
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
from refiners.foundationals.latent_diffusion import SD1UNet
from refiners.foundationals.latent_diffusion.lora import LoraTarget
import refiners.fluxion.layers as fl


def get_unet_mapping(source_path: str) -> dict[str, str]:
    args = UnetConversionArgs(source_path=source_path, verbose=False)
    return convert_unet(args=args).get_mapping()


def get_text_encoder_mapping(source_path: str) -> dict[str, str]:
    args = TextEncoderConversionArgs(source_path=source_path, subfolder="text_encoder", verbose=False)
    return convert_text_encoder(
        args=args,
    ).get_mapping()


def main() -> None:
    parser = argparse.ArgumentParser(description="Converts a refiner's LoRA weights to SD-WebUI's LoRA weights")
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        required=True,
        help="Path to the input file with refiner's LoRA weights (safetensors format)",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="sdwebui_loras.safetensors",
        help="Path to the output file with sd-webui's LoRA weights (safetensors format)",
    )
    parser.add_argument(
        "--sd15",
        type=str,
        required=False,
        default="runwayml/stable-diffusion-v1-5",
        help="Path (preferred) or repository ID of Stable Diffusion 1.5 model (Hugging Face diffusers format)",
    )
    args = parser.parse_args()
    metadata = load_metadata_from_safetensors(path=args.input_file)
    assert metadata is not None, f"Could not load metadata from {args.input_file}"
    tensors = load_from_safetensors(path=args.input_file)

    state_dict: dict[str, Tensor] = {}

    for meta_key, meta_value in metadata.items():
        match meta_key:
            case "unet_targets":
                model = SD1UNet(in_channels=4)
                create_mapping = partial(get_unet_mapping, source_path=args.sd15)
                key_prefix = "unet."
                lora_prefix = "lora_unet_"
            case "text_encoder_targets":
                model = CLIPTextEncoderL()
                create_mapping = partial(get_text_encoder_mapping, source_path=args.sd15)
                key_prefix = "text_encoder."
                lora_prefix = "lora_te_"
            case "lda_targets":
                raise ValueError("SD-WebUI does not support LoRA for the auto-encoder")
            case _:
                raise ValueError(f"Unexpected key in checkpoint metadata: {meta_key}")

        submodule_to_key: dict[fl.Module, str] = {}
        for name, submodule in model.named_modules():
            submodule_to_key[submodule] = name

        # SD-WebUI expects LoRA state dicts with keys derived from the diffusers format, e.g.:
        #
        #     lora_unet_down_blocks_0_attentions_0_proj_in.alpha
        #     lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight
        #     lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight
        #     ...
        #
        # Internally SD-WebUI has some logic[1] to convert such keys into the CompVis format. See
        # `convert_diffusers_name_to_compvis` for more details.
        #
        # [1]: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/394ffa7/extensions-builtin/Lora/lora.py#L158-L225

        refiners_to_diffusers = create_mapping()
        assert refiners_to_diffusers is not None

        # Compute the corresponding diffusers' keys where LoRA layers must be applied
        lora_injection_points: list[str] = [
            refiners_to_diffusers[submodule_to_key[linear]]
            for target in [LoraTarget(t) for t in meta_value.split(sep=",")]
            for layer in model.layers(layer_type=target.get_class())
            for linear in layer.layers(layer_type=fl.Linear)
        ]

        lora_weights = [w for w in [tensors[k] for k in sorted(tensors) if k.startswith(key_prefix)]]
        assert len(lora_injection_points) == len(lora_weights) // 2

        # Map LoRA weights to each key using SD-WebUI conventions (proper prefix and suffix, underscores)
        for i, diffusers_key in enumerate(iterable=lora_injection_points):
            lora_key = lora_prefix + diffusers_key.replace(".", "_")
            # Note: no ".alpha" weights (those are used to scale the LoRA by alpha/rank). Refiners uses a scale = 1.0
            # by default (see `lora_calc_updown` in SD-WebUI for more details)
            state_dict[lora_key + ".lora_up.weight"] = lora_weights[2 * i]
            state_dict[lora_key + ".lora_down.weight"] = lora_weights[2 * i + 1]

    assert state_dict
    save_to_safetensors(path=args.output_file, tensors=state_dict)


if __name__ == "__main__":
    main()
