from refiners.fluxion.utils import (
    load_from_safetensors,
    load_metadata_from_safetensors,
    save_to_safetensors,
)
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
from refiners.foundationals.clip.tokenizer import CLIPTokenizer
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
from refiners.foundationals.latent_diffusion.lora import LoraTarget
from refiners.fluxion.layers.module import Module
import refiners.fluxion.layers as fl
from refiners.fluxion.utils import create_state_dict_mapping

import torch

from diffusers import DiffusionPipeline  # type: ignore
from diffusers.models.unet_2d_condition import UNet2DConditionModel  # type: ignore
from transformers.models.clip.modeling_clip import CLIPTextModel  # type: ignore


@torch.no_grad()
def create_unet_mapping(src_model: UNet2DConditionModel, dst_model: SD1UNet) -> dict[str, str] | None:
    x = torch.randn(1, 4, 32, 32)
    timestep = torch.tensor(data=[0])
    clip_text_embeddings = torch.randn(1, 77, 768)

    src_args = (x, timestep, clip_text_embeddings)
    dst_model.set_timestep(timestep=timestep)
    dst_model.set_clip_text_embedding(clip_text_embedding=clip_text_embeddings)
    dst_args = (x,)

    return create_state_dict_mapping(source_model=src_model, target_model=dst_model, source_args=src_args, target_args=dst_args)  # type: ignore


@torch.no_grad()
def create_text_encoder_mapping(src_model: CLIPTextModel, dst_model: CLIPTextEncoderL) -> dict[str, str] | None:
    tokenizer = dst_model.find(layer_type=CLIPTokenizer)
    assert tokenizer is not None, "Could not find tokenizer"
    tokens = tokenizer("Nice cat")
    return create_state_dict_mapping(source_model=src_model, target_model=dst_model, source_args=[tokens], target_args=["Nice cat"])  # type: ignore


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
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
        required=True,
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
    assert metadata is not None
    tensors = load_from_safetensors(path=args.input_file)

    diffusers_sd = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path=args.sd15)  # type: ignore

    state_dict: dict[str, torch.Tensor] = {}

    for meta_key, meta_value in metadata.items():
        match meta_key:
            case "unet_targets":
                src_model = diffusers_sd.unet  # type: ignore
                dst_model = SD1UNet(in_channels=4, clip_embedding_dim=768)
                create_mapping = create_unet_mapping
                key_prefix = "unet."
                lora_prefix = "lora_unet_"
            case "text_encoder_targets":
                src_model = diffusers_sd.text_encoder  # type: ignore
                dst_model = CLIPTextEncoderL()
                create_mapping = create_text_encoder_mapping
                key_prefix = "text_encoder."
                lora_prefix = "lora_te_"
            case "lda_targets":
                raise ValueError("SD-WebUI does not support LoRA for the auto-encoder")
            case _:
                raise ValueError(f"Unexpected key in checkpoint metadata: {meta_key}")

        submodule_to_key: dict[Module, str] = {}
        for name, submodule in dst_model.named_modules():
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

        refiners_to_diffusers = create_mapping(src_model, dst_model)  # type: ignore
        assert refiners_to_diffusers is not None

        # Compute the corresponding diffusers' keys where LoRA layers must be applied
        lora_injection_points: list[str] = [
            refiners_to_diffusers[submodule_to_key[linear]]
            for target in [LoraTarget(t) for t in meta_value.split(sep=",")]
            for layer in dst_model.layers(layer_type=target.get_class())
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
