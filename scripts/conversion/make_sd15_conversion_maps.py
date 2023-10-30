
from pathlib import Path

from refiners.fluxion.conversion.diffusers_autoencoder_kl import setup_autoencoder_kl_converter
from refiners.fluxion.conversion.diffusers_unet import setup_diffusers_unet_converter
from refiners.fluxion.conversion.transformers_clip_text_model import setup_transformers_clip_text_converter


def make_maps():
    """Create and save the conversion maps needed to load the SD1.5 diffusers models"""

    repo = "runwayml/stable-diffusion-v1-5"

    project_root_path = Path(__file__).parent.parent.parent
    mapping_path = project_root_path / "src" / "refiners" / "fluxion" / "conversion_maps"

    converter = setup_transformers_clip_text_converter(source_path=repo, subfolder="text_encoder", verbose=False)
    converter.save_conversion_mapping(path=mapping_path / "transformers_clip_text_model_sd15.json")

    converter = setup_autoencoder_kl_converter(source_path=repo, subfolder="vae")
    # handle deprecated attention block modules
    # https://github.com/huggingface/diffusers/blob/9135e54e768a59ddcf8ad18818d2ffe69ea3a32a/src/diffusers/models/modeling_utils.py#L1121-L1125C13
    vae_aliases = {
        "encoder.mid_block.attentions.0.value": "encoder.mid_block.attentions.0.to_v",
        "decoder.mid_block.attentions.0.value": "decoder.mid_block.attentions.0.to_v",
        "decoder.mid_block.attentions.0.proj_attn": "decoder.mid_block.attentions.0.to_out.0",
        "encoder.mid_block.attentions.0.proj_attn": "encoder.mid_block.attentions.0.to_out.0",
        "encoder.mid_block.attentions.0.key": "encoder.mid_block.attentions.0.to_k",
        "decoder.mid_block.attentions.0.key": "decoder.mid_block.attentions.0.to_k",
        "decoder.mid_block.attentions.0.query": "decoder.mid_block.attentions.0.to_q",
        "encoder.mid_block.attentions.0.query": "encoder.mid_block.attentions.0.to_q",
    }

    converter.save_conversion_mapping(path=mapping_path / "diffusers_sd15_vae.json", aliases=vae_aliases)

    converter = setup_diffusers_unet_converter(source_path=repo, subfolder="unet")
    converter.save_conversion_mapping(path=mapping_path / "diffusers_sd15_unet.json")


if __name__ == "__main__":
    make_maps()
