import argparse
from typing import Callable, cast

import torch
import torch.nn as nn
from torch import Tensor
from transformers import VitMatteForImageMatting  # type: ignore
from transformers.models.vitdet.modeling_vitdet import VitDetLayerNorm  # type: ignore

import refiners.fluxion.layers as fl
from refiners.fluxion.model_converter import ModelConverter
from refiners.fluxion.utils import manual_seed, save_to_safetensors
from refiners.foundationals.vit_matte import DetailCapture, ViT, ViTMatteH


class VitMatte(nn.Module):
    vit_backbone: nn.Module
    decoder: nn.Module


VitMatteForImageMatting = cast(Callable[[], VitMatte], VitMatteForImageMatting)


assert issubclass(VitDetLayerNorm, nn.Module)
custom_layers = {VitDetLayerNorm: fl.LayerNorm2d}


class Args(argparse.Namespace):
    source_path: str
    output_path: str
    half: bool
    verbose: bool


def convert_vit(vit: nn.Module) -> dict[str, Tensor]:
    manual_seed(seed=0)
    refiners_vit_matte_h = ViTMatteH().ensure_find(ViT)

    converter = ModelConverter(
        source_model=vit,
        target_model=refiners_vit_matte_h,
        custom_layer_mapping=custom_layers,  # type: ignore
        verbose=True,
    )
    converter.skip_init_check = True

    x = torch.randn((1, 4, 512, 512))
    mapping = converter.map_state_dicts(source_args=(x,))
    assert mapping

    mapping["PositionalEncoder.Parameter.weight"] = "embeddings.position_embeddings"
    target_state_dict = refiners_vit_matte_h.state_dict()
    del target_state_dict["PositionalEncoder.Parameter.weight"]

    source_state_dict = vit.state_dict()
    pos_embed = source_state_dict["embeddings.position_embeddings"]
    del source_state_dict["embeddings.position_embeddings"]

    target_rel_keys = [
        (
            f"Transformer.TransformerLayer_{i}.Residual_1.FusedSelfAttention.RelativePositionAttention.horizontal_embedding",
            f"Transformer.TransformerLayer_{i}.Residual_1.FusedSelfAttention.RelativePositionAttention.vertical_embedding",
        )
        for i in range(1, 13)
    ]
    source_rel_keys = [
        (f"encoder.layer.{i}.attention.rel_pos_w", f"encoder.layer.{i}.attention.rel_pos_h") for i in range(12)
    ]

    rel_items: dict[str, Tensor] = {}

    for (key_w, key_h), (target_key_w, target_key_h) in zip(source_rel_keys, target_rel_keys):
        rel_items[target_key_w] = source_state_dict[key_w]
        rel_items[target_key_h] = source_state_dict[key_h]
        del source_state_dict[key_w]
        del source_state_dict[key_h]
        del target_state_dict[target_key_w]
        del target_state_dict[target_key_h]

    converted_source = converter._convert_state_dict(  # pyright: ignore[reportPrivateUsage]
        source_state_dict=source_state_dict, target_state_dict=target_state_dict, state_dict_mapping=mapping
    )

    converted_source["PositionalEncoder.Parameter.weight"] = pos_embed.squeeze()  # type: ignore
    converted_source.update(rel_items)

    refiners_vit_matte_h.load_state_dict(state_dict=converted_source)
    assert converter.compare_models((x,), threshold=1e-2)

    return converted_source


def convert_decoder(decoder: nn.Module) -> dict[str, Tensor]:
    manual_seed(seed=0)
    features = torch.randn((1, 384, 32, 32))
    images = torch.randn((1, 4, 512, 512))
    d_capture = ViTMatteH().ensure_find(DetailCapture)
    d_capture.set_context("detail_capture", {"images": images})

    converter = ModelConverter(
        source_model=decoder,  # type: ignore
        target_model=d_capture,
        verbose=True,
    )
    converter.run(source_args=(features, images), target_args=(features,))

    assert converter.compare_models(source_args=(features, images), target_args=(features,), threshold=1e-3)

    return converter.get_state_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Converts a ViT Matte model to a Refiners VitMatte model")
    parser.add_argument(
        "--from",
        type=str,
        dest="pretrained_model",
        default="hustvl/vitmatte-small-composition-1k",
        # required=True,
        help="Path to the refiners VitMatte weights",
    )
    parser.add_argument(
        "--to",
        type=str,
        dest="output_path",
        default="vitmatte_h.safetensors",
        help="Output path for converted model (as safetensors).",
    )
    parser.add_argument("--half", action="store_true", default=False, help="Convert to half precision. Default: False")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Prints additional information during conversion. Default: False",
    )
    args = parser.parse_args(namespace=Args())

    vitm_h = VitMatteForImageMatting.from_pretrained(args.pretrained_model)  # type: ignore

    vit_state_dict = convert_vit(vit=vitm_h.backbone)  # type: ignore
    decoder_state_dict = convert_decoder(decoder=vitm_h.decoder)  # type: ignore

    output_state_dict = {
        **{".".join(("ViT", key)): value for key, value in vit_state_dict.items()},
        **{".".join(("DetailCapture", key)): value for key, value in decoder_state_dict.items()},
    }

    myViTMatte = ViTMatteH()
    myViTMatte.load_state_dict(output_state_dict)

    images = torch.randn((1, 4, 512, 512))

    res_target = myViTMatte(images)
    res_source = vitm_h(images)

    assert torch.allclose(res_target, res_source.alphas, 0.0001)

    if args.half:
        output_state_dict = {key: value.half() for key, value in output_state_dict.items()}

    save_to_safetensors(path=args.output_path, tensors=output_state_dict)


if __name__ == "__main__":
    main()
