import argparse
import types
from typing import Any, Callable, cast

import torch
import torch.nn as nn
from segment_anything import build_sam_vit_h  # type: ignore
from segment_anything.modeling.common import LayerNorm2d  # type: ignore
from torch import Tensor

import refiners.fluxion.layers as fl
from refiners.fluxion.model_converter import ModelConverter
from refiners.fluxion.utils import load_tensors, manual_seed, save_to_safetensors
from refiners.foundationals.segment_anything.image_encoder import PositionalEncoder, SAMViTH
from refiners.foundationals.segment_anything.mask_decoder import MaskDecoder
from refiners.foundationals.segment_anything.prompt_encoder import MaskEncoder, PointEncoder


class FacebookSAM(nn.Module):
    image_encoder: nn.Module
    prompt_encoder: nn.Module
    mask_decoder: nn.Module


build_sam_vit_h = cast(Callable[[], FacebookSAM], build_sam_vit_h)


assert issubclass(LayerNorm2d, nn.Module)
custom_layers = {LayerNorm2d: fl.LayerNorm2d}


class Args(argparse.Namespace):
    source_path: str
    output_path: str
    half: bool
    verbose: bool


def convert_mask_encoder(prompt_encoder: nn.Module) -> dict[str, Tensor]:
    manual_seed(seed=0)
    refiners_mask_encoder = MaskEncoder()

    converter = ModelConverter(
        source_model=prompt_encoder.mask_downscaling,
        target_model=refiners_mask_encoder,
        custom_layer_mapping=custom_layers,  # type: ignore
    )

    x = torch.randn(1, 256, 256)
    mapping = converter.map_state_dicts(source_args=(x,))
    assert mapping

    source_state_dict = prompt_encoder.mask_downscaling.state_dict()
    target_state_dict = refiners_mask_encoder.state_dict()

    # Mapping handled manually (see below) because nn.Parameter is a special case
    del target_state_dict["no_mask_embedding"]

    converted_source = converter._convert_state_dict(  # pyright: ignore[reportPrivateUsage]
        source_state_dict=source_state_dict, target_state_dict=target_state_dict, state_dict_mapping=mapping
    )

    state_dict: dict[str, Tensor] = {
        "no_mask_embedding": nn.Parameter(data=prompt_encoder.no_mask_embed.weight.clone()),  # type: ignore
    }

    state_dict.update(converted_source)

    refiners_mask_encoder.load_state_dict(state_dict=state_dict)

    return state_dict


def convert_point_encoder(prompt_encoder: nn.Module) -> dict[str, Tensor]:
    manual_seed(seed=0)
    point_embeddings: list[Tensor] = [pe.weight for pe in prompt_encoder.point_embeddings] + [
        prompt_encoder.not_a_point_embed.weight
    ]  # type: ignore
    pe = prompt_encoder.pe_layer.positional_encoding_gaussian_matrix  # type: ignore
    assert isinstance(pe, Tensor)
    state_dict: dict[str, Tensor] = {
        "Residual.PointTypeEmbedding.weight": nn.Parameter(data=torch.cat(tensors=point_embeddings, dim=0)),
        "CoordinateEncoder.Linear.weight": nn.Parameter(data=pe.T.contiguous()),
    }

    refiners_prompt_encoder = PointEncoder()
    refiners_prompt_encoder.load_state_dict(state_dict=state_dict)

    return state_dict


def convert_vit(vit: nn.Module) -> dict[str, Tensor]:
    manual_seed(seed=0)
    refiners_sam_vit_h = SAMViTH()

    converter = ModelConverter(
        source_model=vit,
        target_model=refiners_sam_vit_h,
        custom_layer_mapping=custom_layers,  # type: ignore
    )
    converter.skip_init_check = True

    x = torch.randn(1, 3, 1024, 1024)
    mapping = converter.map_state_dicts(source_args=(x,))
    assert mapping

    mapping["PositionalEncoder.Parameter.weight"] = "pos_embed"

    target_state_dict = refiners_sam_vit_h.state_dict()
    del target_state_dict["PositionalEncoder.Parameter.weight"]

    source_state_dict = vit.state_dict()
    pos_embed = source_state_dict["pos_embed"]
    del source_state_dict["pos_embed"]

    target_rel_keys = [
        (
            f"Transformer.TransformerLayer_{i}.Residual_1.FusedSelfAttention.RelativePositionAttention.horizontal_embedding",
            f"Transformer.TransformerLayer_{i}.Residual_1.FusedSelfAttention.RelativePositionAttention.vertical_embedding",
        )
        for i in range(1, 33)
    ]
    source_rel_keys = [(f"blocks.{i}.attn.rel_pos_w", f"blocks.{i}.attn.rel_pos_h") for i in range(32)]

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

    positional_encoder = refiners_sam_vit_h.layer("PositionalEncoder", PositionalEncoder)
    embed = pos_embed.reshape_as(positional_encoder.layer("Parameter", fl.Parameter).weight)
    converted_source["PositionalEncoder.Parameter.weight"] = embed  # type: ignore
    converted_source.update(rel_items)

    refiners_sam_vit_h.load_state_dict(state_dict=converted_source)
    assert converter.compare_models((x,), threshold=1e-2)

    return converted_source


def convert_mask_decoder(mask_decoder: nn.Module) -> dict[str, Tensor]:
    manual_seed(seed=0)

    refiners_mask_decoder = MaskDecoder()

    image_embedding = torch.randn(1, 256, 64, 64)
    dense_positional_embedding = torch.randn(1, 256, 64, 64)
    point_embedding = torch.randn(1, 3, 256)
    mask_embedding = torch.randn(1, 256, 64, 64)

    from segment_anything.modeling.common import LayerNorm2d  # type: ignore

    import refiners.fluxion.layers as fl

    assert issubclass(LayerNorm2d, nn.Module)
    custom_layers = {LayerNorm2d: fl.LayerNorm2d}

    converter = ModelConverter(
        source_model=mask_decoder,
        target_model=refiners_mask_decoder,
        custom_layer_mapping=custom_layers,  # type: ignore
    )

    inputs = {
        "image_embeddings": image_embedding,
        "image_pe": dense_positional_embedding,
        "sparse_prompt_embeddings": point_embedding,
        "dense_prompt_embeddings": mask_embedding,
        "multimask_output": True,
    }

    refiners_mask_decoder.set_image_embedding(image_embedding)
    refiners_mask_decoder.set_point_embedding(point_embedding)
    refiners_mask_decoder.set_mask_embedding(mask_embedding)
    refiners_mask_decoder.set_dense_positional_embedding(dense_positional_embedding)

    mapping = converter.map_state_dicts(source_args=inputs, target_args={})
    assert mapping is not None
    mapping["IOUMaskEncoder"] = "iou_token"

    state_dict = converter._convert_state_dict(  # type: ignore
        source_state_dict=mask_decoder.state_dict(),
        target_state_dict=refiners_mask_decoder.state_dict(),
        state_dict_mapping=mapping,
    )
    state_dict["IOUMaskEncoder.weight"] = torch.cat(
        tensors=[mask_decoder.iou_token.weight, mask_decoder.mask_tokens.weight], dim=0
    )  # type: ignore

    refiners_mask_decoder.load_state_dict(state_dict=state_dict)

    refiners_mask_decoder.set_image_embedding(image_embedding)
    refiners_mask_decoder.set_point_embedding(point_embedding)
    refiners_mask_decoder.set_mask_embedding(mask_embedding)
    refiners_mask_decoder.set_dense_positional_embedding(dense_positional_embedding)

    # Perform (1) upscaling then (2) mask prediction in this order (= like in the official implementation) to make
    # `compare_models` happy (MaskPrediction's Matmul runs those in the reverse order by default)
    matmul = refiners_mask_decoder.ensure_find(fl.Matmul)

    def forward_swapped_order(self: Any, *args: Any) -> Any:
        y = self[1](*args)  # (1)
        x = self[0](*args)  # (2)
        return torch.matmul(input=x, other=y)

    matmul.forward = types.MethodType(forward_swapped_order, matmul)

    assert converter.compare_models(source_args=inputs, target_args={}, threshold=1e-3)

    return state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Converts a Segment Anything ViT model to a Refiners SAMViTH model")
    parser.add_argument(
        "--from",
        type=str,
        dest="source_path",
        default="sam_vit_h_4b8939.pth",
        # required=True,
        help="Path to the Segment Anything model weights",
    )
    parser.add_argument(
        "--to",
        type=str,
        dest="output_path",
        default="segment-anything-h.safetensors",
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

    sam_h = build_sam_vit_h()  # type: ignore
    sam_h.load_state_dict(state_dict=load_tensors(args.source_path))

    vit_state_dict = convert_vit(vit=sam_h.image_encoder)
    mask_decoder_state_dict = convert_mask_decoder(mask_decoder=sam_h.mask_decoder)
    point_encoder_state_dict = convert_point_encoder(prompt_encoder=sam_h.prompt_encoder)
    mask_encoder_state_dict = convert_mask_encoder(prompt_encoder=sam_h.prompt_encoder)

    output_state_dict = {
        **{".".join(("image_encoder", key)): value for key, value in vit_state_dict.items()},
        **{".".join(("mask_decoder", key)): value for key, value in mask_decoder_state_dict.items()},
        **{".".join(("point_encoder", key)): value for key, value in point_encoder_state_dict.items()},
        **{".".join(("mask_encoder", key)): value for key, value in mask_encoder_state_dict.items()},
    }
    if args.half:
        output_state_dict = {key: value.half() for key, value in output_state_dict.items()}

    save_to_safetensors(path=args.output_path, tensors=output_state_dict)


if __name__ == "__main__":
    main()
