from math import isclose
from pathlib import Path
from typing import cast
from warnings import warn

import pytest
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torch import Tensor
from refiners.fluxion import manual_seed
from refiners.fluxion.model_converter import ModelConverter

from refiners.fluxion.utils import image_to_tensor
from refiners.foundationals.segment_anything.image_encoder import FusedSelfAttention
from refiners.foundationals.segment_anything.model import SegmentAnythingH
from refiners.foundationals.segment_anything.transformer import TwoWayTranformerLayer
from tests.foundationals.segment_anything.utils import (
    FacebookSAM,
    FacebookSAMPredictor,
    SAMPrompt,
    intersection_over_union,
)

# See predictor_example.ipynb official notebook (note: mask_input is not yet properly supported)
PROMPTS: list[SAMPrompt] = [
    SAMPrompt(foreground_points=((500, 375),)),
    SAMPrompt(background_points=((500, 375),)),
    SAMPrompt(foreground_points=((500, 375), (1125, 625))),
    SAMPrompt(foreground_points=((500, 375),), background_points=((1125, 625),)),
    SAMPrompt(box_points=[[(425, 600), (700, 875)]]),
    SAMPrompt(box_points=[[(425, 600), (700, 875)]], background_points=((575, 750),)),
]


@pytest.fixture(params=PROMPTS)
def prompt(request: pytest.FixtureRequest) -> SAMPrompt:
    return request.param


@pytest.fixture
def one_prompt() -> SAMPrompt:
    return PROMPTS[0]


@pytest.fixture(scope="module")
def facebook_sam_h_weights(test_weights_path: Path) -> Path:
    sam_h_weights = test_weights_path / "sam_vit_h_4b8939.pth"
    if not sam_h_weights.is_file():
        warn(f"could not find weights at {sam_h_weights}, skipping")
        pytest.skip(allow_module_level=True)
    return sam_h_weights


@pytest.fixture(scope="module")
def sam_h_weights(test_weights_path: Path) -> Path:
    sam_h_weights = test_weights_path / "segment-anything-h.safetensors"
    if not sam_h_weights.is_file():
        warn(f"could not find weights at {sam_h_weights}, skipping")
        pytest.skip(allow_module_level=True)
    return sam_h_weights


@pytest.fixture(scope="module")
def facebook_sam_h(facebook_sam_h_weights: Path, test_device: torch.device) -> FacebookSAM:
    from segment_anything import build_sam_vit_h  # type: ignore

    sam_h = cast(FacebookSAM, build_sam_vit_h())
    sam_h.load_state_dict(state_dict=torch.load(f=facebook_sam_h_weights))  # type: ignore
    return sam_h.to(device=test_device)


@pytest.fixture(scope="module")
def facebook_sam_h_predictor(facebook_sam_h: FacebookSAM) -> FacebookSAMPredictor:
    from segment_anything import SamPredictor  # type: ignore
    from segment_anything.modeling import Sam  # type: ignore

    predictor = SamPredictor(cast(Sam, facebook_sam_h))
    return cast(FacebookSAMPredictor, predictor)


@pytest.fixture(scope="module")
def sam_h(sam_h_weights: Path, test_device: torch.device) -> SegmentAnythingH:
    sam_h = SegmentAnythingH(device=test_device)
    # TODO: make strict=True when the MasKEncoder conversion is done
    sam_h.load_from_safetensors(tensors_path=sam_h_weights, strict=False)
    return sam_h


@pytest.fixture(scope="module")
def ref_path(test_sam_path: Path) -> Path:
    return test_sam_path / "test_sam_ref"


@pytest.fixture
def truck(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "truck.jpg").convert("RGB")


@torch.no_grad()
def test_fused_self_attention(facebook_sam_h: FacebookSAM) -> None:
    manual_seed(seed=0)
    x = torch.randn(25, 14, 14, 1280, device=facebook_sam_h.device)

    attention = cast(nn.Module, facebook_sam_h.image_encoder.blocks[0].attn)  # type: ignore

    refiners_attention = FusedSelfAttention(
        embedding_dim=1280, num_heads=16, spatial_size=(14, 14), device=facebook_sam_h.device
    )
    refiners_attention.Linear_1.weight = attention.qkv.weight  # type: ignore
    refiners_attention.Linear_1.bias = attention.qkv.bias  # type: ignore
    refiners_attention.Linear_2.weight = attention.proj.weight  # type: ignore
    refiners_attention.Linear_2.bias = attention.proj.bias  # type: ignore
    refiners_attention.RelativePositionAttention.horizontal_embedding = attention.rel_pos_w
    refiners_attention.RelativePositionAttention.vertical_embedding = attention.rel_pos_h

    y_1 = attention(x)
    assert y_1.shape == x.shape

    y_2 = refiners_attention(x)
    assert y_2.shape == x.shape

    assert torch.equal(input=y_1, other=y_2)


@torch.no_grad()
def test_image_encoder(sam_h: SegmentAnythingH, facebook_sam_h: FacebookSAM, truck: Image.Image) -> None:
    image_tensor = image_to_tensor(image=truck.resize(size=(1024, 1024)), device=facebook_sam_h.device)
    y_1 = facebook_sam_h.image_encoder(image_tensor)
    y_2 = sam_h.image_encoder(image_tensor)

    assert torch.allclose(input=y_1, other=y_2, atol=1e-4)


@torch.no_grad()
def test_prompt_encoder_dense_positional_embedding(facebook_sam_h: FacebookSAM, sam_h: SegmentAnythingH) -> None:
    facebook_prompt_encoder = facebook_sam_h.prompt_encoder
    refiners_prompt_encoder = sam_h.point_encoder

    facebook_dense_pe: Tensor = cast(Tensor, facebook_prompt_encoder.get_dense_pe())  # type: ignore
    refiners_dense_pe = refiners_prompt_encoder.get_dense_positional_embedding(image_embedding_size=(64, 64))

    assert torch.equal(input=refiners_dense_pe, other=facebook_dense_pe)


@torch.no_grad()
def test_prompt_encoder_no_mask_dense_embedding(facebook_sam_h: FacebookSAM, sam_h: SegmentAnythingH) -> None:
    facebook_prompt_encoder = facebook_sam_h.prompt_encoder
    refiners_prompt_encoder = sam_h.mask_encoder

    _, facebook_dense_pe = facebook_prompt_encoder(points=None, boxes=None, masks=None)
    refiners_dense_pe = refiners_prompt_encoder.get_no_mask_dense_embedding(image_embedding_size=(64, 64))

    assert torch.equal(input=refiners_dense_pe, other=facebook_dense_pe)


@torch.no_grad()
def test_point_encoder(facebook_sam_h: FacebookSAM, sam_h: SegmentAnythingH, prompt: SAMPrompt) -> None:
    facebook_prompt_encoder = facebook_sam_h.prompt_encoder
    refiners_prompt_encoder = sam_h.point_encoder

    facebook_sparse_pe, _ = facebook_prompt_encoder(
        **prompt.facebook_prompt_encoder_kwargs(device=facebook_sam_h.device)
    )

    coordinates, type_mask = refiners_prompt_encoder.points_to_tensor(**prompt.__dict__)
    # Shift to center of pixel + normalize in [0, 1] (see `_embed_points` in segment-anything official repo)
    coordinates[:, :, 0] = (coordinates[:, :, 0] + 0.5) / 1024.0
    coordinates[:, :, 1] = (coordinates[:, :, 1] + 0.5) / 1024.0
    refiners_prompt_encoder.set_type_mask(type_mask=type_mask)
    refiners_sparse_pe = refiners_prompt_encoder(coordinates)

    assert torch.equal(input=refiners_sparse_pe, other=facebook_sparse_pe)


@torch.no_grad()
def test_two_way_transformer(facebook_sam_h: FacebookSAM) -> None:
    dense_embedding = torch.randn(1, 64 * 64, 256, device=facebook_sam_h.device)
    dense_positional_embedding = torch.randn(1, 64 * 64, 256, device=facebook_sam_h.device)
    sparse_embedding = torch.randn(1, 3, 256, device=facebook_sam_h.device)

    refiners_layer = TwoWayTranformerLayer(
        embedding_dim=256, feed_forward_dim=2048, num_heads=8, device=facebook_sam_h.device
    )
    facebook_layer = facebook_sam_h.mask_decoder.transformer.layers[1]  # type: ignore
    assert isinstance(facebook_layer, nn.Module)

    refiners_layer.set_context(
        context="mask_decoder",
        value={
            "dense_embedding": dense_embedding,
            "dense_positional_embedding": dense_positional_embedding,
            "sparse_embedding": sparse_embedding,
        },
    )
    facebook_inputs = {
        "queries": sparse_embedding,
        "keys": dense_embedding,
        "query_pe": sparse_embedding,
        "key_pe": dense_positional_embedding,
    }

    converter = ModelConverter(
        source_model=facebook_layer,
        target_model=refiners_layer,
        skip_output_check=True,  # done below, manually
    )

    assert converter.run(source_args=facebook_inputs, target_args=(sparse_embedding,))

    refiners_layer.set_context(
        context="mask_decoder",
        value={
            "dense_embedding": dense_embedding,
            "dense_positional_embedding": dense_positional_embedding,
            "sparse_embedding": sparse_embedding,
        },
    )
    y_1 = facebook_layer(**facebook_inputs)[0]
    y_2 = refiners_layer(sparse_embedding)[0]

    assert torch.equal(input=y_1, other=y_2)


@torch.no_grad()
def test_mask_decoder(facebook_sam_h: FacebookSAM, sam_h: SegmentAnythingH) -> None:
    manual_seed(seed=0)
    facebook_mask_decoder = facebook_sam_h.mask_decoder
    refiners_mask_decoder = sam_h.mask_decoder

    image_embedding = torch.randn(1, 256, 64, 64, device=facebook_sam_h.device)
    dense_positional_embedding = torch.randn(1, 256, 64, 64, device=facebook_sam_h.device)
    point_embedding = torch.randn(1, 3, 256, device=facebook_sam_h.device)
    mask_embedding = torch.randn(1, 256, 64, 64, device=facebook_sam_h.device)

    import refiners.fluxion.layers as fl
    from segment_anything.modeling.common import LayerNorm2d  # type: ignore

    assert issubclass(LayerNorm2d, nn.Module)
    custom_layers = {LayerNorm2d: fl.LayerNorm2d}

    converter = ModelConverter(
        source_model=facebook_mask_decoder,
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

    state_dict = converter._convert_state_dict(source_state_dict=facebook_mask_decoder.state_dict(), target_state_dict=refiners_mask_decoder.state_dict(), state_dict_mapping=mapping)  # type: ignore
    state_dict["IOUMaskEncoder.weight"] = torch.cat([facebook_mask_decoder.iou_token.weight, facebook_mask_decoder.mask_tokens.weight], dim=0)  # type: ignore
    refiners_mask_decoder.load_state_dict(state_dict=state_dict)

    facebook_output = facebook_mask_decoder(**inputs)

    refiners_mask_decoder.set_image_embedding(image_embedding)
    refiners_mask_decoder.set_point_embedding(point_embedding)
    refiners_mask_decoder.set_mask_embedding(mask_embedding)
    refiners_mask_decoder.set_dense_positional_embedding(dense_positional_embedding)
    mask_prediction, iou_prediction = refiners_mask_decoder()

    facebook_masks = facebook_output[0]
    facebook_prediction = facebook_output[1]

    assert torch.equal(input=mask_prediction, other=facebook_masks)
    assert torch.equal(input=iou_prediction, other=facebook_prediction)


@torch.no_grad()
def test_predictor(
    facebook_sam_h_predictor: FacebookSAMPredictor, sam_h: SegmentAnythingH, truck: Image.Image, prompt: SAMPrompt
) -> None:
    predictor = facebook_sam_h_predictor
    predictor.set_image(np.array(truck))
    facebook_masks, facebook_scores, _ = predictor.predict(**prompt.facebook_predict_kwargs())  # type: ignore

    assert len(facebook_masks) == 3

    masks, scores, _ = sam_h.predict(truck, **prompt.__dict__)
    masks = masks.squeeze(0)
    scores = scores.squeeze(0)

    assert len(masks) == 3

    for i in range(3):
        mask_prediction = masks[i].cpu()
        facebook_mask = torch.as_tensor(facebook_masks[i])
        assert isclose(intersection_over_union(mask_prediction, facebook_mask), 1.0, rel_tol=5e-05)
        assert isclose(scores[i].item(), facebook_scores[i].item(), rel_tol=1e-05)


@torch.no_grad()
def test_predictor_image_embedding(sam_h: SegmentAnythingH, truck: Image.Image, one_prompt: SAMPrompt) -> None:
    masks_ref, scores_ref, _ = sam_h.predict(truck, **one_prompt.__dict__)

    image_embedding = sam_h.compute_image_embedding(truck)
    masks, scores, _ = sam_h.predict(image_embedding, **one_prompt.__dict__)

    assert torch.equal(masks, masks_ref)
    assert torch.equal(scores_ref, scores)
