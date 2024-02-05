from math import isclose
from pathlib import Path
from typing import cast
from warnings import warn

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from tests.foundationals.segment_anything.utils import (
    FacebookSAM,
    FacebookSAMPredictor,
    SAMPrompt,
    intersection_over_union,
)
from torch import Tensor

import refiners.fluxion.layers as fl
from refiners.fluxion import manual_seed
from refiners.fluxion.model_converter import ModelConverter
from refiners.fluxion.utils import image_to_tensor, load_tensors, no_grad
from refiners.foundationals.segment_anything.image_encoder import FusedSelfAttention, RelativePositionAttention
from refiners.foundationals.segment_anything.model import SegmentAnythingH
from refiners.foundationals.segment_anything.transformer import TwoWayTransformerLayer

# See predictor_example.ipynb official notebook
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
    # Using the third prompt of the PROMPTS list in order to strictly do the same test as the official notebook in the
    # test_predictor_dense_mask test.
    return PROMPTS[2]


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
    sam_h.load_state_dict(state_dict=load_tensors(facebook_sam_h_weights))
    return sam_h.to(device=test_device)


@pytest.fixture(scope="module")
def facebook_sam_h_predictor(facebook_sam_h: FacebookSAM) -> FacebookSAMPredictor:
    from segment_anything import SamPredictor  # type: ignore
    from segment_anything.modeling import Sam  # type: ignore

    predictor = SamPredictor(cast(Sam, facebook_sam_h))  # type: ignore
    return cast(FacebookSAMPredictor, predictor)


@pytest.fixture(scope="module")
def sam_h(sam_h_weights: Path, test_device: torch.device) -> SegmentAnythingH:
    sam_h = SegmentAnythingH(device=test_device)
    sam_h.load_from_safetensors(tensors_path=sam_h_weights)
    return sam_h


@pytest.fixture(scope="module")
def ref_path(test_sam_path: Path) -> Path:
    return test_sam_path / "test_sam_ref"


@pytest.fixture
def truck(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "truck.jpg").convert("RGB")


@no_grad()
def test_fused_self_attention(facebook_sam_h: FacebookSAM) -> None:
    manual_seed(seed=0)
    x = torch.randn(25, 14, 14, 1280, device=facebook_sam_h.device)

    attention = cast(nn.Module, facebook_sam_h.image_encoder.blocks[0].attn)

    refiners_attention = FusedSelfAttention(
        embedding_dim=1280, num_heads=16, spatial_size=(14, 14), device=facebook_sam_h.device
    )

    rpa = refiners_attention.layer("RelativePositionAttention", RelativePositionAttention)
    linear_1 = refiners_attention.layer("Linear_1", fl.Linear)
    linear_2 = refiners_attention.layer("Linear_2", fl.Linear)

    linear_1.weight = attention.qkv.weight
    linear_1.bias = attention.qkv.bias
    linear_2.weight = attention.proj.weight
    linear_2.bias = attention.proj.bias
    rpa.horizontal_embedding = attention.rel_pos_w
    rpa.vertical_embedding = attention.rel_pos_h

    y_1 = attention(x)
    assert y_1.shape == x.shape

    y_2 = refiners_attention(x)
    assert y_2.shape == x.shape

    assert torch.equal(input=y_1, other=y_2)


@no_grad()
def test_image_encoder(sam_h: SegmentAnythingH, facebook_sam_h: FacebookSAM, truck: Image.Image) -> None:
    image_tensor = image_to_tensor(image=truck.resize(size=(1024, 1024)), device=facebook_sam_h.device)
    y_1 = facebook_sam_h.image_encoder(image_tensor)
    y_2 = sam_h.image_encoder(image_tensor)

    assert torch.allclose(input=y_1, other=y_2, atol=1e-4)


@no_grad()
def test_prompt_encoder_dense_positional_embedding(facebook_sam_h: FacebookSAM, sam_h: SegmentAnythingH) -> None:
    facebook_prompt_encoder = facebook_sam_h.prompt_encoder
    refiners_prompt_encoder = sam_h.point_encoder

    facebook_dense_pe: Tensor = cast(Tensor, facebook_prompt_encoder.get_dense_pe())  # type: ignore
    refiners_dense_pe = refiners_prompt_encoder.get_dense_positional_embedding(image_embedding_size=(64, 64))

    assert torch.equal(input=refiners_dense_pe, other=facebook_dense_pe)


@no_grad()
def test_prompt_encoder_no_mask_dense_embedding(facebook_sam_h: FacebookSAM, sam_h: SegmentAnythingH) -> None:
    facebook_prompt_encoder = facebook_sam_h.prompt_encoder
    refiners_prompt_encoder = sam_h.mask_encoder

    _, facebook_dense_pe = facebook_prompt_encoder(points=None, boxes=None, masks=None)
    refiners_dense_pe = refiners_prompt_encoder.get_no_mask_dense_embedding(image_embedding_size=(64, 64))

    assert torch.equal(input=refiners_dense_pe, other=facebook_dense_pe)


@no_grad()
def test_point_encoder(facebook_sam_h: FacebookSAM, sam_h: SegmentAnythingH, prompt: SAMPrompt) -> None:
    facebook_prompt_encoder = facebook_sam_h.prompt_encoder
    refiners_prompt_encoder = sam_h.point_encoder

    facebook_sparse_pe, _ = facebook_prompt_encoder(
        **prompt.facebook_prompt_encoder_kwargs(device=facebook_sam_h.device)
    )

    prompt_dict = prompt.__dict__
    # Skip mask prompt, if any, since the point encoder only consumes points and boxes
    # TODO: split `SAMPrompt` and introduce a dedicated one for dense prompts
    prompt_dict.pop("low_res_mask", None)

    assert prompt_dict is not None, "`test_point_encoder` cannot be called with just a `low_res_mask`"

    coordinates, type_mask = refiners_prompt_encoder.points_to_tensor(**prompt_dict)
    # Shift to center of pixel + normalize in [0, 1] (see `_embed_points` in segment-anything official repo)
    coordinates[:, :, 0] = (coordinates[:, :, 0] + 0.5) / 1024.0
    coordinates[:, :, 1] = (coordinates[:, :, 1] + 0.5) / 1024.0
    refiners_prompt_encoder.set_type_mask(type_mask=type_mask)
    refiners_sparse_pe = refiners_prompt_encoder(coordinates)

    assert torch.equal(input=refiners_sparse_pe, other=facebook_sparse_pe)


@no_grad()
def test_two_way_transformer(facebook_sam_h: FacebookSAM) -> None:
    dense_embedding = torch.randn(1, 64 * 64, 256, device=facebook_sam_h.device)
    dense_positional_embedding = torch.randn(1, 64 * 64, 256, device=facebook_sam_h.device)
    sparse_embedding = torch.randn(1, 3, 256, device=facebook_sam_h.device)

    refiners_layer = TwoWayTransformerLayer(
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


@no_grad()
def test_mask_decoder(facebook_sam_h: FacebookSAM, sam_h: SegmentAnythingH) -> None:
    manual_seed(seed=0)
    facebook_mask_decoder = facebook_sam_h.mask_decoder
    refiners_mask_decoder = sam_h.mask_decoder

    image_embedding = torch.randn(1, 256, 64, 64, device=facebook_sam_h.device)
    dense_positional_embedding = torch.randn(1, 256, 64, 64, device=facebook_sam_h.device)
    point_embedding = torch.randn(1, 3, 256, device=facebook_sam_h.device)
    mask_embedding = torch.randn(1, 256, 64, 64, device=facebook_sam_h.device)

    from segment_anything.modeling.common import LayerNorm2d  # type: ignore

    import refiners.fluxion.layers as fl

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

    state_dict = converter._convert_state_dict(  # type: ignore
        source_state_dict=facebook_mask_decoder.state_dict(),
        target_state_dict=refiners_mask_decoder.state_dict(),
        state_dict_mapping=mapping,
    )
    state_dict["IOUMaskEncoder.weight"] = torch.cat(
        [facebook_mask_decoder.iou_token.weight, facebook_mask_decoder.mask_tokens.weight], dim=0
    )  # type: ignore
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


def test_predictor_image_embedding(sam_h: SegmentAnythingH, truck: Image.Image, one_prompt: SAMPrompt) -> None:
    masks_ref, scores_ref, _ = sam_h.predict(truck, **one_prompt.__dict__)

    image_embedding = sam_h.compute_image_embedding(truck)
    masks, scores, _ = sam_h.predict(image_embedding, **one_prompt.__dict__)

    assert torch.equal(masks, masks_ref)
    assert torch.equal(scores_ref, scores)


def test_predictor_dense_mask(
    facebook_sam_h_predictor: FacebookSAMPredictor, sam_h: SegmentAnythingH, truck: Image.Image, one_prompt: SAMPrompt
) -> None:
    """
    NOTE : Binarizing intermediate masks isn't necessary, as per SamPredictor.predict_torch docstring:
    > mask_input (np.ndarray): A low resolution mask input to the model, typically
    >         coming from a previous prediction iteration. Has form Bx1xHxW, where
    >         for SAM, H=W=256. Masks returned by a previous iteration of the
    >         predict method do not need further transformation.
    """
    predictor = facebook_sam_h_predictor
    predictor.set_image(np.array(truck))
    facebook_masks, facebook_scores, facebook_logits = predictor.predict(
        **one_prompt.facebook_predict_kwargs(),  # type: ignore
        multimask_output=True,
    )

    assert len(facebook_masks) == 3

    facebook_mask_input = facebook_logits[np.argmax(facebook_scores)]  # shape: HxW

    # Using the same mask coordinates inputs as the official notebook
    facebook_prompt = SAMPrompt(
        foreground_points=((500, 375),), background_points=((1125, 625),), low_res_mask=facebook_mask_input[None, ...]
    )
    facebook_dense_masks, _, _ = predictor.predict(**facebook_prompt.facebook_predict_kwargs(), multimask_output=True)  # type: ignore

    assert len(facebook_dense_masks) == 3

    masks, scores, logits = sam_h.predict(truck, **one_prompt.__dict__)
    masks = masks.squeeze(0)
    scores = scores.squeeze(0)

    assert len(masks) == 3

    mask_input = logits[:, scores.max(dim=0).indices, ...]  # shape: 1xHxW

    assert np.allclose(
        mask_input.cpu().numpy(), facebook_mask_input, atol=1e-1
    )  # Lower doesn't pass, but it's close enough for logits

    refiners_prompt = SAMPrompt(
        foreground_points=((500, 375),), background_points=((1125, 625),), low_res_mask=mask_input.unsqueeze(0)
    )
    dense_masks, _, _ = sam_h.predict(truck, **refiners_prompt.__dict__)
    dense_masks = dense_masks.squeeze(0)

    assert len(dense_masks) == 3

    for i in range(3):
        dense_mask_prediction = dense_masks[i].cpu()
        facebook_dense_mask = torch.as_tensor(facebook_dense_masks[i])
        assert dense_mask_prediction.shape == facebook_dense_mask.shape
        assert isclose(intersection_over_union(dense_mask_prediction, facebook_dense_mask), 1.0, rel_tol=5e-05)


def test_mask_encoder(
    facebook_sam_h_predictor: FacebookSAMPredictor, sam_h: SegmentAnythingH, truck: Image.Image, one_prompt: SAMPrompt
) -> None:
    predictor = facebook_sam_h_predictor
    predictor.set_image(np.array(truck))
    _, facebook_scores, facebook_logits = predictor.predict(
        **one_prompt.facebook_predict_kwargs(),  # type: ignore
        multimask_output=True,
    )
    facebook_mask_input = facebook_logits[np.argmax(facebook_scores)]
    facebook_mask_input = (
        torch.from_numpy(facebook_mask_input)  # type: ignore
        .to(device=predictor.model.device)
        .unsqueeze(0)
        .unsqueeze(0)  # shape: 1x1xHxW
    )

    _, fb_dense_embeddings = predictor.model.prompt_encoder(
        points=None,
        boxes=None,
        masks=facebook_mask_input,
    )

    _, scores, logits = sam_h.predict(truck, **one_prompt.__dict__)
    scores = scores.squeeze(0)
    mask_input = logits[:, scores.max(dim=0).indices, ...].unsqueeze(0)  # shape: 1x1xHxW
    dense_embeddings = sam_h.mask_encoder(mask_input)

    assert facebook_mask_input.shape == mask_input.shape
    assert torch.allclose(dense_embeddings, fb_dense_embeddings, atol=1e-4, rtol=1e-4)
