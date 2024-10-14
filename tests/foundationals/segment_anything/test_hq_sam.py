from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from PIL import Image
from segment_anything_hq import (  # type: ignore
    SamPredictor as SamPredictorHQ,
    sam_model_registry as sam_model_registry_hq,  # type: ignore
)
from segment_anything_hq.modeling.sam import Sam  # type: ignore
from tests.foundationals.segment_anything.utils import FacebookSAM, FacebookSAMPredictorHQ, SAMPrompt
from torch.optim.sgd import SGD

from refiners.fluxion.utils import image_to_tensor, load_from_safetensors, no_grad
from refiners.foundationals.segment_anything.hq_sam import (
    CompressViTFeat,
    EmbeddingEncoder,
    HQSAMAdapter,
    HQTokenMLP,
    MaskDecoderTokensExtender,
    PredictionsPostProc,
)
from refiners.foundationals.segment_anything.model import ImageEmbedding, SegmentAnythingH


@pytest.fixture(scope="module")
def one_prompt() -> SAMPrompt:
    return SAMPrompt(box_points=[[(4, 13), (1007, 1023)]])


@pytest.fixture(scope="module")
def tennis(ref_path: Path) -> Image.Image:
    return Image.open(ref_path / "tennis.png").convert("RGB")  # type: ignore


@pytest.fixture
def sam_h(sam_h_weights_path: Path, test_device: torch.device) -> SegmentAnythingH:
    # HQSAMAdapter is designed to be used with single-output only, hence multimask_output=False.
    sam_h = SegmentAnythingH(multimask_output=False, device=test_device)
    sam_h.load_from_safetensors(tensors_path=sam_h_weights_path)
    return sam_h


@pytest.fixture(scope="module")
def reference_sam_h(sam_h_hq_adapter_unconverted_weights_path: Path, test_device: torch.device) -> FacebookSAM:
    sam_h = cast(FacebookSAM, sam_model_registry_hq["vit_h"](checkpoint=sam_h_hq_adapter_unconverted_weights_path))
    return sam_h.to(device=test_device)


@pytest.fixture(scope="module")
def reference_sam_h_predictor(reference_sam_h: FacebookSAM) -> FacebookSAMPredictorHQ:
    predictor = SamPredictorHQ(cast(Sam, reference_sam_h))
    return cast(FacebookSAMPredictorHQ, predictor)


def test_inject_eject() -> None:
    sam_h = SegmentAnythingH(multimask_output=False)
    initial_repr = repr(sam_h)
    adapter = HQSAMAdapter(sam_h)
    assert repr(sam_h) == initial_repr
    adapter.inject()
    assert repr(sam_h) != initial_repr
    adapter.eject()
    assert repr(sam_h) == initial_repr


def test_multimask_forbidden() -> None:
    with pytest.raises(NotImplementedError, match="not supported"):
        HQSAMAdapter(target=SegmentAnythingH(multimask_output=True))


def test_output_shape_hq_adapter(tennis: Image.Image, one_prompt: SAMPrompt) -> None:
    sam_h = SegmentAnythingH(multimask_output=False)
    HQSAMAdapter(sam_h).inject()
    high_res_masks, iou_predictions, low_res_masks = sam_h.predict(tennis, **one_prompt.__dict__)
    assert high_res_masks.shape == (1, 1, 1024, 1024)
    assert iou_predictions.shape == (1, 1)
    assert low_res_masks.shape == (1, 1, 256, 256)


def test_mask_decoder_tokens_extender() -> None:
    sam_h = SegmentAnythingH(multimask_output=False)
    sam_h.requires_grad_(False)

    # MaskDecoderTokens requires image_embedding context to be set
    image_embedding = torch.randn(2, 256, 64, 64)
    sam_h.mask_decoder.set_image_embedding(image_embedding)

    HQSAMAdapter(sam_h).inject()

    mask_decoder_tokens = sam_h.ensure_find(MaskDecoderTokensExtender)

    tokens_before = mask_decoder_tokens()
    assert tokens_before.shape == torch.Size([2, 6, 256])

    for p in mask_decoder_tokens.parameters():
        match p.shape:
            case torch.Size([5, 256]):
                assert not p.requires_grad
            case torch.Size([1, 256]):
                assert p.requires_grad
            case _:
                raise ValueError

    optimizer = SGD(mask_decoder_tokens.parameters(), lr=10)
    optimizer.zero_grad()

    ones = torch.ones_like(tokens_before)
    loss = torch.nn.functional.mse_loss(tokens_before, ones)
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    tokens_after = mask_decoder_tokens()

    assert torch.equal(tokens_before[:, :5, :], tokens_after[:, :5, :])
    assert not torch.equal(tokens_before[:, 5, :], tokens_after[:, 5, :])


@no_grad()
def test_early_vit_embedding(
    sam_h: SegmentAnythingH,
    sam_h_hq_adapter_weights_path: Path,
    reference_sam_h: FacebookSAM,
    tennis: Image.Image,
) -> None:
    HQSAMAdapter(sam_h, weights=load_from_safetensors(sam_h_hq_adapter_weights_path)).inject()

    image_tensor = image_to_tensor(image=tennis.resize(size=(1024, 1024)))  # type: ignore

    _ = sam_h.image_encoder(image_tensor.to(sam_h.device))
    early_vit_embedding_refiners = sam_h.use_context(context_name="hq_sam")["early_vit_embedding"]

    _, intermediate_embeddings = reference_sam_h.image_encoder(image_tensor.to(reference_sam_h.device))
    early_vit_embedding = intermediate_embeddings[0]

    assert torch.equal(early_vit_embedding, early_vit_embedding_refiners)


def test_tokens(sam_h: SegmentAnythingH, sam_h_hq_adapter_weights_path: Path, reference_sam_h: FacebookSAM) -> None:
    HQSAMAdapter(sam_h, weights=load_from_safetensors(sam_h_hq_adapter_weights_path)).inject()

    mask_decoder_tokens_extender = sam_h.mask_decoder.ensure_find(MaskDecoderTokensExtender)

    # HF Token (1, 256)
    assert torch.equal(reference_sam_h.mask_decoder.hf_token.weight, mask_decoder_tokens_extender.hq_token.weight)

    # Regular Tokens (5, 256)
    assert torch.equal(
        torch.cat([reference_sam_h.mask_decoder.iou_token.weight, reference_sam_h.mask_decoder.mask_tokens.weight]),
        mask_decoder_tokens_extender.regular_tokens.weight,
    )


@no_grad()
def test_compress_vit_feat(
    sam_h: SegmentAnythingH, sam_h_hq_adapter_weights_path: Path, reference_sam_h: FacebookSAM
) -> None:
    HQSAMAdapter(sam_h, weights=load_from_safetensors(sam_h_hq_adapter_weights_path)).inject()

    early_vit_embedding = torch.randn(1, 64, 64, 1280, device=sam_h.device, dtype=sam_h.dtype)

    sam_h.set_context(context="hq_sam", value={"early_vit_embedding": early_vit_embedding})
    refiners_output = sam_h.ensure_find(CompressViTFeat)()

    reference_output = reference_sam_h.mask_decoder.compress_vit_feat(early_vit_embedding.permute(0, 3, 1, 2))

    assert torch.equal(refiners_output, reference_output)


@no_grad()
def test_embedding_encoder(
    sam_h: SegmentAnythingH, sam_h_hq_adapter_weights_path: Path, reference_sam_h: FacebookSAM
) -> None:
    HQSAMAdapter(sam_h, weights=load_from_safetensors(sam_h_hq_adapter_weights_path)).inject()

    x = torch.randn(1, 256, 64, 64, device=sam_h.device, dtype=sam_h.dtype)

    sam_h.set_context(context="mask_decoder", value={"image_embedding": x})
    refiners_output = sam_h.ensure_find(EmbeddingEncoder)()

    reference_output = reference_sam_h.mask_decoder.embedding_encoder(x)

    assert torch.equal(refiners_output, reference_output)


@no_grad()
def test_hq_token_mlp(
    sam_h: SegmentAnythingH, sam_h_hq_adapter_weights_path: Path, reference_sam_h: FacebookSAM
) -> None:
    HQSAMAdapter(sam_h, weights=load_from_safetensors(sam_h_hq_adapter_weights_path)).inject()

    x = torch.randn(1, 6, 256, device=sam_h.device, dtype=sam_h.dtype)

    refiners_output = sam_h.ensure_find(HQTokenMLP)(x)
    reference_output = reference_sam_h.mask_decoder.hf_mlp(x[:, -1, :]).unsqueeze(0)

    assert torch.equal(refiners_output, reference_output)


@pytest.mark.parametrize("hq_mask_only", [True, False])
def test_predictor(
    sam_h: SegmentAnythingH,
    sam_h_hq_adapter_weights_path: Path,
    hq_mask_only: bool,
    reference_sam_h_predictor: FacebookSAMPredictorHQ,
    tennis: Image.Image,
    one_prompt: SAMPrompt,
) -> None:
    adapter = HQSAMAdapter(sam_h, weights=load_from_safetensors(sam_h_hq_adapter_weights_path)).inject()

    adapter.hq_mask_only = hq_mask_only
    assert sam_h.ensure_find(PredictionsPostProc).hq_mask_only == hq_mask_only

    # Refiners
    high_res_masks, iou_predictions, low_res_masks = sam_h.predict(tennis, **one_prompt.__dict__)
    refiners_high_res_mask_hq = high_res_masks[0, 0, ...].to(dtype=torch.float32).detach().cpu()
    refiners_low_res_mask_hq = low_res_masks[0, 0, ...].to(dtype=torch.float32).detach().cpu()
    iou_predictions = iou_predictions[0, :].to(dtype=torch.float32).detach().cpu()

    # Reference
    reference_sam_h_predictor.set_image(np.array(tennis))

    predictor_prompt = one_prompt.__dict__["box_points"]
    masks_np, iou_predictions_np, low_res_masks_np = reference_sam_h_predictor.predict(
        box=np.array(predictor_prompt).flatten(),
        multimask_output=False,
        hq_token_only=hq_mask_only,
    )

    reference_high_res_mask_hq = torch.from_numpy(masks_np[0, ...]).to(dtype=torch.float32)  # type: ignore
    reference_low_res_mask_hq = torch.from_numpy(low_res_masks_np[0, ...]).to(dtype=torch.float32)  # type: ignore
    iou_predictions_np = torch.from_numpy(iou_predictions_np).to(dtype=torch.float32)  # type: ignore

    # NOTE: Diff on logits is relatively high,
    # see test_predictor_equal for a stricter version
    assert torch.allclose(
        reference_low_res_mask_hq,
        refiners_low_res_mask_hq,
        atol=1e-2,
    )
    assert (  # absolute diff in number of pixels
        torch.abs(reference_high_res_mask_hq - refiners_high_res_mask_hq).flatten().sum() <= 10
    )
    assert torch.allclose(
        iou_predictions_np,
        torch.max(iou_predictions),
        atol=1e-4,
    )


@pytest.mark.parametrize("hq_mask_only", [True, False])
def test_predictor_equal(
    sam_h: SegmentAnythingH,
    sam_h_hq_adapter_weights_path: Path,
    hq_mask_only: bool,
    reference_sam_h_predictor: FacebookSAMPredictorHQ,
    tennis: Image.Image,
    one_prompt: SAMPrompt,
) -> None:
    adapter = HQSAMAdapter(sam_h, weights=load_from_safetensors(sam_h_hq_adapter_weights_path)).inject()

    adapter.hq_mask_only = hq_mask_only
    assert sam_h.ensure_find(PredictionsPostProc).hq_mask_only == hq_mask_only

    # See in test_sam.py test_predictor_resized_single_output
    # to do torch.equal we need to resize the image before
    # and to use image_embedding as input

    size = (1024, 1024)
    resized_tennis = tennis.resize(size)  # type: ignore

    # Reference
    reference_sam_h_predictor.set_image(np.array(resized_tennis))

    predictor_prompt = one_prompt.__dict__["box_points"]
    masks_np, _, low_res_masks_np = reference_sam_h_predictor.predict(
        box=np.array(predictor_prompt).flatten(),
        multimask_output=False,
        hq_token_only=hq_mask_only,
    )

    reference_high_res_mask_hq = torch.from_numpy(masks_np[0, ...]).to(dtype=torch.float32)  # type: ignore
    reference_low_res_mask_hq = torch.from_numpy(low_res_masks_np[0, ...]).to(dtype=torch.float32)  # type: ignore

    # Refiners

    # We bypass the refiners ViT by using directly the image features and interm_features
    # from the reference implementation: this gives the ability to do torch.equal
    reference_image_embedding = ImageEmbedding(features=reference_sam_h_predictor.features, original_image_size=size)
    adapter.set_context("hq_sam", {"early_vit_embedding": reference_sam_h_predictor.interm_features[0]})

    high_res_masks, _, low_res_masks = sam_h.predict(reference_image_embedding, **one_prompt.__dict__)
    refiners_high_res_mask_hq = high_res_masks[0, 0, ...].to(dtype=torch.float32).detach().cpu()
    refiners_low_res_mask_hq = low_res_masks[0, 0, ...].to(dtype=torch.float32).detach().cpu()

    assert torch.equal(
        reference_low_res_mask_hq,
        refiners_low_res_mask_hq,
    )
    assert torch.abs(reference_high_res_mask_hq - refiners_high_res_mask_hq).flatten().sum() == 0


@no_grad()
def test_batch_mask_decoder(sam_h: SegmentAnythingH, sam_h_hq_adapter_weights_path: Path) -> None:
    HQSAMAdapter(sam_h, weights=load_from_safetensors(sam_h_hq_adapter_weights_path)).inject()

    batch_size = 5

    image_embedding = torch.randn(1, 256, 64, 64, device=sam_h.device, dtype=sam_h.dtype).repeat(batch_size, 1, 1, 1)
    mask_embedding = torch.randn(1, 256, 64, 64, device=sam_h.device, dtype=sam_h.dtype).repeat(batch_size, 1, 1, 1)
    dense_positional_embedding = torch.randn(1, 256, 64, 64, device=sam_h.device, dtype=sam_h.dtype).repeat(
        batch_size, 1, 1, 1
    )
    point_embedding = torch.randn(1, 2, 256, device=sam_h.device, dtype=sam_h.dtype).repeat(batch_size, 1, 1)
    early_vit_embedding = torch.randn(1, 64, 64, 1280, device=sam_h.device, dtype=sam_h.dtype).repeat(
        batch_size, 1, 1, 1
    )

    sam_h.mask_decoder.set_image_embedding(image_embedding)
    sam_h.mask_decoder.set_mask_embedding(mask_embedding)
    sam_h.mask_decoder.set_point_embedding(point_embedding)
    sam_h.mask_decoder.set_dense_positional_embedding(dense_positional_embedding)
    sam_h.mask_decoder.set_context(
        context="hq_sam", value={"early_vit_embedding": early_vit_embedding.to(sam_h.device, sam_h.dtype)}
    )

    mask_prediction, iou_prediction = sam_h.mask_decoder()

    assert mask_prediction.shape == (batch_size, 1, 256, 256)
    assert iou_prediction.shape == (batch_size, 1)
    assert torch.equal(mask_prediction[0], mask_prediction[1])


def test_hq_sam_load_save_weights(
    sam_h: SegmentAnythingH, sam_h_hq_adapter_weights_path: Path, test_device: torch.device
) -> None:
    weights = load_from_safetensors(sam_h_hq_adapter_weights_path, device=test_device)

    hq_sam_adapter = HQSAMAdapter(sam_h)
    out_weights_init = hq_sam_adapter.weights

    assert set(out_weights_init.keys()) == set(weights.keys())

    hq_sam_adapter = HQSAMAdapter(sam_h, weights=weights)
    out_weights = hq_sam_adapter.weights

    assert set(out_weights.keys()) == set(weights.keys())
    for key in out_weights.keys():
        assert torch.equal(out_weights[key], weights[key])
