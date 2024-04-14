from math import isclose
from pathlib import Path
from warnings import warn

import pytest
import torch

from refiners.fluxion.utils import load_from_safetensors, manual_seed, no_grad
from refiners.foundationals.fuyu import Fuyu8b, create_fuyu




def test_encoder(
    ref_backbone: Dinov2Model,
    our_backbone: ViT,
    test_device: torch.device,
):
    manual_seed(42)

    # Position encoding interpolation [1] at runtime is not supported yet. So stick to the default image resolution
    # e.g. using (224, 224) pixels as input would give a runtime error (sequence size mismatch)
    # [1]: https://github.com/facebookresearch/dinov2/blob/2302b6b/dinov2/models/vision_transformer.py#L179
    assert our_backbone.image_size == 518

    x = torch.randn(1, 3, 518, 518).to(test_device)

    with no_grad():
        ref_features = ref_backbone(x).last_hidden_state
        our_features = our_backbone(x)

    assert (our_features - ref_features).abs().max() < 1e-3


# Mainly for DINOv2 + registers coverage (this test can be removed once `test_encoder` supports all flavors)
def test_encoder_only(
    our_backbone: ViT,
    seed_expected_norm: tuple[int, float],
    test_device: torch.device,
):
    seed, expected_norm = seed_expected_norm
    manual_seed(seed)

    x = torch.randn(1, 3, 518, 518).to(test_device)

    our_features = our_backbone(x)

    assert isclose(our_features.norm().item(), expected_norm, rel_tol=1e-04)
