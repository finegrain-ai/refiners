from functools import cache
from pathlib import Path
from textwrap import dedent

import piq  # type: ignore
import torch
import torch.nn as nn
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer  # type: ignore

from refiners.conversion.models import dinov2
from refiners.fluxion.utils import image_to_tensor
from refiners.foundationals.dinov2 import DINOv2_small


@cache
def get_small_dinov2_model() -> DINOv2_small:
    model = DINOv2_small()
    model.load_from_safetensors(
        dinov2.small.converted.local_path
        if dinov2.small.converted.local_path.exists()
        else dinov2.small.converted.hf_cache_path
    )
    return model


def compare_images(
    img_1: Image.Image,
    img_2: Image.Image,
) -> tuple[float, float, float]:
    x1 = image_to_tensor(img_1)
    x2 = image_to_tensor(img_2)

    psnr = piq.psnr(x1, x2)  # type: ignore
    ssim = piq.ssim(x1, x2)  # type: ignore

    dinov2_model = get_small_dinov2_model()
    dinov2 = torch.nn.functional.cosine_similarity(
        dinov2_model(x1)[:, 0],
        dinov2_model(x2)[:, 0],
    )

    return psnr.item(), ssim.item(), dinov2.item()  # type: ignore


def ensure_similar_images(
    img_1: Image.Image,
    img_2: Image.Image,
    min_psnr: int = 45,
    min_ssim: float = 0.99,
    min_dinov2: float = 0.99,
) -> None:
    psnr, ssim, dinov2 = compare_images(img_1, img_2)
    if (psnr < min_psnr) or (ssim < min_ssim) or (dinov2 < min_dinov2):
        raise AssertionError(
            dedent(f"""
            Images are not similar enough!
              - PSNR: {psnr:08.05f} (required at least {min_psnr:08.05f})
              - SSIM: {ssim:08.06f} (required at least {min_ssim:08.06f})
              - DINO: {dinov2:08.06f} (required at least {min_dinov2:08.06f})
            """).strip()
        )


class T5TextEmbedder(nn.Module):
    def __init__(
        self,
        pretrained_path: Path | str,
        max_length: int | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()  # type: ignore[reportUnknownMemberType]
        self.model: nn.Module = T5EncoderModel.from_pretrained(  # type: ignore
            pretrained_path,
            local_files_only=local_files_only,
        )
        self.tokenizer: transformers.T5Tokenizer = T5Tokenizer.from_pretrained(  # type: ignore
            pretrained_path,
            local_files_only=local_files_only,
        )
        self.max_length = max_length

    def forward(
        self,
        caption: str,
        text_input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        max_length: int | None = None,
    ) -> torch.Tensor:
        if max_length is None:
            max_length = self.max_length

        if text_input_ids is None or attention_mask is None:
            if max_length is not None:
                text_inputs = self.tokenizer(  # type: ignore
                    caption,
                    return_tensors="pt",
                    add_special_tokens=True,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                )
            else:
                text_inputs = self.tokenizer(caption, return_tensors="pt", add_special_tokens=True)  # type: ignore
            _text_input_ids: torch.Tensor = text_inputs.input_ids.to(self.model.device)  # type: ignore
            _attention_mask: torch.Tensor = text_inputs.attention_mask.to(self.model.device)  # type: ignore
        else:
            _text_input_ids: torch.Tensor = text_input_ids.to(self.model.device)  # type: ignore
            _attention_mask: torch.Tensor = attention_mask.to(self.model.device)  # type: ignore

        outputs = self.model(_text_input_ids, attention_mask=_attention_mask)

        embeddings = outputs.last_hidden_state
        return embeddings
