from pathlib import Path

import numpy as np
import piq  # type: ignore
import torch
import torch.nn as nn
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer  # type: ignore


def compare_images(img_1: Image.Image, img_2: Image.Image) -> tuple[int, float]:
    x1, x2 = (
        torch.tensor(np.array(x).astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0 for x in (img_1, img_2)
    )
    return (piq.psnr(x1, x2), piq.ssim(x1, x2).item())  # type: ignore


def ensure_similar_images(img_1: Image.Image, img_2: Image.Image, min_psnr: int = 45, min_ssim: float = 0.99):
    psnr, ssim = compare_images(img_1, img_2)
    assert (psnr >= min_psnr) and (
        ssim >= min_ssim
    ), f"PSNR {psnr} / SSIM {ssim}, expected at least {min_psnr} / {min_ssim}"


class T5TextEmbedder(nn.Module):
    def __init__(
        self, pretrained_path: Path = Path("tests/weights/QQGYLab/T5XLFP16"), max_length: int | None = None
    ) -> None:
        super().__init__()  # type: ignore[reportUnknownMemberType]
        self.model: nn.Module = T5EncoderModel.from_pretrained(pretrained_path, local_files_only=True)  # type: ignore
        self.tokenizer: transformers.T5Tokenizer = T5Tokenizer.from_pretrained(pretrained_path, local_files_only=True)  # type: ignore
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
