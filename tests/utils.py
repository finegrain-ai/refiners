import numpy as np
import piq  # type: ignore
import torch
from PIL import Image


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
