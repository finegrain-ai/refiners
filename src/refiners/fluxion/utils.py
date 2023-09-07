from typing import Iterable, Literal, TypeVar
from PIL import Image
from numpy import array, float32
from pathlib import Path
from safetensors import safe_open as _safe_open  # type: ignore
from safetensors.torch import save_file as _save_file  # type: ignore
from torch import norm as _norm, manual_seed as _manual_seed  # type: ignore
import torch
from torch.nn.functional import pad as _pad, interpolate as _interpolate  # type: ignore
from torch import Tensor, device as Device, dtype as DType


T = TypeVar("T")
E = TypeVar("E")


def norm(x: Tensor) -> Tensor:
    return _norm(x)  # type: ignore


def manual_seed(seed: int) -> None:
    _manual_seed(seed)


def pad(x: Tensor, pad: Iterable[int], value: float = 0.0) -> Tensor:
    return _pad(input=x, pad=pad, value=value)  # type: ignore


def interpolate(x: Tensor, factor: float | torch.Size, mode: str = "nearest") -> Tensor:
    return (
        _interpolate(x, scale_factor=factor, mode=mode)
        if isinstance(factor, float | int)
        else _interpolate(x, size=factor, mode=mode)
    )  # type: ignore


def image_to_tensor(image: Image.Image, device: Device | str | None = None, dtype: DType | None = None) -> Tensor:
    return torch.tensor(array(image).astype(float32).transpose(2, 0, 1) / 255.0, device=device, dtype=dtype).unsqueeze(
        0
    )


def tensor_to_image(tensor: Tensor) -> Image.Image:
    return Image.fromarray((tensor.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))  # type: ignore


def safe_open(
    path: Path | str,
    framework: Literal["pytorch", "tensorflow", "flax", "numpy"],
    device: Device | str = "cpu",
) -> dict[str, Tensor]:
    framework_mapping = {
        "pytorch": "pt",
        "tensorflow": "tf",
        "flax": "flax",
        "numpy": "numpy",
    }
    return _safe_open(str(path), framework=framework_mapping[framework], device=str(device))  # type: ignore


def load_from_safetensors(path: Path | str, device: Device | str = "cpu") -> dict[str, Tensor]:
    with safe_open(path=path, framework="pytorch", device=device) as tensors:  # type: ignore
        return {key: tensors.get_tensor(key) for key in tensors.keys()}  # type: ignore


def load_metadata_from_safetensors(path: Path | str) -> dict[str, str] | None:
    with safe_open(path=path, framework="pytorch") as tensors:  # type: ignore
        return tensors.metadata()  # type: ignore


def save_to_safetensors(path: Path | str, tensors: dict[str, Tensor], metadata: dict[str, str] | None = None) -> None:
    _save_file(tensors, path, metadata)  # type: ignore
