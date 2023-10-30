import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal, TypeVar

import torch
from PIL import Image
from jaxtyping import Float
from numpy import array, float32
from safetensors import safe_open as _safe_open  # type: ignore
from safetensors.torch import save_file as _save_file  # type: ignore
from torch import Tensor, device as Device, dtype as DType
from torch import norm as _norm, manual_seed as _manual_seed  # type: ignore
from torch.nn.functional import pad as _pad, interpolate as _interpolate, conv2d  # type: ignore

T = TypeVar("T")
E = TypeVar("E")


def norm(x: Tensor) -> Tensor:
    return _norm(x)  # type: ignore


def manual_seed(seed: int) -> None:
    _manual_seed(seed)


def pad(x: Tensor, pad: Iterable[int], value: float = 0.0, mode: str = "constant") -> Tensor:
    return _pad(input=x, pad=pad, value=value, mode=mode)  # type: ignore


def interpolate(x: Tensor, factor: float | torch.Size, mode: str = "nearest") -> Tensor:
    return (
        _interpolate(x, scale_factor=factor, mode=mode)
        if isinstance(factor, float | int)
        else _interpolate(x, size=factor, mode=mode)
    )  # type: ignore


# Adapted from https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py
def normalize(
    tensor: Float[Tensor, "*batch channels height width"], mean: list[float], std: list[float]
) -> Float[Tensor, "*batch channels height width"]:
    assert tensor.is_floating_point()
    assert tensor.ndim >= 3

    dtype = tensor.dtype
    pixel_mean = torch.tensor(mean, dtype=dtype, device=tensor.device).view(-1, 1, 1)
    pixel_std = torch.tensor(std, dtype=dtype, device=tensor.device).view(-1, 1, 1)
    if (pixel_std == 0).any():
        raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")

    return (tensor - pixel_mean) / pixel_std


# Adapted from https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py
def gaussian_blur(
    tensor: Float[Tensor, "*batch channels height width"],
    kernel_size: int | tuple[int, int],
    sigma: float | tuple[float, float] | None = None,
) -> Float[Tensor, "*batch channels height width"]:
    assert torch.is_floating_point(tensor)

    def get_gaussian_kernel1d(kernel_size: int, sigma: float) -> Float[Tensor, "kernel_size"]:
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()
        return kernel1d

    def get_gaussian_kernel2d(
        kernel_size_x: int, kernel_size_y: int, sigma_x: float, sigma_y: float, dtype: DType, device: Device
    ) -> Float[Tensor, "kernel_size_y kernel_size_x"]:
        kernel1d_x = get_gaussian_kernel1d(kernel_size_x, sigma_x).to(device, dtype=dtype)
        kernel1d_y = get_gaussian_kernel1d(kernel_size_y, sigma_y).to(device, dtype=dtype)
        kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
        return kernel2d

    def default_sigma(kernel_size: int) -> float:
        return kernel_size * 0.15 + 0.35

    if isinstance(kernel_size, int):
        kx, ky = kernel_size, kernel_size
    else:
        kx, ky = kernel_size

    if sigma is None:
        sx, sy = default_sigma(kx), default_sigma(ky)
    elif isinstance(sigma, float):
        sx, sy = sigma, sigma
    else:
        assert isinstance(sigma, tuple)
        sx, sy = sigma

    channels = tensor.shape[-3]
    kernel = get_gaussian_kernel2d(kx, ky, sx, sy, dtype=tensor.dtype, device=tensor.device)
    kernel = kernel.expand(channels, 1, kernel.shape[0], kernel.shape[1])

    # pad = (left, right, top, bottom)
    tensor = pad(tensor, pad=(kx // 2, kx // 2, ky // 2, ky // 2), mode="reflect")
    tensor = conv2d(tensor, weight=kernel, groups=channels)

    return tensor


def image_to_tensor(image: Image.Image, device: Device | str | None = None, dtype: DType | None = None) -> Tensor:
    """
    Convert a PIL Image to a Tensor.

    If the image is in mode `RGB` the tensor will have shape `[3, H, W]`, otherwise
    `[1, H, W]` for mode `L` (grayscale) or `[4, H, W]` for mode `RGBA`.

    Values are clamped to the range `[0, 1]`.
    """
    image_tensor = torch.tensor(array(image).astype(float32) / 255.0, device=device, dtype=dtype)

    match image.mode:
        case "L":
            image_tensor = image_tensor.unsqueeze(0)
        case "RGBA" | "RGB":
            image_tensor = image_tensor.permute(2, 0, 1)
        case _:
            raise ValueError(f"Unsupported image mode: {image.mode}")

    return image_tensor.unsqueeze(0)


def tensor_to_image(tensor: Tensor) -> Image.Image:
    """
    Convert a Tensor to a PIL Image.

    The tensor must have shape `[1, channels, height, width]` where the number of
    channels is either 1 (grayscale) or 3 (RGB) or 4 (RGBA).

    Expected values are in the range `[0, 1]` and are clamped to this range.
    """
    assert tensor.ndim == 4 and tensor.shape[0] == 1, f"Unsupported tensor shape: {tensor.shape}"
    num_channels = tensor.shape[1]
    tensor = tensor.clamp(0, 1).squeeze(0)

    match num_channels:
        case 1:
            tensor = tensor.squeeze(0)
        case 3 | 4:
            tensor = tensor.permute(1, 2, 0)
        case _:
            raise ValueError(f"Unsupported number of channels: {num_channels}")

    return Image.fromarray((tensor.cpu().numpy() * 255).astype("uint8"))  # type: ignore[reportUnknownType]


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
        state_dict = {key: tensors.get_tensor(key) for key in tensors.keys()}  # type: ignore
    state_dict = autotranslate_state_dict(state_dict, state_dict_origin_path=path)
    return state_dict


@lru_cache(maxsize=None)
def load_state_dict_conversion_maps():
    conversion_maps = {}
    from importlib.resources import files

    for file in files("refiners").joinpath("fluxion/conversion_maps").iterdir():
        if file.is_file() and file.suffix == ".json":
            conversion_maps[file.name] = json.loads(file.read_text())
    return conversion_maps


def key_prefixes(keys):
    return set(k.rsplit(".", 1)[0] for k in keys)


def autotranslate_state_dict(state_dict, state_dict_origin_path=""):
    """Autotranslate state dict from one module type to another."""

    state_dict_prefixes = key_prefixes(state_dict.keys())
    conversion_maps = load_state_dict_conversion_maps()
    print(f"Finding conversion for {state_dict_origin_path}")
    for conversion_map_name, conversion_map in conversion_maps.items():
        conversion_map_prefixes = (
            set(conversion_map["mapping"].keys())
            | set(conversion_map["source_aliases"].keys())
            | set(conversion_map["ignorable_prefixes"])
        )
        if state_dict_prefixes.issubset(conversion_map_prefixes):
            print(f"  Using conversion map {conversion_map_name} for {state_dict_origin_path}")
            return convert_state_dict(
                source_state_dict=state_dict,
                state_dict_mapping=conversion_map["mapping"],
                source_aliases=conversion_map["source_aliases"],
            )

    return state_dict


def convert_state_dict(
    source_state_dict: dict[str, Tensor], state_dict_mapping: dict[str, str], source_aliases: dict[str, str]
) -> dict[str, Tensor]:
    converted_state_dict: dict[str, Tensor] = {}

    for source_key in source_state_dict:
        source_prefix, suffix = source_key.rsplit(sep=".", maxsplit=1)
        # handle aliases
        source_prefix = source_aliases.get(source_prefix, source_prefix)
        try:
            target_prefix = state_dict_mapping[source_prefix]
        except KeyError:
            continue
        target_key = ".".join([target_prefix, suffix])
        converted_state_dict[target_key] = source_state_dict[source_key]

    return converted_state_dict


def load_metadata_from_safetensors(path: Path | str) -> dict[str, str] | None:
    with safe_open(path=path, framework="pytorch") as tensors:  # type: ignore
        return tensors.metadata()  # type: ignore


def save_to_safetensors(path: Path | str, tensors: dict[str, Tensor], metadata: dict[str, str] | None = None) -> None:
    _save_file(tensors, path, metadata)  # type: ignore


def summarize_tensor(tensor: torch.Tensor, /) -> str:
    return (
        "Tensor("
        + ", ".join(
            [
                f"shape=({', '.join(map(str, tensor.shape))})",
                f"dtype={str(object=tensor.dtype).removeprefix('torch.')}",
                f"device={tensor.device}",
                f"min={tensor.min():.2f}",  # type: ignore
                f"max={tensor.max():.2f}",  # type: ignore
                f"mean={tensor.mean():.2f}",
                f"std={tensor.std():.2f}",
                f"norm={norm(x=tensor):.2f}",
                f"grad={tensor.requires_grad}",
            ]
        )
        + ")"
    )


def get_cache_dir():
    xdg_cache_home = os.getenv("XDG_CACHE_HOME", None)
    if xdg_cache_home is None:
        user_home = os.getenv("HOME", None)
        if user_home:
            xdg_cache_home = os.path.join(user_home, ".cache")

    if xdg_cache_home is not None:
        return os.path.join(xdg_cache_home, "refiners")

    return os.path.join(os.path.dirname(__file__), ".cached-aimg")


def download_diffusers_weights(repo, sub, filename):
    url = f"https://huggingface.co/{repo}/resolve/main/{sub}/{filename}"
    dest = f"{get_cache_dir()}/{repo}/{sub}/{filename}"

    if os.path.exists(dest):
        return dest

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {url} to {dest}")
    torch.hub.download_url_to_file(url, dest)
    return dest


@lru_cache
def default_device() -> str:
    """Return the best torch backend available."""
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps:0"

    return "cpu"
