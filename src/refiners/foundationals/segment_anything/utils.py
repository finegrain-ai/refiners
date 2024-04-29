from PIL import Image
from torch import Size, Tensor, device as Device, dtype as DType

from refiners.fluxion.utils import image_to_tensor, interpolate, normalize, pad


def compute_scaled_size(size: tuple[int, int], image_encoder_resolution: int) -> tuple[int, int]:
    """Compute the scaled size as expected by the image encoder.
    This computed size keep the ratio of the input image, and scale it to fit inside the square (image_encoder_resolution, image_encoder_resolution) of image encoder.

    Args:
        size: The size (h, w) of the input image.
        image_encoder_resolution: Image encoder resolution.

    Returns:
        The target height.
        The target width.
    """
    oldh, oldw = size
    scale = image_encoder_resolution * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def image_to_scaled_tensor(
    image: Image.Image, scaled_size: tuple[int, int], device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    """Resize the image to `scaled_size` and convert it to a tensor.

    Args:
        image: The image.
        scaled_size: The target size (h, w).
        device: Tensor device.
        dtype: Tensor dtype.
    Returns:
        a Tensor of shape (1, c, h, w)
    """
    h, w = scaled_size
    resized = image.resize((w, h), resample=Image.Resampling.BILINEAR)  # type: ignore

    return image_to_tensor(resized, device=device, dtype=dtype) * 255.0


def preprocess_image(
    image: Image.Image, image_encoder_resolution: int, device: Device | None = None, dtype: DType | None = None
) -> Tensor:
    """Preprocess an image without distorting its aspect ratio.

    Args:
        image: The image to preprocess before calling the image encoder.
        image_encoder_resolution: Image encoder resolution.
        device: Tensor device (None by default).
        dtype: Tensor dtype (None by default).

    Returns:
        The preprocessed image.
    """

    scaled_size = compute_scaled_size((image.height, image.width), image_encoder_resolution)

    image_tensor = image_to_scaled_tensor(image, scaled_size, device=device, dtype=dtype)

    return pad_image_tensor(
        normalize(image_tensor, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        scaled_size,
        image_encoder_resolution,
    )


def pad_image_tensor(image_tensor: Tensor, scaled_size: tuple[int, int], image_encoder_resolution: int) -> Tensor:
    """Pad an image with zeros to make it square.

    Args:
        image_tensor: The image tensor to pad.
        scaled_size: The scaled size (h, w).
        image_encoder_resolution: Image encoder resolution.

    Returns:
        The padded image.
    """
    assert len(image_tensor.shape) == 4
    assert image_tensor.shape[2] <= image_encoder_resolution
    assert image_tensor.shape[3] <= image_encoder_resolution

    h, w = scaled_size
    padh = image_encoder_resolution - h
    padw = image_encoder_resolution - w
    return pad(image_tensor, (0, padw, 0, padh))


def postprocess_masks(low_res_masks: Tensor, original_size: tuple[int, int], image_encoder_resolution: int) -> Tensor:
    """Postprocess the masks to fit the original image size and remove zero-padding (if any).

    Args:
        low_res_masks: The masks to postprocess.
        original_size: The original size (h, w).
        image_encoder_resolution: Image encoder resolution.

    Returns:
        The postprocessed masks.
    """
    scaled_size = compute_scaled_size(original_size, image_encoder_resolution)
    masks = interpolate(low_res_masks, size=Size((image_encoder_resolution, image_encoder_resolution)), mode="bilinear")
    masks = masks[..., : scaled_size[0], : scaled_size[1]]  # remove padding added at `preprocess_image` time
    masks = interpolate(masks, size=Size(original_size), mode="bilinear")
    return masks


def normalize_coordinates(coordinates: Tensor, original_size: tuple[int, int], image_encoder_resolution: int) -> Tensor:
    """Normalize the coordinates in the [0,1] range

    Args:
        coordinates: The coordinates to normalize.
        original_size: The original image size.
        image_encoder_resolution: Image encoder resolution.

    Returns:
        The normalized coordinates.
    """
    scaled_size = compute_scaled_size(original_size, image_encoder_resolution)
    coordinates[:, :, 0] = (
        (coordinates[:, :, 0] * (scaled_size[1] / original_size[1])) + 0.5
    ) / image_encoder_resolution
    coordinates[:, :, 1] = (
        (coordinates[:, :, 1] * (scaled_size[0] / original_size[0])) + 0.5
    ) / image_encoder_resolution
    return coordinates
