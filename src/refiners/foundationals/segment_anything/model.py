from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor, device as Device, dtype as DType

import refiners.fluxion.layers as fl
from refiners.fluxion.utils import interpolate, no_grad, normalize, pad
from refiners.foundationals.segment_anything.image_encoder import SAMViT, SAMViTH
from refiners.foundationals.segment_anything.mask_decoder import MaskDecoder
from refiners.foundationals.segment_anything.prompt_encoder import MaskEncoder, PointEncoder


@dataclass
class ImageEmbedding:
    features: Tensor
    original_image_size: tuple[int, int]  # (height, width)


class SegmentAnything(fl.Module):
    """SegmentAnything model.

    See [[arXiv:2304.02643] Segment Anything](https://arxiv.org/abs/2304.02643)

    Attributes:
        mask_threshold (float): 0.0
    """

    mask_threshold: float = 0.0

    def __init__(
        self,
        image_encoder: SAMViT,
        point_encoder: PointEncoder,
        mask_encoder: MaskEncoder,
        mask_decoder: MaskDecoder,
        device: Device | str = "cpu",
        dtype: DType = torch.float32,
    ) -> None:
        """Initialize SegmentAnything model.

        Args:
            image_encoder: The image encoder to use.
            point_encoder: The point encoder to use.
            mask_encoder: The mask encoder to use.
            mask_decoder: The mask decoder to use.
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """
        super().__init__()
        self.device: Device = device if isinstance(device, Device) else Device(device=device)
        self.dtype = dtype
        self.image_encoder = image_encoder.to(device=self.device, dtype=self.dtype)
        self.point_encoder = point_encoder.to(device=self.device, dtype=self.dtype)
        self.mask_encoder = mask_encoder.to(device=self.device, dtype=self.dtype)
        self.mask_decoder = mask_decoder.to(device=self.device, dtype=self.dtype)

    @no_grad()
    def compute_image_embedding(self, image: Image.Image) -> ImageEmbedding:
        """Compute the emmbedding of an image.

        Args:
            image: The image to compute the embedding of.

        Returns:
            The computed image embedding.
        """
        original_size = (image.height, image.width)
        target_size = self.compute_target_size(original_size)
        return ImageEmbedding(
            features=self.image_encoder(self.preprocess_image(image=image, target_size=target_size)),
            original_image_size=original_size,
        )

    @no_grad()
    def predict(
        self,
        input: Image.Image | ImageEmbedding,
        foreground_points: Sequence[tuple[float, float]] | None = None,
        background_points: Sequence[tuple[float, float]] | None = None,
        box_points: Sequence[Sequence[tuple[float, float]]] | None = None,
        low_res_mask: Float[Tensor, "1 1 256 256"] | None = None,
        binarize: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Predict the masks of the input image.

        Args:
            input: The input image or its embedding.
            foreground_points: The points of the foreground.
            background_points: The points of the background.
            box_points: The points of the box.
            low_res_mask: The low resolution mask.
            binarize: Whether to binarize the masks.

        Returns:
            The predicted masks.
            The IOU prediction.
            The low resolution masks.
        """
        if isinstance(input, ImageEmbedding):
            original_size = input.original_image_size
            target_size = self.compute_target_size(original_size)
            image_embedding = input.features
        else:
            original_size = (input.height, input.width)
            target_size = self.compute_target_size(original_size)
            image_embedding = self.image_encoder(self.preprocess_image(image=input, target_size=target_size))

        coordinates, type_mask = self.point_encoder.points_to_tensor(
            foreground_points=foreground_points,
            background_points=background_points,
            box_points=box_points,
        )
        self.point_encoder.set_type_mask(type_mask=type_mask)

        if low_res_mask is not None:
            mask_embedding = self.mask_encoder(low_res_mask)
        else:
            mask_embedding = self.mask_encoder.get_no_mask_dense_embedding(
                image_embedding_size=self.image_encoder.image_embedding_size
            )

        point_embedding = self.point_encoder(
            self.normalize(coordinates, target_size=target_size, original_size=original_size)
        )
        dense_positional_embedding = self.point_encoder.get_dense_positional_embedding(
            image_embedding_size=self.image_encoder.image_embedding_size
        )

        self.mask_decoder.set_image_embedding(image_embedding=image_embedding)
        self.mask_decoder.set_mask_embedding(mask_embedding=mask_embedding)
        self.mask_decoder.set_point_embedding(point_embedding=point_embedding)
        self.mask_decoder.set_dense_positional_embedding(dense_positional_embedding=dense_positional_embedding)

        low_res_masks, iou_predictions = self.mask_decoder()

        high_res_masks = self.postprocess_masks(
            masks=low_res_masks, target_size=target_size, original_size=original_size
        )

        if binarize:
            high_res_masks = high_res_masks > self.mask_threshold

        return high_res_masks, iou_predictions, low_res_masks

    @property
    def image_size(self) -> int:
        """The image size."""
        w, h = self.image_encoder.image_size
        assert w == h
        return w

    def compute_target_size(self, size: tuple[int, int]) -> tuple[int, int]:
        """Compute the target size as expected by the image encoder.

        Args:
            size: The size of the input image.

        Returns:
            The target height.
            The target width.
        """
        oldh, oldw = size
        scale = self.image_size * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def preprocess_image(self, image: Image.Image, target_size: tuple[int, int]) -> Tensor:
        """Preprocess an image without distorting its aspect ratio.

        Args:
            image: The image to preprocess.
            target_size: The target size.

        Returns:
            The preprocessed image.
        """
        h, w = target_size
        padh = self.image_size - h
        padw = self.image_size - w
        image_tensor = torch.tensor(
            np.array(image.resize((w, h), resample=Image.Resampling.BILINEAR)).astype(np.float32).transpose(2, 0, 1),
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0)
        return pad(
            normalize(image_tensor, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]), (0, padw, 0, padh)
        )

    def normalize(self, coordinates: Tensor, target_size: tuple[int, int], original_size: tuple[int, int]) -> Tensor:
        """Normalize the coordinates.

        Args:
            coordinates: The coordinates to normalize.
            target_size: The target size.
            original_size: The original size.

        Returns:
            The normalized coordinates.
        """
        coordinates[:, :, 0] = ((coordinates[:, :, 0] * (target_size[1] / original_size[1])) + 0.5) / self.image_size
        coordinates[:, :, 1] = ((coordinates[:, :, 1] * (target_size[0] / original_size[0])) + 0.5) / self.image_size
        return coordinates

    def postprocess_masks(self, masks: Tensor, target_size: tuple[int, int], original_size: tuple[int, int]) -> Tensor:
        """Postprocess the masks.

        Args:
            masks: The masks to postprocess.
            target_size: The target size.
            original_size: The original size.

        Returns:
            The postprocessed masks.
        """
        masks = interpolate(masks, factor=torch.Size((self.image_size, self.image_size)), mode="bilinear")
        masks = masks[..., : target_size[0], : target_size[1]]  # remove padding added at `preprocess_image` time
        masks = interpolate(masks, factor=torch.Size(original_size), mode="bilinear")
        return masks


class SegmentAnythingH(SegmentAnything):
    """SegmentAnything huge model."""

    def __init__(
        self,
        image_encoder: SAMViTH | None = None,
        point_encoder: PointEncoder | None = None,
        mask_encoder: MaskEncoder | None = None,
        mask_decoder: MaskDecoder | None = None,
        device: Device | str = "cpu",
        dtype: DType = torch.float32,
    ) -> None:
        """Initialize SegmentAnything huge model.

        Args:
            image_encoder: The image encoder to use.
            point_encoder: The point encoder to use.
            mask_encoder: The mask encoder to use.
            mask_decoder: The mask decoder to use.
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """
        image_encoder = image_encoder or SAMViTH()
        point_encoder = point_encoder or PointEncoder()
        mask_encoder = mask_encoder or MaskEncoder()
        mask_decoder = mask_decoder or MaskDecoder()

        super().__init__(
            image_encoder=image_encoder,
            point_encoder=point_encoder,
            mask_encoder=mask_encoder,
            mask_decoder=mask_decoder,
            device=device,
            dtype=dtype,
        )
