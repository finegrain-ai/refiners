from typing import Any

import torch

from refiners.fluxion.utils import image_to_tensor, no_grad, tensor_to_image
from refiners.solutions import BoxSegmenter as _BoxSegmenter
from refiners.solutions.box_segmenter import BoundingBox


class LoadBoxSegmenter:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "checkpoint": ("PATH", {}),
                "margin": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "device": ("STRING", {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    DESCRIPTION = "Load a BoxSegmenter refiners model."
    CATEGORY = "Refiners/Solutions"
    FUNCTION = "load"

    def load(
        self,
        checkpoint: str,
        margin: float,
        device: str,
    ) -> tuple[_BoxSegmenter]:
        """Load a BoxSegmenter refiners model.

        Args:
            checkpoint: The path to the checkpoint file.
            margin: The bbox margin to use when processing images.
            device: The torch device to load the model on.

        Returns:
            A BoxSegmenter model instance.
        """
        return (
            _BoxSegmenter(
                weights=checkpoint,
                margin=margin,
                device=device,
            ),
        )


class BoxSegmenter:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL", {}),
                "image": ("IMAGE", {}),
            },
            "optional": {
                "bbox": ("BOUNDING_BOX", {}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    DESCRIPTION = "Segment an image using a BoxSegmenter model and a bbox."
    CATEGORY = "Refiners/Solutions"
    FUNCTION = "process"

    @no_grad()
    def process(
        self,
        model: _BoxSegmenter,
        image: torch.Tensor,
        bbox: BoundingBox | None = None,
    ) -> tuple[torch.Tensor]:
        """Segment an image using a BoxSegmenter model and a bbox.

        Args:
            model: The BoxSegmenter model to use.
            image: The input image to process.
            bbox: Where in the image to apply the model.

        Returns:
            The mask of the segmented object.
        """
        pil_image = tensor_to_image(image.permute(0, 3, 1, 2))
        mask = model(img=pil_image, box_prompt=bbox)
        mask_tensor = image_to_tensor(mask).squeeze(1)
        return (mask_tensor,)


NODE_CLASS_MAPPINGS: dict[str, Any] = {
    "BoxSegmenter": BoxSegmenter,
    "LoadBoxSegmenter": LoadBoxSegmenter,
}
