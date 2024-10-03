from typing import Any

import torch
from PIL import ImageDraw

from refiners.fluxion.utils import image_to_tensor, tensor_to_image

BoundingBox = tuple[int, int, int, int]


class DrawBoundingBox:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {}),
                "bbox": ("BOUNDING_BOX", {}),
                "color": ("STRING", {"default": "red"}),
                "width": ("INT", {"default": 3}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = "Draw a bounding box on an image."
    CATEGORY = "Refiners/Helpers"
    FUNCTION = "process"

    def process(
        self,
        image: torch.Tensor,
        bbox: BoundingBox,
        color: str,
        width: int,
    ) -> tuple[torch.Tensor]:
        """Draw a bounding box on an image.

        Args:
            image: The image to draw on.
            bbox: The bounding box to draw.
            color: The color of the bounding box.
            width: The width of the bounding box.
        """
        pil_image = tensor_to_image(image.permute(0, 3, 1, 2))
        draw = ImageDraw.Draw(pil_image)
        draw.rectangle(bbox, outline=color, width=width)
        image = image_to_tensor(pil_image).permute(0, 2, 3, 1)
        return (image,)


NODE_CLASS_MAPPINGS: dict[str, Any] = {
    "DrawBoundingBox": DrawBoundingBox,
}
