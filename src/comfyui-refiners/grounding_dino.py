from typing import Any, Sequence

import torch
from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor  # type: ignore

from refiners.fluxion.utils import no_grad, str_to_dtype, tensor_to_image

from .utils import BoundingBox


class LoadGroundingDino:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "checkpoint": ("PATH", {}),
                "dtype": (
                    "STRING",
                    {
                        "default": "float32",
                    },
                ),
                "device": (
                    "STRING",
                    {
                        "default": "cuda",
                    },
                ),
            }
        }

    RETURN_TYPES = ("PROCESSOR", "MODEL")
    RETURN_NAMES = ("processor", "model")
    DESCRIPTION = "Load a grounding dino model."
    CATEGORY = "Refiners/Solutions"
    FUNCTION = "load"

    def load(
        self,
        checkpoint: str,
        dtype: str,
        device: str,
    ) -> tuple[GroundingDinoProcessor, GroundingDinoForObjectDetection]:
        """Load a grounding dino model.

        Args:
            checkpoint: The path to the checkpoint folder.
            dtype: The torch data type to use.
            device: The torch device to load the model on.

        Returns:
            The grounding dino processor and model instances.
        """
        processor = GroundingDinoProcessor.from_pretrained(checkpoint)  # type: ignore
        assert isinstance(processor, GroundingDinoProcessor)

        model = GroundingDinoForObjectDetection.from_pretrained(checkpoint, torch_dtype=str_to_dtype(dtype))  # type: ignore
        model = model.to(device=device)  # type: ignore
        assert isinstance(model, GroundingDinoForObjectDetection)

        return (processor, model)


# NOTE: not yet natively supported in Refiners, hence the transformers dependency
class GroundingDino:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "processor": ("PROCESSOR", {}),
                "model": ("MODEL", {}),
                "image": ("IMAGE", {}),
                "prompt": ("STRING", {}),
                "box_threshold": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "text_threshold": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
            },
        }

    RETURN_TYPES = ("BOUNDING_BOX",)
    RETURN_NAMES = ("bbox",)
    DESCRIPTION = "Detect an object in an image using a GroundingDino model."
    CATEGORY = "Refiners/Solutions"
    FUNCTION = "process"

    @staticmethod
    def corners_to_pixels_format(
        bboxes: torch.Tensor,
        width: int,
        height: int,
    ) -> torch.Tensor:
        x1, y1, x2, y2 = bboxes.round().to(torch.int32).unbind(-1)
        return torch.stack(
            tensors=(
                x1.clamp_(0, width),
                y1.clamp_(0, height),
                x2.clamp_(0, width),
                y2.clamp_(0, height),
            ),
            dim=-1,
        )

    @staticmethod
    def bbox_union(bboxes: Sequence[list[int]]) -> BoundingBox | None:
        if not bboxes:
            return None
        for bbox in bboxes:
            assert len(bbox) == 4
            assert all(isinstance(x, int) for x in bbox)
        return (
            min(bbox[0] for bbox in bboxes),
            min(bbox[1] for bbox in bboxes),
            max(bbox[2] for bbox in bboxes),
            max(bbox[3] for bbox in bboxes),
        )

    @no_grad()
    def process(
        self,
        processor: GroundingDinoProcessor,
        model: GroundingDinoForObjectDetection,
        image: torch.Tensor,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> tuple[BoundingBox]:
        """Detect an object in an image using a GroundingDino model and a text prompt.

        Args:
            processor: The image processor to use.
            model: The grounding dino model to use.
            image: The input image to detect in.
            prompt: The text prompt of what to detect in the image.
            box_threshold: The score threshold for the bounding boxes.
            text_threshold: The score threshold for the text.

        Returns:
            The union of the bounding boxes found in the image.
        """
        # prepare the inputs
        pil_image = tensor_to_image(image.permute(0, 3, 1, 2))

        # NOTE: queries must be in lower cas + end with a dot. See:
        # https://github.com/IDEA-Research/GroundingDINO/blob/856dde2/groundingdino/util/inference.py#L22-L26
        inputs = processor(images=pil_image, text=f"{prompt.lower()}.", return_tensors="pt").to(device=model.device)

        # get the model's prediction
        outputs = model(**inputs)

        # post-process the model's prediction
        results: dict[str, Any] = processor.post_process_grounded_object_detection(  # type: ignore
            outputs=outputs,
            input_ids=inputs["input_ids"],
            target_sizes=[(pil_image.height, pil_image.width)],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )[0]

        # retrieve the bounding boxes
        assert "boxes" in results
        bboxes = results["boxes"].cpu()  # type: ignore
        assert isinstance(bboxes, torch.Tensor)
        assert bboxes.shape[0] != 0, "No bounding boxes found. Try adjusting the thresholds or pick another prompt."
        bboxes = self.corners_to_pixels_format(bboxes, pil_image.width, pil_image.height)  # type: ignore

        # compute the union of the bounding boxes
        bbox = self.bbox_union(bboxes.numpy().tolist())  # type: ignore
        assert bbox is not None

        return (bbox,)


NODE_CLASS_MAPPINGS: dict[str, Any] = {
    "GroundingDino": GroundingDino,
    "LoadGroundingDino": LoadGroundingDino,
}
