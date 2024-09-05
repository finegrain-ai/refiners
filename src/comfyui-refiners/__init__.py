from typing import Any

from .box_segmenter import NODE_CLASS_MAPPINGS as box_segmenter_mappings
from .grounding_dino import NODE_CLASS_MAPPINGS as grounding_dino_mappings
from .huggingface import NODE_CLASS_MAPPINGS as huggingface_mappings
from .utils import NODE_CLASS_MAPPINGS as utils_mappings

NODE_CLASS_MAPPINGS: dict[str, Any] = {}
NODE_CLASS_MAPPINGS.update(box_segmenter_mappings)
NODE_CLASS_MAPPINGS.update(grounding_dino_mappings)
NODE_CLASS_MAPPINGS.update(huggingface_mappings)
NODE_CLASS_MAPPINGS.update(utils_mappings)

NODE_DISPLAY_NAME_MAPPINGS = {k: v.__name__ for k, v in NODE_CLASS_MAPPINGS.items()}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
