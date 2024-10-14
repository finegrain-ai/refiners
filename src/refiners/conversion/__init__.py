import sys
from importlib import import_module
from importlib.metadata import requires

from packaging.requirements import Requirement

refiners_requires = requires("refiners")
assert refiners_requires is not None

# Some dependencies have different module names than their package names
req_to_module: dict[str, str] = {
    "huggingface-hub": "huggingface_hub",
    "segment-anything-py": "segment_anything",
}

for dep in refiners_requires:
    req = Requirement(dep)
    marker = req.marker
    if marker is None or not marker.evaluate({"extra": "conversion"}):
        continue

    module_name = req_to_module.get(req.name, req.name)

    try:
        import_module(module_name)
    except ImportError:
        print(
            f"Some dependencies are missing: {req.name}. "
            "Please install refiners with the `conversion` extra, e.g. `pip install refiners[conversion]`",
            file=sys.stderr,
        )
        sys.exit(1)

from .models import (
    autoencoder_sd15,
    autoencoder_sdxl,
    clip_image_sd21,
    clip_text_sd15,
    clip_text_sdxl,
    controllora_sdxl,
    controlnet_sd15,
    dinov2,
    ella,
    hq_sam,
    ipadapter_sd15,
    ipadapter_sdxl,
    loras,
    mvanet,
    preprocessors,
    sam,
    t2iadapter_sd15,
    t2iadapter_sdxl,
    unet_sd15,
    unet_sdxl,
)

__all__ = [
    "autoencoder_sd15",
    "autoencoder_sdxl",
    "clip_image_sd21",
    "clip_text_sd15",
    "clip_text_sdxl",
    "controllora_sdxl",
    "controlnet_sd15",
    "dinov2",
    "ella",
    "hq_sam",
    "ipadapter_sd15",
    "ipadapter_sdxl",
    "loras",
    "mvanet",
    "preprocessors",
    "sam",
    "t2iadapter_sd15",
    "t2iadapter_sdxl",
    "unet_sd15",
    "unet_sdxl",
]
