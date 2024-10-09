import torch

from refiners.conversion.models.autoencoder_sd15 import civitai_recipe, diffusers_recipe
from refiners.conversion.utils import Conversion, Hub

stability = Conversion(
    original=Hub(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        filename="vae/diffusion_pytorch_model.safetensors",
        revision="91704abbae38a0e1f60d433fb08d7f7d99081d21",
        expected_sha256="1598f3d24932bcfe6634e8b618ea1e30ab1d57f5aad13a6d2de446d2199f2341",
    ),
    converted=Hub(
        repo_id="refiners/sdxl.autoencoder",
        filename="model.safetensors",
        expected_sha256="6534be9990496fcb4086e5cf71e0ceb208b9f5c728823247c6a51e13564c38af",
    ),
    recipe=diffusers_recipe,
    dtype=torch.float16,
)
madebyollin_fp16fix = Conversion(
    original=Hub(
        repo_id="madebyollin/sdxl-vae-fp16-fix",
        filename="diffusion_pytorch_model.safetensors",
        revision="6d1073461cd0b5a6ea4fda10b812e3d9d58a8330",
        expected_sha256="1b909373b28f2137098b0fd9dbc6f97f8410854f31f84ddc9fa04b077b0ace2c",
    ),
    converted=Hub(
        repo_id="refiners/sdxl.autoencoder_fp16fix",
        filename="model.safetensors",
        expected_sha256="ede1e84626900ebeb0e7911814b1ac98e8916327340f411cce2b77e056e84dd3",
    ),
    recipe=diffusers_recipe,
    dtype=torch.float16,
)
juggernautXL_v10 = Conversion(
    original=Hub(
        repo_id="civitai/KandooAi/juggernautXL",
        filename="v10/onefile_fp16.safetensors",
        expected_sha256="d91d35736d8f2be038f760a9b0009a771ecf0a417e9b38c244a84ea4cb9c0c45",
        download_url="https://civitai.com/api/download/models/456194?type=Model&format=SafeTensor&size=full&fp=fp16",
    ),
    converted=Hub(
        repo_id="refiners/sdxl.juggernaut.v10.autoencoder",
        filename="model.safetensors",
        expected_sha256="ede1e84626900ebeb0e7911814b1ac98e8916327340f411cce2b77e056e84dd3",
    ),
    recipe=civitai_recipe,
    dtype=torch.float16,
)
