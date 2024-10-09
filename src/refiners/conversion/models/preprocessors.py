import torch

from refiners.conversion.utils import Conversion, Hub, WeightRecipe

informative_drawings_recipe = WeightRecipe(
    key_map={
        "model0.1": "Chain_1.Conv2d",
        "model1.0": "Chain_2.Conv2d",
        "model1.3": "Chain_3.Conv2d",
        "model2.0.conv_block.1": "Residual_1.Conv2d_1",
        "model2.0.conv_block.5": "Residual_1.Conv2d_2",
        "model2.1.conv_block.1": "Residual_2.Conv2d_1",
        "model2.1.conv_block.5": "Residual_2.Conv2d_2",
        "model2.2.conv_block.1": "Residual_3.Conv2d_1",
        "model2.2.conv_block.5": "Residual_3.Conv2d_2",
        "model3.0": "Chain_4.ConvTranspose2d",
        "model3.3": "Chain_5.ConvTranspose2d",
        "model4.1": "Chain_6.Conv2d",
    },
)

informative_drawings = Conversion(
    original=Hub(
        repo_id="carolineec/informativedrawings",
        filename="model.pth",
        expected_sha256="30a534781061f34e83bb9406b4335da4ff2616c95d22a585c1245aa8363e74e0",
        download_url="https://huggingface.co/spaces/carolineec/informativedrawings/resolve/main/model2.pth",
    ),
    converted=Hub(
        repo_id="refiners/preprocessor.informativedrawings",
        filename="model.safetensors",
        expected_sha256="0f9a34bfcd95d89aedcc213b8d279ba1bab1279b73d8d009d1632d6276e6fcf3",
    ),
    recipe=informative_drawings_recipe,
    dtype=torch.float32,
)
