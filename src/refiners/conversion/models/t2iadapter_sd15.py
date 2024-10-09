import torch

from refiners.conversion.utils import Conversion, Hub, WeightRecipe

diffusers_recipe = WeightRecipe(
    key_map={
        "adapter.conv_in": "Conv2d",
        "adapter.body.0.resnets.0.block1": "StatefulResidualBlocks_1.ResidualBlocks.Chain.ResidualBlock_1.Conv2d_1",
        "adapter.body.0.resnets.1.block1": "StatefulResidualBlocks_1.ResidualBlocks.Chain.ResidualBlock_2.Conv2d_1",
        "adapter.body.0.resnets.0.block2": "StatefulResidualBlocks_1.ResidualBlocks.Chain.ResidualBlock_1.Conv2d_2",
        "adapter.body.0.resnets.1.block2": "StatefulResidualBlocks_1.ResidualBlocks.Chain.ResidualBlock_2.Conv2d_2",
        "adapter.body.1.downsample": "StatefulResidualBlocks_2.ResidualBlocks.Downsample2d",
        "adapter.body.2.downsample": "StatefulResidualBlocks_3.ResidualBlocks.Downsample2d",
        "adapter.body.3.downsample": "StatefulResidualBlocks_4.ResidualBlocks.Downsample2d",
        "adapter.body.1.in_conv": "StatefulResidualBlocks_2.ResidualBlocks.Conv2d",
        "adapter.body.1.resnets.0.block1": "StatefulResidualBlocks_2.ResidualBlocks.Chain.ResidualBlock_1.Conv2d_1",
        "adapter.body.1.resnets.1.block1": "StatefulResidualBlocks_2.ResidualBlocks.Chain.ResidualBlock_2.Conv2d_1",
        "adapter.body.1.resnets.0.block2": "StatefulResidualBlocks_2.ResidualBlocks.Chain.ResidualBlock_1.Conv2d_2",
        "adapter.body.1.resnets.1.block2": "StatefulResidualBlocks_2.ResidualBlocks.Chain.ResidualBlock_2.Conv2d_2",
        "adapter.body.2.in_conv": "StatefulResidualBlocks_3.ResidualBlocks.Conv2d",
        "adapter.body.2.resnets.0.block1": "StatefulResidualBlocks_3.ResidualBlocks.Chain.ResidualBlock_1.Conv2d_1",
        "adapter.body.2.resnets.1.block1": "StatefulResidualBlocks_3.ResidualBlocks.Chain.ResidualBlock_2.Conv2d_1",
        "adapter.body.3.resnets.0.block1": "StatefulResidualBlocks_4.ResidualBlocks.Chain.ResidualBlock_1.Conv2d_1",
        "adapter.body.3.resnets.1.block1": "StatefulResidualBlocks_4.ResidualBlocks.Chain.ResidualBlock_2.Conv2d_1",
        "adapter.body.2.resnets.0.block2": "StatefulResidualBlocks_3.ResidualBlocks.Chain.ResidualBlock_1.Conv2d_2",
        "adapter.body.2.resnets.1.block2": "StatefulResidualBlocks_3.ResidualBlocks.Chain.ResidualBlock_2.Conv2d_2",
        "adapter.body.3.resnets.0.block2": "StatefulResidualBlocks_4.ResidualBlocks.Chain.ResidualBlock_1.Conv2d_2",
        "adapter.body.3.resnets.1.block2": "StatefulResidualBlocks_4.ResidualBlocks.Chain.ResidualBlock_2.Conv2d_2",
    },
)

depth = Conversion(
    original=Hub(
        repo_id="TencentARC/t2iadapter_depth_sd15v2",
        filename="diffusion_pytorch_model.bin",
        revision="9f96518933daa6c9386692914f72af81a0f6978f",
        expected_sha256="68aaebf5e7d5eeb62eaea9476c68d279ba98d0876b385cc925e12c43cee19edd",
    ),
    converted=Hub(
        repo_id="refiners/sd15.t2i_adapter.depth",
        filename="model.safetensors",
        expected_sha256="0178baeb59713ef4ae4dcbca0a2d3447fdd42bbeeaed019d3dc01f0f1913f74f",
    ),
    recipe=diffusers_recipe,
    dtype=torch.float16,
)
