import torch

from refiners.conversion.utils import Conversion, Hub, WeightRecipe

diffusers_recipe = WeightRecipe(
    key_map={
        # first part
        "adapter.conv_in": "Conv2d",
        "adapter.body.0.resnets.0.block1": "StatefulResidualBlocks_1.ResidualBlocks.Chain.ResidualBlock_1.Conv2d_1",
        "adapter.body.0.resnets.0.block2": "StatefulResidualBlocks_1.ResidualBlocks.Chain.ResidualBlock_1.Conv2d_2",
        "adapter.body.0.resnets.1.block1": "StatefulResidualBlocks_1.ResidualBlocks.Chain.ResidualBlock_2.Conv2d_1",
        "adapter.body.0.resnets.1.block2": "StatefulResidualBlocks_1.ResidualBlocks.Chain.ResidualBlock_2.Conv2d_2",
        # second part
        "adapter.body.1.in_conv": "StatefulResidualBlocks_2.ResidualBlocks.Conv2d",
        "adapter.body.1.resnets.0.block1": "StatefulResidualBlocks_2.ResidualBlocks.Chain.ResidualBlock_1.Conv2d_1",
        "adapter.body.1.resnets.0.block2": "StatefulResidualBlocks_2.ResidualBlocks.Chain.ResidualBlock_1.Conv2d_2",
        "adapter.body.1.resnets.1.block1": "StatefulResidualBlocks_2.ResidualBlocks.Chain.ResidualBlock_2.Conv2d_1",
        "adapter.body.1.resnets.1.block2": "StatefulResidualBlocks_2.ResidualBlocks.Chain.ResidualBlock_2.Conv2d_2",
        # third part
        "adapter.body.2.in_conv": "StatefulResidualBlocks_3.ResidualBlocks.Conv2d",
        "adapter.body.2.resnets.0.block1": "StatefulResidualBlocks_3.ResidualBlocks.Chain.ResidualBlock_1.Conv2d_1",
        "adapter.body.2.resnets.0.block2": "StatefulResidualBlocks_3.ResidualBlocks.Chain.ResidualBlock_1.Conv2d_2",
        "adapter.body.2.resnets.1.block1": "StatefulResidualBlocks_3.ResidualBlocks.Chain.ResidualBlock_2.Conv2d_1",
        "adapter.body.2.resnets.1.block2": "StatefulResidualBlocks_3.ResidualBlocks.Chain.ResidualBlock_2.Conv2d_2",
        # fourth part
        "adapter.body.3.resnets.0.block1": "StatefulResidualBlocks_4.ResidualBlocks.Chain.ResidualBlock_1.Conv2d_1",
        "adapter.body.3.resnets.0.block2": "StatefulResidualBlocks_4.ResidualBlocks.Chain.ResidualBlock_1.Conv2d_2",
        "adapter.body.3.resnets.1.block1": "StatefulResidualBlocks_4.ResidualBlocks.Chain.ResidualBlock_2.Conv2d_1",
        "adapter.body.3.resnets.1.block2": "StatefulResidualBlocks_4.ResidualBlocks.Chain.ResidualBlock_2.Conv2d_2",
    },
)

canny = Conversion(
    original=Hub(
        repo_id="TencentARC/t2i-adapter-canny-sdxl-1.0",
        filename="diffusion_pytorch_model.safetensors",
        revision="2d7244ba45ded9129cfbf8e96a4befb7f6094210",
        expected_sha256="b601b28b7df0c0dcbbaf704ab8ba6fd22bcf35c9a875fa0c9bc933d47cc27438",
    ),
    converted=Hub(
        repo_id="refiners/sdxl.t2i_adapter.canny",
        filename="model.safetensors",
        expected_sha256="3aabc9b964b220b0ff80ad383eebf1885f6298f74425c1dbee659c86127d4b60",
    ),
    recipe=diffusers_recipe,
    dtype=torch.float16,
)
depth_zoe = Conversion(
    original=Hub(
        repo_id="TencentARC/t2i-adapter-depth-zoe-sdxl-1.0",
        filename="diffusion_pytorch_model.safetensors",
        revision="d6282b3be9e0376d3fd0fd132959fdab143e2e51",
        expected_sha256="613884bf550bf6750f2f22cdb17bb92f49a148cad01fdd952118df4d386acd55",
    ),
    converted=Hub(
        repo_id="refiners/sdxl.t2i_adapter.depth.zoe",
        filename="model.safetensors",
        expected_sha256="f7199c4c6b1f38bc50ed8c7cb6220a0fd09d18779c29044e28d8ff582e941f82",
    ),
    recipe=diffusers_recipe,
    dtype=torch.float16,
)
