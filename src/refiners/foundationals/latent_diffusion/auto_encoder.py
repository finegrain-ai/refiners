from PIL import Image
from torch import Tensor, device as Device, dtype as DType

from refiners.fluxion.context import Contexts
from refiners.fluxion.layers import (
    Chain,
    Conv2d,
    Conv3d,
    Downsample,
    Downsample3d,
    GroupNorm,
    Identity,
    Residual,
    SelfAttention2d,
    SelfAttention3d,
    SiLU,
    Slicing,
    Sum,
    Upsample,
    Upsample3d,
)
from refiners.fluxion.utils import images_to_tensor, tensor_to_images


class Resnet(Sum):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 32,
        spatial_dims: int = 2,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        if spatial_dims == 2:
            Conv = Conv2d
        elif spatial_dims == 3:
            Conv = Conv3d
        else:
            raise ValueError(f"Unsupported spatial dimension {spatial_dims}")
        
        shortcut = (
            Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, device=device, dtype=dtype)
            if in_channels != out_channels
            else Identity()
        )
        super().__init__(
            shortcut,
            Chain(
                GroupNorm(channels=in_channels, num_groups=num_groups, device=device, dtype=dtype),
                SiLU(),
                Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
                GroupNorm(channels=out_channels, num_groups=num_groups, device=device, dtype=dtype),
                SiLU(),
                Conv(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )


class Encoder(Chain):
    def __init__(self, 
        spatial_dims: int = 2, 
        num_groups: int = 32, 
        resnet_sizes: list[int] = [128, 256, 512, 512, 512],
        input_channels: int = 3,
        n_down_samples: int = 3,
        latent_dim: int = 8,
        slide_end: int = 4,
        device: Device | str | None = None, 
        dtype: DType | None = None,        
    ) -> None:
        if spatial_dims == 2:
            Conv = Conv2d
            SelfAttention = SelfAttention2d
            Dsample = Downsample
        elif spatial_dims == 3:
            Conv = Conv3d
            SelfAttention = SelfAttention3d
            Dsample = Downsample3d
        else:
            raise ValueError(f"Unsupported spatial dimension {spatial_dims}")
        
        resnet_layers: list[Chain] = [
            Chain(
                [
                    Resnet(
                        in_channels=resnet_sizes[i - 1] if i > 0 else resnet_sizes[0],
                        out_channels=resnet_sizes[i],
                        num_groups=num_groups,
                        spatial_dims=spatial_dims,
                        device=device,
                        dtype=dtype,
                    ),
                    Resnet(
                        in_channels=resnet_sizes[i],
                        num_groups=num_groups,
                        out_channels=resnet_sizes[i],
                        spatial_dims=spatial_dims,                        
                        device=device,
                        dtype=dtype,
                    ),
                ]
            )
            for i in range(len(resnet_sizes))
        ]
        for _, layer in zip(range(n_down_samples), resnet_layers):
            channels: int = layer[-1].out_channels  # type: ignore
            layer.append(Dsample(channels=channels, scale_factor=2, device=device, dtype=dtype))

        attention_layer = Residual(
            GroupNorm(channels=resnet_sizes[-1], num_groups=num_groups, eps=1e-6, device=device, dtype=dtype),
            SelfAttention(channels=resnet_sizes[-1], device=device, dtype=dtype),
        )
        resnet_layers[-1].insert_after_type(Resnet, attention_layer)
        super().__init__(
            Conv(
                in_channels=input_channels,
                out_channels=resnet_sizes[0],
                kernel_size=3,
                padding=1,
                device=device,
                dtype=dtype,
            ),
            Chain(*resnet_layers),
            Chain(
                GroupNorm(channels=resnet_sizes[-1], num_groups=num_groups, eps=1e-6, device=device, dtype=dtype),
                SiLU(),
                Conv(
                    in_channels=resnet_sizes[-1],
                    out_channels=latent_dim,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
            ),
            Chain(
                Conv(in_channels=latent_dim, out_channels=latent_dim, kernel_size=1, device=device, dtype=dtype),
                Slicing(dim=1, end=slide_end),
            ),
        )

    def init_context(self) -> Contexts:
        return {"sampling": {"shapes": []}}


class Decoder(Chain):
    def __init__(self, 
        spatial_dims: int = 2, 
        num_groups: int = 32, 
        resnet_sizes: list[int] = [128, 256, 512, 512, 512],
        output_channels: int = 3,
        latent_dim: int = 8,
        n_up_samples: int = 3,
        device: Device | str | None = None, 
        dtype: DType | None = None,     
    ) -> None:    
                     
        if spatial_dims == 2:
            Conv = Conv2d
            SelfAttention = SelfAttention2d
            Usample = Upsample
        elif spatial_dims == 3:
            Conv = Conv3d
            SelfAttention = SelfAttention3d
            Usample = Upsample3d
        else:
            raise ValueError(f"Unsupported spatial dimension {spatial_dims}")
         
        resnet_sizes = resnet_sizes[::-1]
        
        resnet_layers: list[Chain] = [
            (
                Chain(
                    [
                        Resnet(
                            in_channels=resnet_sizes[i - 1],
                            out_channels=resnet_sizes[i],
                            num_groups=num_groups,
                            spatial_dims=spatial_dims,
                            device=device,
                            dtype=dtype,
                        )
                        if i > 0
                        else Identity(),
                        Resnet(
                            in_channels=resnet_sizes[i],
                            out_channels=resnet_sizes[i],
                            num_groups=num_groups,
                            spatial_dims=spatial_dims,
                            device=device,
                            dtype=dtype,
                        ),
                        Resnet(
                            in_channels=resnet_sizes[i],
                            out_channels=resnet_sizes[i],
                            num_groups=num_groups,
                            spatial_dims=spatial_dims,
                            device=device,
                            dtype=dtype,
                        ),
                    ]
                )
            )
            for i in range(len(resnet_sizes))
        ]
        attention_layer = Residual(
            GroupNorm(channels=resnet_sizes[0], num_groups=num_groups, eps=1e-6, device=device, dtype=dtype),
            SelfAttention(channels=resnet_sizes[0], device=device, dtype=dtype),
        )
        resnet_layers[0].insert(1, attention_layer)
        if n_up_samples > len(resnet_layers) - 1:
            raise ValueError(
                f"Number of up-samples ({n_up_samples}) must be less than or equal to the number of resnet layers - 1 ({len(resnet_layers)})"
            )
        for _, layer in zip(range(n_up_samples), resnet_layers[1:]):
            channels: int = layer[-1].out_channels
            layer.insert(-1, Usample(channels=channels, upsample_factor=2, device=device, dtype=dtype))
        super().__init__(
            Conv(
                in_channels=latent_dim, out_channels=latent_dim, kernel_size=1, device=device, dtype=dtype
            ),
            Conv(
                in_channels=latent_dim,
                out_channels=resnet_sizes[0],
                kernel_size=3,
                padding=1,
                device=device,
                dtype=dtype,
            ),
            Chain(*resnet_layers),
            Chain(
                GroupNorm(channels=resnet_sizes[-1], num_groups=num_groups, eps=1e-6, device=device, dtype=dtype),
                SiLU(),
                Conv(
                    in_channels=resnet_sizes[-1],
                    out_channels=output_channels,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )


class LatentDiffusionAutoencoder(Chain):
    encoder_scale = 0.18125

    def __init__(
        self,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            Encoder(device=device, dtype=dtype),
            Decoder(device=device, dtype=dtype),
        )

    def encode(self, x: Tensor) -> Tensor:
        encoder = self[0]
        x = self.encoder_scale * encoder(x)
        return x

    def decode(self, x: Tensor) -> Tensor:
        decoder = self[1]
        x = decoder(x / self.encoder_scale)
        return x

    def image_to_latent(self, image: Image.Image) -> Tensor:
        return self.images_to_latents([image])

    def images_to_latents(self, images: list[Image.Image]) -> Tensor:
        x = images_to_tensor(images)
        x = 2 * x - 1
        return self.encode(x)
    
    # backward-compatibility alias
    def decode_latents(self, x: Tensor) -> Image.Image:
        return self.latent_to_image(x)

    def latent_to_image(self, x: Tensor) -> Image.Image:
        if x.shape[0] != 1:
            raise ValueError(f"Expected batch size of 1, got {x.shape[0]}")

        return self.latents_to_images(x)[0]

    def latents_to_images(self, x: Tensor) -> list[Image.Image]:
        x = self.decode(x)
        x = (x + 1) / 2
        return tensor_to_images(x)
