from contextlib import contextmanager
from typing import Generator, NamedTuple

import torch
from PIL import Image
from torch import Tensor, device as Device, dtype as DType
from torch.nn import functional as F

from refiners.fluxion import layers as fl
from refiners.fluxion.adapters.adapter import Adapter
from refiners.fluxion.context import Contexts
from refiners.fluxion.layers import (
    Chain,
    Conv2d,
    Downsample,
    GroupNorm,
    Identity,
    Residual,
    SelfAttention2d,
    SiLU,
    Slicing,
    Sum,
    Upsample,
)
from refiners.fluxion.utils import image_to_tensor, images_to_tensor, no_grad, tensor_to_image, tensor_to_images


class _ImageSize(NamedTuple):
    height: int
    width: int


class _Tile(NamedTuple):
    top: int
    left: int
    bottom: int
    right: int


class Resnet(Sum):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 32,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        shortcut = (
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, device=device, dtype=dtype)
            if in_channels != out_channels
            else Identity()
        )
        super().__init__(
            shortcut,
            Chain(
                GroupNorm(channels=in_channels, num_groups=num_groups, device=device, dtype=dtype),
                SiLU(),
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
                GroupNorm(channels=out_channels, num_groups=num_groups, device=device, dtype=dtype),
                SiLU(),
                Conv2d(
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
    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        resnet_sizes: list[int] = [128, 256, 512, 512, 512]
        input_channels: int = 3
        latent_dim: int = 8
        resnet_layers: list[Chain] = [
            Chain(
                [
                    Resnet(
                        in_channels=resnet_sizes[i - 1] if i > 0 else resnet_sizes[0],
                        out_channels=resnet_sizes[i],
                        device=device,
                        dtype=dtype,
                    ),
                    Resnet(in_channels=resnet_sizes[i], out_channels=resnet_sizes[i], device=device, dtype=dtype),
                ]
            )
            for i in range(len(resnet_sizes))
        ]
        for _, layer in zip(range(3), resnet_layers):
            channels: int = layer[-1].out_channels  # type: ignore
            layer.append(Downsample(channels=channels, scale_factor=2, device=device, dtype=dtype))

        attention_layer = Residual(
            GroupNorm(channels=resnet_sizes[-1], num_groups=32, eps=1e-6, device=device, dtype=dtype),
            SelfAttention2d(channels=resnet_sizes[-1], device=device, dtype=dtype),
        )
        resnet_layers[-1].insert_after_type(Resnet, attention_layer)
        super().__init__(
            Conv2d(
                in_channels=input_channels,
                out_channels=resnet_sizes[0],
                kernel_size=3,
                padding=1,
                device=device,
                dtype=dtype,
            ),
            Chain(*resnet_layers),
            Chain(
                GroupNorm(channels=resnet_sizes[-1], num_groups=32, eps=1e-6, device=device, dtype=dtype),
                SiLU(),
                Conv2d(
                    in_channels=resnet_sizes[-1],
                    out_channels=latent_dim,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
            ),
            Chain(
                Conv2d(in_channels=8, out_channels=8, kernel_size=1, device=device, dtype=dtype),
                Slicing(dim=1, end=4),
            ),
        )

    def init_context(self) -> Contexts:
        return {"sampling": {"shapes": []}}


class Decoder(Chain):
    def __init__(self, device: Device | str | None = None, dtype: DType | None = None) -> None:
        self.resnet_sizes: list[int] = [128, 256, 512, 512, 512]
        self.latent_dim: int = 4
        self.output_channels: int = 3
        resnet_sizes = self.resnet_sizes[::-1]
        resnet_layers: list[Chain] = [
            (
                Chain(
                    [
                        Resnet(
                            in_channels=resnet_sizes[i - 1] if i > 0 else resnet_sizes[0],
                            out_channels=resnet_sizes[i],
                            device=device,
                            dtype=dtype,
                        ),
                        Resnet(in_channels=resnet_sizes[i], out_channels=resnet_sizes[i], device=device, dtype=dtype),
                        Resnet(in_channels=resnet_sizes[i], out_channels=resnet_sizes[i], device=device, dtype=dtype),
                    ]
                )
                if i > 0
                else Chain(
                    [
                        Resnet(in_channels=resnet_sizes[0], out_channels=resnet_sizes[i], device=device, dtype=dtype),
                        Resnet(in_channels=resnet_sizes[i], out_channels=resnet_sizes[i], device=device, dtype=dtype),
                    ]
                )
            )
            for i in range(len(resnet_sizes))
        ]
        attention_layer = Residual(
            GroupNorm(channels=resnet_sizes[0], num_groups=32, eps=1e-6, device=device, dtype=dtype),
            SelfAttention2d(channels=resnet_sizes[0], device=device, dtype=dtype),
        )
        resnet_layers[0].insert(1, attention_layer)
        for _, layer in zip(range(3), resnet_layers[1:]):
            channels: int = layer.layer(-1, Resnet).out_channels
            layer.insert(-1, Upsample(channels=channels, upsample_factor=2, device=device, dtype=dtype))
        super().__init__(
            Conv2d(
                in_channels=self.latent_dim, out_channels=self.latent_dim, kernel_size=1, device=device, dtype=dtype
            ),
            Conv2d(
                in_channels=self.latent_dim,
                out_channels=resnet_sizes[0],
                kernel_size=3,
                padding=1,
                device=device,
                dtype=dtype,
            ),
            Chain(*resnet_layers),
            Chain(
                GroupNorm(channels=resnet_sizes[-1], num_groups=32, eps=1e-6, device=device, dtype=dtype),
                SiLU(),
                Conv2d(
                    in_channels=resnet_sizes[-1],
                    out_channels=self.output_channels,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
            ),
        )


class FixedGroupNorm(fl.Chain, Adapter[fl.GroupNorm]):
    """
    Adapter for GroupNorm layers to fix the running mean and variance.

    This is useful when running tiled inference with a autoencoder to ensure that the statistics of the GroupNorm layers
    are consistent across tiles.
    """

    mean: torch.Tensor | None
    var: torch.Tensor | None

    def __init__(self, target: fl.GroupNorm) -> None:
        self.mean = None
        self.var = None
        with self.setup_adapter(target):
            super().__init__(fl.Lambda(self.compute_group_norm))

    def compute_group_norm(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        # Reshape the tensor to apply batch norm to each group separately (to mimic group norm behavior)
        x = x.reshape(
            1,
            batch * self.target.num_groups,
            int(channels / self.target.num_groups),
            height,
            width,
        )

        if self.mean is None or self.var is None:
            self.var, self.mean = torch.var_mean(x, dim=(0, 2, 3, 4), correction=0)

        result = F.batch_norm(
            input=x,
            running_mean=self.mean,
            running_var=self.var,
            weight=None,
            bias=None,
            training=False,
            momentum=0,
            eps=self.target.eps,
        )
        result = result.reshape(batch, channels, height, width)
        return result * self.target.weight.reshape(1, -1, 1, 1) + self.target.bias.reshape(1, -1, 1, 1)


def _create_blending_mask(
    size: _ImageSize,
    blending: int,
    num_channels: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    is_edge: tuple[bool, bool, bool, bool] = (False, False, False, False),
) -> torch.Tensor:
    mask = torch.ones(size, device=device, dtype=dtype)
    if blending == 0:
        return mask
    blending = min(blending, min(size) // 2)

    ramp = torch.linspace(0, 1, steps=blending, device=device, dtype=dtype)

    # Apply ramps only if not at the corresponding edge
    if not is_edge[0]:  # top
        mask[:blending, :] *= ramp.view(-1, 1)
    if not is_edge[1]:  # bottom
        mask[-blending:, :] *= ramp.flip(0).view(-1, 1)
    if not is_edge[2]:  # left
        mask[:, :blending] *= ramp.view(1, -1)
    if not is_edge[3]:  # right
        mask[:, -blending:] *= ramp.flip(0).view(1, -1)

    return mask.unsqueeze(0).unsqueeze(0).expand(1, num_channels, *size)


class LatentDiffusionAutoencoder(Chain):
    """Latent diffusion autoencoder model.

    Attributes:
        encoder_scale: The encoder scale to use.
    """

    encoder_scale = 0.18125

    def __init__(
        self,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initializes the model.

        Args:
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """
        super().__init__(
            Encoder(device=device, dtype=dtype),
            Decoder(device=device, dtype=dtype),
        )
        self._tile_size = None
        self._blending = None

    def encode(self, x: Tensor) -> Tensor:
        """Encode an image.

        Args:
            x: The image tensor to encode.

        Returns:
            The encoded tensor.
        """
        encoder = self[0]
        x = self.encoder_scale * encoder(x)
        return x

    def decode(self, x: Tensor) -> Tensor:
        """Decode a latent tensor.

        Args:
            x: The latent to decode.

        Returns:
            The decoded image tensor.
        """
        decoder = self[1]
        x = decoder(x / self.encoder_scale)
        return x

    def image_to_latents(self, image: Image.Image) -> Tensor:
        """
        Encode an image to latents.
        """
        return self.images_to_latents([image])

    def tiled_image_to_latents(self, image: Image.Image) -> Tensor:
        """
        Convert an image to latents with gradient blending to smooth tile edges.

        You need to activate the tiled inference context manager with the `tiled_inference` method to use this method.

        ```python
        with lda.tiled_inference(sample_image, tile_size=(768, 1024)):
            latents = lda.tiled_image_to_latents(sample_image)
        """
        if self._tile_size is None:
            raise ValueError("Tiled inference context manager not active. Use `tiled_inference` method to activate.")

        assert self._tile_size is not None and self._blending is not None
        image_tensor = image_to_tensor(image, device=self.device, dtype=self.dtype)
        image_tensor = 2 * image_tensor - 1
        return self._tiled_encode(image_tensor, self._tile_size, self._blending)

    def images_to_latents(self, images: list[Image.Image]) -> Tensor:
        """Convert a list of images to latents.

        Args:
            images: The list of images to convert.

        Returns:
            A tensor containing the latents associated with the images.
        """
        x = images_to_tensor(images, device=self.device, dtype=self.dtype)
        x = 2 * x - 1
        return self.encode(x)

    def latents_to_image(self, x: Tensor) -> Image.Image:
        """
        Decode latents to an image.
        """
        if x.shape[0] != 1:
            raise ValueError(f"Expected batch size of 1, got {x.shape[0]}")

        return self.latents_to_images(x)[0]

    def tiled_latents_to_image(self, x: Tensor) -> Image.Image:
        """
        Convert latents to an image with gradient blending to smooth tile edges.

        You need to activate the tiled inference context manager with the `tiled_inference` method to use this method.

        ```python
        with lda.tiled_inference(sample_image, tile_size=(768, 1024)):
            image = lda.tiled_latents_to_image(latents)
        """
        if self._tile_size is None:
            raise ValueError("Tiled inference context manager not active. Use `tiled_inference` method to activate.")

        assert self._tile_size is not None and self._blending is not None
        result = self._tiled_decode(x, self._tile_size, self._blending)
        return tensor_to_image((result + 1) / 2)

    def latents_to_images(self, x: Tensor) -> list[Image.Image]:
        """Convert a tensor of latents to images.

        Args:
            x: The tensor of latents to convert.

        Returns:
            A list of images associated with the latents.
        """
        x = self.decode(x)
        x = (x + 1) / 2
        return tensor_to_images(x)

    @staticmethod
    def _generate_latent_tiles(size: _ImageSize, tile_size: _ImageSize, overlap: int = 8) -> list[_Tile]:
        """
        Generate tiles for a given latent size and tile size with a given overlap.
        """
        tiles: list[_Tile] = []

        for x in range(0, size.width, tile_size.width - overlap):
            for y in range(0, size.height, tile_size.height - overlap):
                tile = _Tile(
                    top=max(0, y),
                    left=max(0, x),
                    bottom=min(size.height, y + tile_size.height),
                    right=min(size.width, x + tile_size.width),
                )
                tiles.append(tile)

        return tiles

    @no_grad()
    def _add_fixed_group_norm(self, image: Image.Image, inference_size: _ImageSize) -> None:
        """
        Set the running mean and variance of the group norm layers in the latent diffusion autoencoder.

        We replace the GroupNorm layers with FixedGroupNorm layers that will compute the group norm statistics on its
        first forward pass and then fix them for all subsequent passes. This is useful when running tiled inference to
        ensure that the statistics of the GroupNorm layers are consistent across tiles.
        """
        for group_norm, parent in self.walk(fl.GroupNorm):
            FixedGroupNorm(group_norm).inject(parent)

        downscaled_image = image.resize((inference_size.width, inference_size.height))  # type: ignore

        image_tensor = image_to_tensor(image, device=self.device, dtype=self.dtype)
        downscaled_image_tensor = image_to_tensor(downscaled_image, device=self.device, dtype=self.dtype)
        downscaled_image_tensor.clamp_(min=image_tensor.min(), max=image_tensor.max())

        std, mean = torch.std_mean(image_tensor, dim=[0, 2, 3], keepdim=True)
        new_std, new_mean = torch.std_mean(downscaled_image_tensor, dim=[0, 2, 3], keepdim=True)

        downscaled_image_tensor = (downscaled_image_tensor - new_mean) * (std / new_std) + mean
        downscaled_image_tensor = 2 * downscaled_image_tensor - 1

        # We do a forward pass through the encoder and decoder to set the group norm stats in the FixedGroupNorm layers
        latents = self.encode(downscaled_image_tensor)
        self.decode(latents)

    def _remove_fixed_group_norm(self) -> None:
        """
        Remove the FixedGroupNorm layers and restore the original GroupNorm layers.
        """
        for fixed_group_norm in self.layers(FixedGroupNorm):
            fixed_group_norm.eject()

    @no_grad()
    def _tiled_encode(self, image_tensor: torch.Tensor, tile_size: _ImageSize, blending: int = 64) -> torch.Tensor:
        """
        Encode an image to latents with tile-based inference and gradient blending to smooth tile edges.

        If `tile_size` is not provided, the tile size provided in the `tiled_inference` context manager is used, or the
        default tile size of 512x512 is used.
        """
        latent_size = _ImageSize(height=image_tensor.shape[2] // 8, width=image_tensor.shape[3] // 8)
        target_latent_tile_size = _ImageSize(height=tile_size.height // 8, width=tile_size.width // 8)
        tiles = self._generate_latent_tiles(latent_size, tile_size=target_latent_tile_size, overlap=blending // 8)

        if len(tiles) == 1:
            return self.encode(image_tensor)

        result = torch.zeros((1, 4, *latent_size), device=self.device, dtype=self.dtype)
        weights = torch.zeros_like(result)

        for latent_tile in tiles:
            pixel_tile = image_tensor[
                :,
                :,
                latent_tile.top * 8 : latent_tile.bottom * 8,
                latent_tile.left * 8 : latent_tile.right * 8,
            ]
            encoded_tile = self.encode(pixel_tile)

            is_edge = (
                latent_tile.top == 0,
                latent_tile.bottom == latent_size.height,
                latent_tile.left == 0,
                latent_tile.right == latent_size.width,
            )

            latent_tile_size = _ImageSize(
                height=(latent_tile.bottom - latent_tile.top), width=(latent_tile.right - latent_tile.left)
            )

            tile_mask = _create_blending_mask(
                latent_tile_size,
                blending // 8,
                num_channels=4,
                device=self.device,
                dtype=self.dtype,
                is_edge=is_edge,
            )

            result[
                :,
                :,
                latent_tile.top : latent_tile.bottom,
                latent_tile.left : latent_tile.right,
            ] += encoded_tile * tile_mask

            weights[
                :,
                :,
                latent_tile.top : latent_tile.bottom,
                latent_tile.left : latent_tile.right,
            ] += tile_mask

        return result / weights

    @no_grad()
    def _tiled_decode(self, latents: torch.Tensor, tile_size: _ImageSize, blending: int = 64) -> torch.Tensor:
        """
        Convert latents to an image for the given latent diffusion autoencoder, with gradient blending to smooth tile edges.

        If `tile_size` is not provided, the tile size provided in the `tiled_inference` context manager is used, or the
        default tile size of 512x512 is used.
        """
        latent_size = _ImageSize(height=latents.shape[2], width=latents.shape[3])
        pixel_size = _ImageSize(height=latent_size.height * 8, width=latent_size.width * 8)
        target_latent_tile_size = _ImageSize(height=tile_size.height // 8, width=tile_size.width // 8)
        tiles = self._generate_latent_tiles(latent_size, tile_size=target_latent_tile_size, overlap=blending // 8)
        if len(tiles) == 1:
            return self.decode(latents)

        result = torch.zeros((1, 3, *pixel_size), device=self.device, dtype=self.dtype)
        weights = torch.zeros_like(result)

        for latent_tile in tiles:
            pixel_offset = _ImageSize(height=latent_tile.top * 8, width=latent_tile.left * 8)
            latent_tile_size = _ImageSize(
                height=latent_tile.bottom - latent_tile.top, width=latent_tile.right - latent_tile.left
            )
            pixel_tile_size = _ImageSize(height=latent_tile_size.height * 8, width=latent_tile_size.width * 8)

            pixel_tile = self.decode(
                latents[
                    :,
                    :,
                    latent_tile.top : latent_tile.bottom,
                    latent_tile.left : latent_tile.right,
                ]
            )

            is_edge = (
                latent_tile.top == 0,
                latent_tile.bottom == latent_size.height,
                latent_tile.left == 0,
                latent_tile.right == latent_size.width,
            )

            pixel_tile_mask = _create_blending_mask(
                size=pixel_tile_size,
                blending=blending,
                num_channels=3,
                device=self.device,
                dtype=self.dtype,
                is_edge=is_edge,
            )
            result[
                :,
                :,
                pixel_offset.height : pixel_offset.height + pixel_tile_size.height,
                pixel_offset.width : pixel_offset.width + pixel_tile_size.width,
            ] += pixel_tile * pixel_tile_mask

            weights[
                :,
                :,
                pixel_offset.height : pixel_offset.height + pixel_tile_size.height,
                pixel_offset.width : pixel_offset.width + pixel_tile_size.width,
            ] += pixel_tile_mask

        return result / weights

    @contextmanager
    def tiled_inference(
        self, image: Image.Image, tile_size: tuple[int, int] = (512, 512), blending: int = 64
    ) -> Generator[None, None, None]:
        """
        Context manager for tiled inference operations to save VRAM for large images.

        This context manager sets up a consistent GroupNorm statistics for performing tiled operations on the
        autoencoder, including setting and resetting group norm statistics. This allow to make sure that the result is
        consistent across tiles by capturing the statistics of the GroupNorm layers on a downsampled version of the
        image.

        Be careful not to use the normal `image_to_latents` and `latents_to_image` methods while this context manager is
        active, as this will fail silently and run the operation without tiling.

        ```python
        with lda.tiled_inference(sample_image, tile_size=(768, 1024), blending=32):
            latents = lda.tiled_image_to_latents(sample_image)
            decoded_image = lda.tiled_latents_to_image(latents)
        """
        try:
            self._blending = blending
            self._tile_size = _ImageSize(width=tile_size[0], height=tile_size[1])
            self._add_fixed_group_norm(image, inference_size=self._tile_size)
            yield
        finally:
            self._remove_fixed_group_norm()
            self._tile_size = None
            self._blending = None
