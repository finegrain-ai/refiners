import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, NamedTuple, Sequence

import torch
from torch import Tensor
from typing_extensions import TypeVar

from refiners.foundationals.latent_diffusion.solvers.solver import Solver

MAX_STEPS = 1000


class Tile(NamedTuple):
    top: int
    left: int
    bottom: int
    right: int


class Size(NamedTuple):
    height: int
    width: int


@dataclass(kw_only=True)
class DiffusionTarget:
    """
    Represents a target for the tiled diffusion process.

    This class encapsulates the parameters and properties needed to define a specific area (target) within a larger
    diffusion process, allowing for fine-grained control over different regions of the generated image.

    Attributes:
        tile: The tile defining the area of the target within the latent image.
        solver: The solver to use for this target's diffusion process. This is useful because some solvers have an
            internal state that needs to be updated during the diffusion process. Using the same solver instance for
            multiple targets would interfere with this internal state.
        init_latents: The initial latents for this target. If None, the target will be initialized with noise.
        opacity_mask: Mask controlling the target's visibility in the final image.
            If None, the target will be fully visible. Otherwise, 1 means fully opaque and 0 means fully transparent
            which means the target has no influence.
        weight: The importance of this target in the final image. Higher values increase the target's influence.
        start_step: The diffusion step at which this target begins to influence the process.
        end_step: The diffusion step at which this target stops influencing the process.
        size: The size of the target area.
        offset: The top-left offset of the target area within the latent image.

    The combination of `opacity_mask` and `weight` determines the target's overall contribution to the final generated
    image. The `solver` is responsible for the actual diffusion calculations for this target.
    """

    tile: Tile
    solver: Solver
    init_latents: Tensor | None = None
    opacity_mask: Tensor | None = None
    weight: int = 1
    start_step: int = 0
    end_step: int = MAX_STEPS

    @property
    def size(self) -> Size:
        return Size(
            height=self.tile.bottom - self.tile.top,
            width=self.tile.right - self.tile.left,
        )

    @property
    def offset(self) -> tuple[int, int]:
        return self.tile.top, self.tile.left

    def crop(self, tensor: Tensor, /) -> Tensor:
        height, width = self.size
        top_offset, left_offset = self.offset
        return tensor[:, :, top_offset : top_offset + height, left_offset : left_offset + width]

    def paste(self, tensor: Tensor, /, crop: Tensor) -> Tensor:
        height, width = self.size
        top_offset, left_offset = self.offset
        tensor[:, :, top_offset : top_offset + height, left_offset : left_offset + width] = crop
        return tensor


T = TypeVar("T", bound=DiffusionTarget)


class MultiDiffusion(ABC, Generic[T]):
    """
    MultiDiffusion class for performing multi-target diffusion using tiled diffusion.

    For more details, refer to the paper: [MultiDiffusion](https://arxiv.org/abs/2302.08113)
    """

    def __call__(self, x: Tensor, /, noise: Tensor, step: int, targets: Sequence[T]) -> Tensor:
        num_updates = torch.zeros_like(input=x)
        cumulative_values = torch.zeros_like(input=x)

        for target in targets:
            match step:
                case step if step == target.start_step and target.init_latents is not None:
                    noise_view = target.crop(noise)
                    view = target.solver.add_noise(
                        x=target.init_latents,
                        noise=noise_view,
                        step=step,
                    )
                case step if target.start_step <= step <= target.end_step:
                    view = target.crop(x)
                case _:
                    continue
            view = self.diffuse_target(x=view, step=step, target=target)
            weight = target.weight * target.opacity_mask if target.opacity_mask is not None else target.weight
            num_updates = target.paste(num_updates, crop=target.crop(num_updates) + weight)
            cumulative_values = target.paste(cumulative_values, crop=target.crop(cumulative_values) + weight * view)

        return torch.where(condition=num_updates > 0, input=cumulative_values / num_updates, other=x)

    @abstractmethod
    def diffuse_target(self, x: Tensor, step: int, target: T) -> Tensor: ...

    @staticmethod
    def generate_latent_tiles(size: Size, tile_size: Size, min_overlap: int = 8) -> list[Tile]:
        """
        Generate tiles for a latent image with the given size and tile size.

        If one dimension of the `tile_size` is larger than the corresponding dimension of the image size, a single tile is
        used to cover the entire image - and therefore `tile_size` is ignored. This algorithm ensures that the tile size
        is respected as much as possible, while still covering the entire image and respecting the minimum overlap.
        """
        assert (
            0 <= min_overlap < min(tile_size.height, tile_size.width)
        ), "Overlap must be non-negative and less than the tile size"

        if tile_size.width > size.width or tile_size.height > size.height:
            return [Tile(top=0, left=0, bottom=size.height, right=size.width)]

        tiles: list[Tile] = []

        def _compute_tiles_and_overlap(length: int, tile_length: int, min_overlap: int) -> tuple[int, int]:
            if tile_length >= length:
                return 1, 0
            num_tiles = math.ceil((length - tile_length) / (tile_length - min_overlap)) + 1
            overlap = (num_tiles * tile_length - length) // (num_tiles - 1)
            return num_tiles, overlap

        num_tiles_x, overlap_x = _compute_tiles_and_overlap(
            length=size.width, tile_length=tile_size.width, min_overlap=min_overlap
        )
        num_tiles_y, overlap_y = _compute_tiles_and_overlap(
            length=size.height, tile_length=tile_size.height, min_overlap=min_overlap
        )

        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                x = j * (tile_size.width - overlap_x)
                y = i * (tile_size.height - overlap_y)

                # Adjust x and y coordinates to ensure full-sized tiles
                if x + tile_size.width > size.width:
                    x = size.width - tile_size.width
                if y + tile_size.height > size.height:
                    y = size.height - tile_size.height

                tile_right = x + tile_size.width
                tile_bottom = y + tile_size.height
                tiles.append(Tile(top=y, left=x, bottom=tile_bottom, right=tile_right))

        return tiles
