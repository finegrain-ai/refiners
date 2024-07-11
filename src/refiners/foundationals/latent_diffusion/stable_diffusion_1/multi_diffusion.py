from dataclasses import dataclass

from torch import Tensor

from refiners.foundationals.latent_diffusion.multi_diffusion import DiffusionTarget, MultiDiffusion
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import (
    StableDiffusion_1,
)


@dataclass(kw_only=True)
class SD1DiffusionTarget(DiffusionTarget):
    clip_text_embedding: Tensor
    condition_scale: float = 7.0


class SD1MultiDiffusion(MultiDiffusion[SD1DiffusionTarget]):
    def __init__(self, sd: StableDiffusion_1) -> None:
        self.sd = sd

    def diffuse_target(self, x: Tensor, step: int, target: SD1DiffusionTarget) -> Tensor:
        old_solver = self.sd.solver
        self.sd.solver = target.solver
        result = self.sd(
            x=x,
            step=step,
            clip_text_embedding=target.clip_text_embedding,
            condition_scale=target.condition_scale,
        )
        self.sd.solver = old_solver
        return result
