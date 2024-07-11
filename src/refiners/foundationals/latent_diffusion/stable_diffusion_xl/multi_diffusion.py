from dataclasses import dataclass

from torch import Tensor

from refiners.foundationals.latent_diffusion import StableDiffusion_XL
from refiners.foundationals.latent_diffusion.multi_diffusion import DiffusionTarget, MultiDiffusion


@dataclass(kw_only=True)
class SDXLTarget(DiffusionTarget):
    clip_text_embedding: Tensor
    condition_scale: float = 5.0
    pooled_text_embedding: Tensor
    time_ids: Tensor


class SDXLMultiDiffusion(MultiDiffusion[SDXLTarget]):
    def __init__(self, sd: StableDiffusion_XL) -> None:
        self.sd = sd

    def diffuse_target(self, x: Tensor, step: int, target: SDXLTarget) -> Tensor:
        old_solver = self.sd.solver
        self.sd.solver = target.solver
        result = self.sd(
            x=x,
            step=step,
            clip_text_embedding=target.clip_text_embedding,
            pooled_text_embedding=target.pooled_text_embedding,
            time_ids=target.time_ids,
            condition_scale=target.condition_scale,
        )
        self.sd.solver = old_solver
        return result
