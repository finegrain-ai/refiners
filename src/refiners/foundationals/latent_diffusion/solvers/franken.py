import dataclasses
from typing import Any, cast

from torch import Generator, Tensor

from refiners.foundationals.latent_diffusion.solvers.solver import Solver, TimestepSpacing


class FrankenSolver(Solver):
    """Lets you use Diffusers Schedulers as Refiners Solvers.

    For instance:
        from diffusers import EulerDiscreteScheduler
        from refiners.foundationals.latent_diffusion.solvers import FrankenSolver

        scheduler = EulerDiscreteScheduler(...)
        solver = FrankenSolver(scheduler, num_inference_steps=steps)
    """

    default_params = dataclasses.replace(
        Solver.default_params,
        timesteps_spacing=TimestepSpacing.CUSTOM,
    )

    def __init__(
        self,
        diffusers_scheduler: Any,
        num_inference_steps: int,
        first_inference_step: int = 0,
        **kwargs: Any,
    ) -> None:
        self.diffusers_scheduler = diffusers_scheduler
        diffusers_scheduler.set_timesteps(num_inference_steps)
        super().__init__(num_inference_steps=num_inference_steps, first_inference_step=first_inference_step)

    def _generate_timesteps(self) -> Tensor:
        return self.diffusers_scheduler.timesteps

    def rebuild(
        self,
        num_inference_steps: int | None,
        first_inference_step: int | None = None,
    ) -> "FrankenSolver":
        return self.__class__(
            diffusers_scheduler=self.diffusers_scheduler,
            num_inference_steps=self.num_inference_steps if num_inference_steps is None else num_inference_steps,
            first_inference_step=self.first_inference_step if first_inference_step is None else first_inference_step,
        )

    def scale_model_input(self, x: Tensor, step: int) -> Tensor:
        if step == -1:
            return x * self.diffusers_scheduler.init_noise_sigma
        return self.diffusers_scheduler.scale_model_input(x, self.timesteps[step])

    def __call__(self, x: Tensor, predicted_noise: Tensor, step: int, generator: Generator | None = None) -> Tensor:
        timestep = self.timesteps[step]
        return cast(Tensor, self.diffusers_scheduler.step(predicted_noise, timestep, x).prev_sample)
