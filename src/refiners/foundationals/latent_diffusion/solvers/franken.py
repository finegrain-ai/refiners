import dataclasses
from typing import Any, Callable, Protocol, TypeVar

from torch import Generator, Tensor, device as Device, dtype as DType, float32

from refiners.foundationals.latent_diffusion.solvers.solver import Solver, TimestepSpacing

# Should be Tensor, but some Diffusers schedulers
# are improperly typed as only accepting `int`.
SchedulerTimestepT = Any


class SchedulerOutputLike(Protocol):
    @property
    def prev_sample(self) -> Tensor: ...


class SchedulerLike(Protocol):
    timesteps: Tensor

    @property
    def init_noise_sigma(self) -> Tensor | float: ...

    def set_timesteps(self, num_inference_steps: int, *args: Any, **kwargs: Any) -> None: ...

    def scale_model_input(self, sample: Tensor, timestep: SchedulerTimestepT) -> Tensor: ...

    def step(
        self,
        model_output: Tensor,
        timestep: SchedulerTimestepT,
        sample: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> SchedulerOutputLike | tuple[Any]: ...


TFrankenSolver = TypeVar("TFrankenSolver", bound="FrankenSolver")


class FrankenSolver(Solver):
    """Lets you use Diffusers Schedulers as Refiners Solvers.

    For instance:
        from diffusers import EulerDiscreteScheduler
        from refiners.foundationals.latent_diffusion.solvers import FrankenSolver

        scheduler = EulerDiscreteScheduler(...)
        solver = FrankenSolver(lambda: scheduler, num_inference_steps=steps)
    """

    default_params = dataclasses.replace(
        Solver.default_params,
        timesteps_spacing=TimestepSpacing.CUSTOM,
    )

    def __init__(
        self,
        get_diffusers_scheduler: Callable[[], SchedulerLike],
        num_inference_steps: int,
        first_inference_step: int = 0,
        device: Device | str = "cpu",
        dtype: DType = float32,
        **kwargs: Any,  # for typing, ignored
    ) -> None:
        self.get_diffusers_scheduler = get_diffusers_scheduler
        self.diffusers_scheduler = self.get_diffusers_scheduler()
        self.diffusers_scheduler.set_timesteps(num_inference_steps)
        super().__init__(
            num_inference_steps=num_inference_steps,
            first_inference_step=first_inference_step,
            device=device,
            dtype=dtype,
        )

    def _generate_timesteps(self) -> Tensor:
        return self.diffusers_scheduler.timesteps

    def to(self: TFrankenSolver, device: Device | str | None = None, dtype: DType | None = None) -> TFrankenSolver:
        return super().to(device=device, dtype=dtype)  # type: ignore

    def rebuild(
        self,
        num_inference_steps: int | None,
        first_inference_step: int | None = None,
    ) -> "FrankenSolver":
        return self.__class__(
            get_diffusers_scheduler=self.get_diffusers_scheduler,
            num_inference_steps=self.num_inference_steps if num_inference_steps is None else num_inference_steps,
            first_inference_step=self.first_inference_step if first_inference_step is None else first_inference_step,
            device=self.device,
            dtype=self.dtype,
        )

    def scale_model_input(self, x: Tensor, step: int) -> Tensor:
        if step == -1:
            return x * self.diffusers_scheduler.init_noise_sigma
        return self.diffusers_scheduler.scale_model_input(x, self.timesteps[step])

    def __call__(self, x: Tensor, predicted_noise: Tensor, step: int, generator: Generator | None = None) -> Tensor:
        timestep = self.timesteps[step]
        r = self.diffusers_scheduler.step(predicted_noise, timestep, x)
        assert not isinstance(r, tuple), "scheduler returned a tuple"
        return r.prev_sample
