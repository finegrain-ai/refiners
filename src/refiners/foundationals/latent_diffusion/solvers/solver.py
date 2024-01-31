from abc import ABC, abstractmethod
from enum import Enum
from typing import TypeVar

from torch import Generator, Tensor, device as Device, dtype as DType, float32, linspace, log, sqrt

from refiners.fluxion import layers as fl

T = TypeVar("T", bound="Solver")


class NoiseSchedule(str, Enum):
    UNIFORM = "uniform"
    QUADRATIC = "quadratic"
    KARRAS = "karras"


class Solver(fl.Module, ABC):
    """
    A base class for creating a diffusion model solver.

    Solver creates a sequence of noise and scaling factors used in the diffusion process,
    which gradually transforms the original data distribution into a Gaussian one.

    This process is described using several parameters such as initial and final diffusion rates,
    and is encapsulated into a `__call__` method that applies a step of the diffusion process.
    """

    timesteps: Tensor

    def __init__(
        self,
        num_inference_steps: int,
        num_train_timesteps: int = 1_000,
        initial_diffusion_rate: float = 8.5e-4,
        final_diffusion_rate: float = 1.2e-2,
        noise_schedule: NoiseSchedule = NoiseSchedule.QUADRATIC,
        first_inference_step: int = 0,
        device: Device | str = "cpu",
        dtype: DType = float32,
    ) -> None:
        super().__init__()
        self.num_inference_steps = num_inference_steps
        self.num_train_timesteps = num_train_timesteps
        self.initial_diffusion_rate = initial_diffusion_rate
        self.final_diffusion_rate = final_diffusion_rate
        self.noise_schedule = noise_schedule
        self.first_inference_step = first_inference_step
        self.scale_factors = self.sample_noise_schedule()
        self.cumulative_scale_factors = sqrt(self.scale_factors.cumprod(dim=0))
        self.noise_std = sqrt(1.0 - self.scale_factors.cumprod(dim=0))
        self.signal_to_noise_ratios = log(self.cumulative_scale_factors) - log(self.noise_std)
        self.timesteps = self._generate_timesteps()
        self.to(device=device, dtype=dtype)

    @abstractmethod
    def __call__(self, x: Tensor, predicted_noise: Tensor, step: int, generator: Generator | None = None) -> Tensor:
        """
        Applies a step of the diffusion process to the input tensor `x` using the provided `predicted_noise` and `timestep`.

        This method should be overridden by subclasses to implement the specific diffusion process.
        """
        ...

    @abstractmethod
    def _generate_timesteps(self) -> Tensor:
        """
        Generates a tensor of timesteps.

        This method should be overridden by subclasses to provide the specific timesteps for the diffusion process.
        """
        ...

    def add_noise(
        self,
        x: Tensor,
        noise: Tensor,
        step: int,
    ) -> Tensor:
        timestep = self.timesteps[step]
        cumulative_scale_factors = self.cumulative_scale_factors[timestep]
        noise_stds = self.noise_std[timestep]
        noised_x = cumulative_scale_factors * x + noise_stds * noise
        return noised_x

    def remove_noise(self, x: Tensor, noise: Tensor, step: int) -> Tensor:
        timestep = self.timesteps[step]
        cumulative_scale_factors = self.cumulative_scale_factors[timestep]
        noise_stds = self.noise_std[timestep]
        # See equation (15) from https://arxiv.org/pdf/2006.11239.pdf. Useful to preview progress or for guidance like
        # in https://arxiv.org/pdf/2210.00939.pdf (self-attention guidance)
        denoised_x = (x - noise_stds * noise) / cumulative_scale_factors
        return denoised_x

    @property
    def all_steps(self) -> list[int]:
        return list(range(self.num_inference_steps))

    @property
    def inference_steps(self) -> list[int]:
        return self.all_steps[self.first_inference_step :]

    @property
    def device(self) -> Device:
        return self.scale_factors.device

    @property
    def dtype(self) -> DType:
        return self.scale_factors.dtype

    @device.setter
    def device(self, device: Device | str | None = None) -> None:
        self.to(device=device)

    @dtype.setter
    def dtype(self, dtype: DType | None = None) -> None:
        self.to(dtype=dtype)

    def rebuild(self: T, num_inference_steps: int | None, first_inference_step: int | None = None) -> T:
        num_inference_steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        first_inference_step = self.first_inference_step if first_inference_step is None else first_inference_step
        return self.__class__(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=self.num_train_timesteps,
            initial_diffusion_rate=self.initial_diffusion_rate,
            final_diffusion_rate=self.final_diffusion_rate,
            noise_schedule=self.noise_schedule,
            first_inference_step=first_inference_step,
            device=self.device,
            dtype=self.dtype,
        )

    def scale_model_input(self, x: Tensor, step: int) -> Tensor:
        """
        For compatibility with solvers that need to scale the input according to the current timestep.
        """
        return x

    def sample_power_distribution(self, power: float = 2, /) -> Tensor:
        return (
            linspace(
                start=self.initial_diffusion_rate ** (1 / power),
                end=self.final_diffusion_rate ** (1 / power),
                steps=self.num_train_timesteps,
            )
            ** power
        )

    def sample_noise_schedule(self) -> Tensor:
        match self.noise_schedule:
            case "uniform":
                return 1 - self.sample_power_distribution(1)
            case "quadratic":
                return 1 - self.sample_power_distribution(2)
            case "karras":
                return 1 - self.sample_power_distribution(7)
            case _:
                raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")

    def to(self, device: Device | str | None = None, dtype: DType | None = None) -> "Solver":
        super().to(device=device, dtype=dtype)
        for name, attr in [(name, attr) for name, attr in self.__dict__.items() if isinstance(attr, Tensor)]:
            match name:
                case "timesteps":
                    setattr(self, name, attr.to(device=device))
                case _:
                    setattr(self, name, attr.to(device=device, dtype=dtype))
        return self
