from abc import ABC, abstractmethod
from enum import Enum
from typing import TypeVar

import numpy as np
from torch import Generator, Tensor, arange, device as Device, dtype as DType, float32, linspace, log, sqrt, tensor

from refiners.fluxion import layers as fl

T = TypeVar("T", bound="Solver")


class NoiseSchedule(str, Enum):
    """An enumeration of schedules used to sample the noise.

    Attributes:
        UNIFORM: A uniform noise schedule.
        QUADRATIC: A quadratic noise schedule. Corresponds to "Stable Diffusion" in [[arXiv:2305.08891] Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/abs/2305.08891) table 1.
        KARRAS: See [[arXiv:2206.00364] Elucidating the Design Space of Diffusion-Based Generative Models, Equation 5](https://arxiv.org/abs/2206.00364)
    """

    UNIFORM = "uniform"
    QUADRATIC = "quadratic"
    KARRAS = "karras"


class TimestepSpacing(str, Enum):
    """An enumeration of methods to space the timesteps.

    See [[arXiv:2305.08891] Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/abs/2305.08891) table 2.

    Attributes:
        LINSPACE_FLOAT: Sample N steps with linear interpolation, return a floating-point tensor.
        LINSPACE_INT: Same as LINSPACE_FLOAT but return an integer tensor with rounded timesteps.
        LEADING: Sample N+1 steps, do not include the last timestep (i.e. bad - non-zero SNR). Used in DDIM.
        TRAILING: Sample N+1 steps, do not include the first timestep.
        TRAILING_ALT: Variant of TRAILING used in DPM.
    """

    LINSPACE_FLOAT = "linspace_float"
    LINSPACE_INT = "linspace_int"
    LEADING = "leading"
    TRAILING = "trailing"
    TRAILING_ALT = "trailing_alt"


class Solver(fl.Module, ABC):
    """The base class for creating a diffusion model solver.

    Solvers create a sequence of noise and scaling factors used in the diffusion process,
    which gradually transforms the original data distribution into a Gaussian one.

    This process is described using several parameters such as initial and final diffusion rates,
    and is encapsulated into a `__call__` method that applies a step of the diffusion process.
    """

    timesteps: Tensor

    def __init__(
        self,
        num_inference_steps: int,
        num_train_timesteps: int = 1_000,
        timesteps_spacing: TimestepSpacing = TimestepSpacing.LINSPACE_FLOAT,
        timesteps_offset: int = 0,
        initial_diffusion_rate: float = 8.5e-4,
        final_diffusion_rate: float = 1.2e-2,
        noise_schedule: NoiseSchedule = NoiseSchedule.QUADRATIC,
        first_inference_step: int = 0,
        device: Device | str = "cpu",
        dtype: DType = float32,
    ) -> None:
        """Initializes a new `Solver` instance.

        Args:
            num_inference_steps: The number of inference steps to perform.
            num_train_timesteps: The number of timesteps used to train the diffusion process.
            timesteps_spacing: The spacing to use for the timesteps.
            timesteps_offset: The offset to use for the timesteps.
            initial_diffusion_rate: The initial diffusion rate used to sample the noise schedule.
            final_diffusion_rate: The final diffusion rate used to sample the noise schedule.
            noise_schedule: The noise schedule used to sample the noise schedule.
            first_inference_step: The first inference step to perform.
            device: The PyTorch device to use for the solver's tensors.
            dtype: The PyTorch data type to use for the solver's tensors.
        """
        super().__init__()
        self.num_inference_steps = num_inference_steps
        self.num_train_timesteps = num_train_timesteps
        self.timesteps_spacing = timesteps_spacing
        self.timesteps_offset = timesteps_offset
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
        """Apply a step of the diffusion process using the Solver.

        Note:
            This method should be overridden by subclasses to implement the specific diffusion process.

        Args:
            x: The input tensor to apply the diffusion process to.
            predicted_noise: The predicted noise tensor for the current step.
            step: The current step of the diffusion process.
            generator: The random number generator to use for sampling noise.
        """
        ...

    @staticmethod
    def generate_timesteps(
        spacing: TimestepSpacing,
        num_inference_steps: int,
        num_train_timesteps: int = 1000,
        offset: int = 0,
    ) -> Tensor:
        """Generate a tensor of timesteps according to a given spacing.

        Args:
            spacing: The spacing to use for the timesteps.
            num_inference_steps: The number of inference steps to perform.
            num_train_timesteps: The number of timesteps used to train the diffusion process.
            offset: The offset to use for the timesteps.
        """
        max_timestep = num_train_timesteps - 1 + offset
        match spacing:
            case TimestepSpacing.LINSPACE_FLOAT:
                return tensor(np.linspace(offset, max_timestep, num_inference_steps), dtype=float32).flip(0)
            case TimestepSpacing.LINSPACE_INT:
                return tensor(np.linspace(offset, max_timestep, num_inference_steps).round().astype(int)).flip(0)
            case TimestepSpacing.LEADING:
                step_ratio = num_train_timesteps // num_inference_steps
                return (arange(0, num_inference_steps, 1) * step_ratio + offset).flip(0)
            case TimestepSpacing.TRAILING:
                step_ratio = num_train_timesteps // num_inference_steps
                max_timestep = num_train_timesteps - 1 + offset
                return arange(max_timestep, offset, -step_ratio)
            case TimestepSpacing.TRAILING_ALT:
                # We use numpy here because:
                #   numpy.linspace(0,999,31)[15] is 499.49999999999994
                #   torch.linspace(0,999,31)[15] is 499.5
                # and we want the same result as the original DPM codebase.
                np_space = np.linspace(offset, max_timestep, num_inference_steps + 1).round().astype(int)[1:]
                return tensor(np_space).flip(0)

    def _generate_timesteps(self) -> Tensor:
        return self.generate_timesteps(
            spacing=self.timesteps_spacing,
            num_inference_steps=self.num_inference_steps,
            num_train_timesteps=self.num_train_timesteps,
            offset=self.timesteps_offset,
        )

    def add_noise(
        self,
        x: Tensor,
        noise: Tensor,
        step: int,
    ) -> Tensor:
        """Add noise to the input tensor using the solver's parameters.

        Args:
            x: The input tensor to add noise to.
            noise: The noise tensor to add to the input tensor.
            step: The current step of the diffusion process.

        Returns:
            The input tensor with added noise.
        """
        timestep = self.timesteps[step]
        cumulative_scale_factors = self.cumulative_scale_factors[timestep]
        noise_stds = self.noise_std[timestep]
        noised_x = cumulative_scale_factors * x + noise_stds * noise
        return noised_x

    def remove_noise(self, x: Tensor, noise: Tensor, step: int) -> Tensor:
        """Remove noise from the input tensor using the current step of the diffusion process.

        Note:
            See [[arXiv:2006.11239] Denoising Diffusion Probabilistic Models, Equation 15](https://arxiv.org/abs/2006.11239)
            and [[arXiv:2210.00939] Improving Sample Quality of Diffusion Models Using Self-Attention Guidance](https://arxiv.org/abs/2210.00939).

        Args:
            x: The input tensor to remove noise from.
            noise: The noise tensor to remove from the input tensor.
            step: The current step of the diffusion process.

        Returns:
            The denoised input tensor.
        """
        timestep = self.timesteps[step]
        cumulative_scale_factors = self.cumulative_scale_factors[timestep]
        noise_stds = self.noise_std[timestep]
        denoised_x = (x - noise_stds * noise) / cumulative_scale_factors
        return denoised_x

    @property
    def all_steps(self) -> list[int]:
        """Return a list of all inference steps."""
        return list(range(self.num_inference_steps))

    @property
    def inference_steps(self) -> list[int]:
        """Return a list of inference steps to perform."""
        return self.all_steps[self.first_inference_step :]

    @property
    def device(self) -> Device:
        """The PyTorch device used for the solver's tensors."""
        return self.scale_factors.device

    @property
    def dtype(self) -> DType:
        """The PyTorch data type used for the solver's tensors."""
        return self.scale_factors.dtype

    @device.setter
    def device(self, device: Device | str | None = None) -> None:
        self.to(device=device)

    @dtype.setter
    def dtype(self, dtype: DType | None = None) -> None:
        self.to(dtype=dtype)

    def rebuild(self: T, num_inference_steps: int | None, first_inference_step: int | None = None) -> T:
        """Rebuild the solver with new parameters.

        Args:
            num_inference_steps: The number of inference steps to perform.
            first_inference_step: The first inference step to perform.

        Returns:
            A new solver instance with the specified parameters.
        """
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
        """Scale the model's input according to the current timestep.

        Note:
            This method should only be overridden by solvers that
            need to scale the input according to the current timestep.

            By default, this method does not scale the input.
            (scale=1)

        Args:
            x: The input tensor to scale.
            step: The current step of the diffusion process.

        Returns:
            The scaled input tensor.
        """
        return x

    def sample_power_distribution(self, power: float = 2, /) -> Tensor:
        """Sample a power distribution.

        Args:
            power: The power to use for the distribution.

        Returns:
            A tensor representing the power distribution between the initial and final diffusion rates of the solver.
        """
        return (
            linspace(
                start=self.initial_diffusion_rate ** (1 / power),
                end=self.final_diffusion_rate ** (1 / power),
                steps=self.num_train_timesteps,
            )
            ** power
        )

    def sample_noise_schedule(self) -> Tensor:
        """Sample the noise schedule.

        Returns:
            A tensor representing the noise schedule.
        """
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
        """Move the solver to the specified device and data type.

        Args:
            device: The PyTorch device to move the solver to.
            dtype: The PyTorch data type to move the solver to.

        Returns:
            The solver instance, moved to the specified device and data type.
        """
        super().to(device=device, dtype=dtype)
        for name, attr in [(name, attr) for name, attr in self.__dict__.items() if isinstance(attr, Tensor)]:
            match name:
                case "timesteps":
                    setattr(self, name, attr.to(device=device))
                case _:
                    setattr(self, name, attr.to(device=device, dtype=dtype))
        return self
