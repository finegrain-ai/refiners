import dataclasses
from collections import deque
from typing import NamedTuple

import numpy as np
import torch

from refiners.foundationals.latent_diffusion.solvers.solver import (
    BaseSolverParams,
    ModelPredictionType,
    NoiseSchedule,
    Solver,
    TimestepSpacing,
)


def safe_log(x: torch.Tensor, lower_bound: float = 1e-6) -> torch.Tensor:
    """Compute the log of a tensor with a lower bound."""
    return torch.log(torch.maximum(x, torch.tensor(lower_bound)))


def safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    """Compute the square root of a tensor ensuring that the input is non-negative"""
    return torch.sqrt(torch.maximum(x, torch.tensor(0)))


class SolverTensors(NamedTuple):
    cumulative_scale_factors: torch.Tensor
    noise_std: torch.Tensor
    signal_to_noise_ratios: torch.Tensor


class DPMSolver(Solver):
    """Diffusion probabilistic models (DPMs) solver.

    See [[arXiv:2211.01095] DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://arxiv.org/abs/2211.01095)
    for more details.

    Note:
        Regarding last_step_first_order: DPM-Solver++ is known to introduce artifacts
        when used with SDXL and few steps. This parameter is a way to mitigate that
        effect by using a first-order (Euler) update instead of a second-order update
        for the last step of the diffusion.
    """

    default_params = dataclasses.replace(
        Solver.default_params,
        timesteps_spacing=TimestepSpacing.CUSTOM,
    )

    def __init__(
        self,
        num_inference_steps: int,
        first_inference_step: int = 0,
        params: BaseSolverParams | None = None,
        last_step_first_order: bool = False,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initializes a new DPM solver.

        Args:
            num_inference_steps: The number of inference steps to perform.
            first_inference_step: The first inference step to perform.
            params: The common parameters for solvers.
            last_step_first_order: Use a first-order update for the last step.
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """
        if params and params.model_prediction_type not in (ModelPredictionType.NOISE, None):
            raise NotImplementedError
        if params and params.sde_variance not in (0.0, 1.0):
            raise NotImplementedError("DPMSolver only supports sde_variance=0.0 or 1.0")

        super().__init__(
            num_inference_steps=num_inference_steps,
            first_inference_step=first_inference_step,
            params=params,
            device=device,
            dtype=torch.float64,  # compute constants precisely
        )
        self.estimated_data = deque([torch.tensor([])] * 2, maxlen=2)
        self.last_step_first_order = last_step_first_order
        sigmas = self.noise_std / self.cumulative_scale_factors
        self.sigmas = self._rescale_sigmas(sigmas, self.params.sigma_schedule)
        sigma_min = sigmas[0:1]  # corresponds to `final_sigmas_type="sigma_min" in diffusers`
        self.sigmas = torch.cat([self.sigmas, sigma_min])
        self.cumulative_scale_factors, self.noise_std, self.signal_to_noise_ratios = self._solver_tensors_from_sigmas(
            self.sigmas
        )
        self.timesteps = self._timesteps_from_sigmas(sigmas)
        self.to(dtype=dtype)

    def rebuild(
        self: "DPMSolver",
        num_inference_steps: int | None,
        first_inference_step: int | None = None,
    ) -> "DPMSolver":
        """Rebuilds the solver with new parameters.

        Args:
            num_inference_steps: The number of inference steps.
            first_inference_step: The first inference step.
        """
        r = super().rebuild(
            num_inference_steps=num_inference_steps,
            first_inference_step=first_inference_step,
        )
        r.last_step_first_order = self.last_step_first_order
        return r

    def _generate_timesteps(self) -> torch.Tensor:
        if self.params.timesteps_spacing != TimestepSpacing.CUSTOM:
            return super()._generate_timesteps()

        # We use numpy here because:
        #   numpy.linspace(0,999,31)[15] is 499.49999999999994
        #   torch.linspace(0,999,31)[15] is 499.5
        # and we want the same result as the original DPM codebase.
        offset = self.params.timesteps_offset
        max_timestep = self.params.num_train_timesteps - 1 + offset
        np_space = np.linspace(offset, max_timestep, self.num_inference_steps + 1).round().astype(int)[1:]
        return torch.tensor(np_space).flip(0)

    def _rescale_sigmas(self, sigmas: torch.Tensor, sigma_schedule: NoiseSchedule | None) -> torch.Tensor:
        """Rescale the sigmas according to the sigma schedule."""
        match sigma_schedule:
            case NoiseSchedule.UNIFORM:
                rho = 1
            case NoiseSchedule.QUADRATIC:
                rho = 2
            case NoiseSchedule.KARRAS:
                rho = 7
            case None:
                return torch.tensor(
                    np.interp(self.timesteps.cpu(), np.arange(0, len(sigmas)), sigmas.cpu()),
                    device=self.device,
                )

        linear_schedule = torch.linspace(0, 1, steps=self.num_inference_steps, device=self.device)
        first_sigma = sigmas[0]
        last_sigma = sigmas[-1]
        rescaled_sigmas = (
            first_sigma ** (1 / rho) + linear_schedule * (last_sigma ** (1 / rho) - first_sigma ** (1 / rho))
        ) ** rho
        return rescaled_sigmas.flip(0)

    def _timesteps_from_sigmas(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Generate the timesteps from the sigmas."""
        log_sigmas = safe_log(sigmas)
        timesteps: list[torch.Tensor] = []
        for sigma in self.sigmas[:-1]:
            log_sigma = safe_log(sigma)
            distance_matrix = log_sigma - log_sigmas.unsqueeze(1)

            # Determine the range of sigma indices
            low_indices = (distance_matrix >= 0).cumsum(dim=0).argmax(dim=0).clip(max=sigmas.size(0) - 2)
            high_indices = low_indices + 1

            low_log_sigma = log_sigmas[low_indices]
            high_log_sigma = log_sigmas[high_indices]

            # Interpolate sigma values
            interpolation_weights = (low_log_sigma - log_sigma) / (low_log_sigma - high_log_sigma)
            interpolation_weights = torch.clamp(interpolation_weights, 0, 1)
            timestep = (1 - interpolation_weights) * low_indices + interpolation_weights * high_indices
            timesteps.append(timestep)

        return torch.cat(timesteps).round().int()

    def _add_noise(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Add noise to the input tensor using the solver's parameters.

        Args:
            x: The input tensor to add noise to.
            noise: The noise tensor to add to the input tensor.
            step: The current step of the diffusion process.

        Returns:
            The input tensor with added noise.
        """
        cumulative_scale_factors = self.cumulative_scale_factors[step]
        noise_stds = self.noise_std[step]

        # noisify the latents, arXiv:2006.11239 Eq. 4
        noised_x = cumulative_scale_factors * x + noise_stds * noise
        return noised_x

    def _solver_tensors_from_sigmas(self, sigmas: torch.Tensor) -> SolverTensors:
        """Generate the tensors from the sigmas."""
        cumulative_scale_factors = 1 / torch.sqrt(sigmas**2 + 1)
        noise_std = sigmas * cumulative_scale_factors
        signal_to_noise_ratios = safe_log(cumulative_scale_factors) - safe_log(noise_std)
        return SolverTensors(
            cumulative_scale_factors=cumulative_scale_factors,
            noise_std=noise_std,
            signal_to_noise_ratios=signal_to_noise_ratios,
        )

    def dpm_solver_first_order_update(
        self, x: torch.Tensor, noise: torch.Tensor, step: int, sde_noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Applies a first-order backward Euler update to the input data `x`.

        Args:
            x: The input data.
            noise: The predicted noise.
            step: The current step.

        Returns:
            The denoised version of the input data `x`.
        """
        current_ratio = self.signal_to_noise_ratios[step]
        next_ratio = self.signal_to_noise_ratios[step + 1]

        next_scale_factor = self.cumulative_scale_factors[step + 1]

        next_noise_std = self.noise_std[step + 1]
        current_noise_std = self.noise_std[step]

        ratio_delta = current_ratio - next_ratio

        if sde_noise is None:
            return (next_noise_std / current_noise_std) * x + (1.0 - torch.exp(ratio_delta)) * next_scale_factor * noise

        factor = 1.0 - torch.exp(2.0 * ratio_delta)
        return (
            (next_noise_std / current_noise_std) * torch.exp(ratio_delta) * x
            + next_scale_factor * factor * noise
            + next_noise_std * safe_sqrt(factor) * sde_noise
        )

    def multistep_dpm_solver_second_order_update(
        self, x: torch.Tensor, step: int, sde_noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Applies a second-order backward Euler update to the input data `x`.

        Args:
            x: The input data.
            step: The current step.

        Returns:
            The denoised version of the input data `x`.
        """
        current_data_estimation = self.estimated_data[-1]
        previous_data_estimation = self.estimated_data[-2]

        next_ratio = self.signal_to_noise_ratios[step + 1]
        current_ratio = self.signal_to_noise_ratios[step]
        previous_ratio = self.signal_to_noise_ratios[step - 1]

        next_scale_factor = self.cumulative_scale_factors[step + 1]
        next_noise_std = self.noise_std[step + 1]
        current_noise_std = self.noise_std[step]

        estimation_delta = (current_data_estimation - previous_data_estimation) / (
            (current_ratio - previous_ratio) / (next_ratio - current_ratio)
        )
        ratio_delta = current_ratio - next_ratio

        if sde_noise is None:
            factor = 1.0 - torch.exp(ratio_delta)
            return (
                (next_noise_std / current_noise_std) * x
                + next_scale_factor * factor * current_data_estimation
                + 0.5 * next_scale_factor * factor * estimation_delta
            )

        factor = 1.0 - torch.exp(2.0 * ratio_delta)
        return (
            (next_noise_std / current_noise_std) * torch.exp(ratio_delta) * x
            + next_scale_factor * factor * current_data_estimation
            + 0.5 * next_scale_factor * factor * estimation_delta
            + next_noise_std * safe_sqrt(factor) * sde_noise
        )

    def __call__(
        self, x: torch.Tensor, predicted_noise: torch.Tensor, step: int, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Apply one step of the backward diffusion process.

        Note:
            This method works by estimating the denoised version of `x` and applying either a first-order or second-order
            backward Euler update, which is a numerical method commonly used to solve ordinary differential equations
            (ODEs).

        Args:
            x: The input tensor to apply the diffusion process to.
            predicted_noise: The predicted noise tensor for the current step.
            step: The current step of the diffusion process.
            generator: The random number generator to use for sampling noise.

        Returns:
            The denoised version of the input data `x`.
        """
        assert self.first_inference_step <= step < self.num_inference_steps, "invalid step {step}"

        scale_factor = self.cumulative_scale_factors[step]
        noise_ratio = self.noise_std[step]
        estimated_denoised_data = (x - noise_ratio * predicted_noise) / scale_factor
        self.estimated_data.append(estimated_denoised_data)
        variance = self.params.sde_variance
        sde_noise = (
            torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype) * variance
            if variance > 0.0
            else None
        )

        if step == self.first_inference_step or (self.last_step_first_order and step == self.num_inference_steps - 1):
            return self.dpm_solver_first_order_update(
                x=x, noise=estimated_denoised_data, step=step, sde_noise=sde_noise
            )

        return self.multistep_dpm_solver_second_order_update(x=x, step=step, sde_noise=sde_noise)
