import dataclasses
from collections import deque

import numpy as np
from torch import Generator, Tensor, device as Device, dtype as Dtype, exp, float32, tensor

from refiners.foundationals.latent_diffusion.solvers.solver import (
    BaseSolverParams,
    ModelPredictionType,
    Solver,
    TimestepSpacing,
)


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
        device: Device | str = "cpu",
        dtype: Dtype = float32,
    ):
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

        super().__init__(
            num_inference_steps=num_inference_steps,
            first_inference_step=first_inference_step,
            params=params,
            device=device,
            dtype=dtype,
        )
        self.estimated_data = deque([tensor([])] * 2, maxlen=2)
        self.last_step_first_order = last_step_first_order

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

    def _generate_timesteps(self) -> Tensor:
        if self.params.timesteps_spacing != TimestepSpacing.CUSTOM:
            return super()._generate_timesteps()

        # We use numpy here because:
        #   numpy.linspace(0,999,31)[15] is 499.49999999999994
        #   torch.linspace(0,999,31)[15] is 499.5
        # and we want the same result as the original DPM codebase.
        offset = self.params.timesteps_offset
        max_timestep = self.params.num_train_timesteps - 1 + offset
        np_space = np.linspace(offset, max_timestep, self.num_inference_steps + 1).round().astype(int)[1:]
        return tensor(np_space).flip(0)

    def dpm_solver_first_order_update(self, x: Tensor, noise: Tensor, step: int) -> Tensor:
        """Applies a first-order backward Euler update to the input data `x`.

        Args:
            x: The input data.
            noise: The predicted noise.
            step: The current step.

        Returns:
            The denoised version of the input data `x`.
        """
        current_timestep = self.timesteps[step]
        previous_timestep = self.timesteps[step + 1] if step < self.num_inference_steps - 1 else tensor([0])

        previous_ratio = self.signal_to_noise_ratios[previous_timestep]
        current_ratio = self.signal_to_noise_ratios[current_timestep]

        previous_scale_factor = self.cumulative_scale_factors[previous_timestep]

        previous_noise_std = self.noise_std[previous_timestep]
        current_noise_std = self.noise_std[current_timestep]

        factor = exp(-(previous_ratio - current_ratio)) - 1.0
        denoised_x = (previous_noise_std / current_noise_std) * x - (factor * previous_scale_factor) * noise
        return denoised_x

    def multistep_dpm_solver_second_order_update(self, x: Tensor, step: int) -> Tensor:
        """Applies a second-order backward Euler update to the input data `x`.

        Args:
            x: The input data.
            step: The current step.

        Returns:
            The denoised version of the input data `x`.
        """
        previous_timestep = self.timesteps[step + 1] if step < self.num_inference_steps - 1 else tensor([0])
        current_timestep = self.timesteps[step]
        next_timestep = self.timesteps[step - 1]

        current_data_estimation = self.estimated_data[-1]
        next_data_estimation = self.estimated_data[-2]

        previous_ratio = self.signal_to_noise_ratios[previous_timestep]
        current_ratio = self.signal_to_noise_ratios[current_timestep]
        next_ratio = self.signal_to_noise_ratios[next_timestep]

        previous_scale_factor = self.cumulative_scale_factors[previous_timestep]
        previous_noise_std = self.noise_std[previous_timestep]
        current_noise_std = self.noise_std[current_timestep]

        estimation_delta = (current_data_estimation - next_data_estimation) / (
            (current_ratio - next_ratio) / (previous_ratio - current_ratio)
        )
        factor = exp(-(previous_ratio - current_ratio)) - 1.0
        denoised_x = (
            (previous_noise_std / current_noise_std) * x
            - (factor * previous_scale_factor) * current_data_estimation
            - 0.5 * (factor * previous_scale_factor) * estimation_delta
        )
        return denoised_x

    def __call__(self, x: Tensor, predicted_noise: Tensor, step: int, generator: Generator | None = None) -> Tensor:
        """Apply one step of the backward diffusion process.

        Note:
            This method works by estimating the denoised version of `x` and applying either a first-order or second-order
            backward Euler update, which is a numerical method commonly used to solve ordinary differential equations
            (ODEs).

        Args:
            x: The input tensor to apply the diffusion process to.
            predicted_noise: The predicted noise tensor for the current step.
            step: The current step of the diffusion process.
            generator: The random number generator to use for sampling noise (ignored, this solver is deterministic).

        Returns:
            The denoised version of the input data `x`.
        """
        assert self.first_inference_step <= step < self.num_inference_steps, "invalid step {step}"

        current_timestep = self.timesteps[step]
        scale_factor, noise_ratio = self.cumulative_scale_factors[current_timestep], self.noise_std[current_timestep]
        estimated_denoised_data = (x - noise_ratio * predicted_noise) / scale_factor
        self.estimated_data.append(estimated_denoised_data)

        if step == self.first_inference_step or (self.last_step_first_order and step == self.num_inference_steps - 1):
            return self.dpm_solver_first_order_update(x=x, noise=estimated_denoised_data, step=step)

        return self.multistep_dpm_solver_second_order_update(x=x, step=step)
