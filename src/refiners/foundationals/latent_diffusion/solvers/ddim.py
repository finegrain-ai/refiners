import dataclasses

import torch
from torch import Generator, Tensor, device as Device, dtype as Dtype

from refiners.foundationals.latent_diffusion.solvers.solver import (
    BaseSolverParams,
    ModelPredictionType,
    Solver,
    TimestepSpacing,
)


class DDIM(Solver):
    """Denoising Diffusion Implicit Model (DDIM) solver.

    See [[arXiv:2010.02502] Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) for more details.
    """

    default_params = dataclasses.replace(
        Solver.default_params,
        timesteps_spacing=TimestepSpacing.LEADING,
        timesteps_offset=1,
    )

    def __init__(
        self,
        num_inference_steps: int,
        first_inference_step: int = 0,
        params: BaseSolverParams | None = None,
        device: Device | str = "cpu",
        dtype: Dtype = torch.float32,
    ) -> None:
        """Initializes a new DDIM solver.

        Args:
            num_inference_steps: The number of inference steps to perform.
            first_inference_step: The first inference step to perform.
            params: The common parameters for solvers.
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """
        if params and params.model_prediction_type not in (ModelPredictionType.NOISE, None):
            raise NotImplementedError
        if params and params.sde_variance != 0.0:
            raise NotImplementedError("DDIM does not support sde_variance != 0.0 yet")

        super().__init__(
            num_inference_steps=num_inference_steps,
            first_inference_step=first_inference_step,
            params=params,
            device=device,
            dtype=dtype,
        )

    def __call__(self, x: Tensor, predicted_noise: Tensor, step: int, generator: Generator | None = None) -> Tensor:
        """Apply one step of the backward diffusion process.

        Args:
            x: The input tensor to apply the diffusion process to.
            predicted_noise: The predicted noise tensor for the current step.
            step: The current step of the diffusion process.
            generator: The random number generator to use for sampling noise (ignored, this solver is deterministic).

        Returns:
            The denoised version of the input data `x`.
        """
        assert self.first_inference_step <= step < self.num_inference_steps, f"invalid step {step}"

        timestep, previous_timestep = (
            self.timesteps[step],
            (
                self.timesteps[step + 1]
                if step < self.num_inference_steps - 1
                else torch.tensor(data=[0], device=self.device, dtype=self.dtype)
            ),
        )
        current_scale_factor, previous_scale_factor = (
            self.cumulative_scale_factors[timestep],
            (
                self.cumulative_scale_factors[previous_timestep]
                if previous_timestep > 0
                else self.cumulative_scale_factors[0]
            ),
        )
        predicted_x = (x - torch.sqrt(1 - current_scale_factor**2) * predicted_noise) / current_scale_factor
        noise_factor = torch.sqrt(1 - previous_scale_factor**2)

        # Do not add noise at the last step to avoid visual artifacts.
        if step == self.num_inference_steps - 1:
            noise_factor = 0

        denoised_x = previous_scale_factor * predicted_x + noise_factor * predicted_noise

        return denoised_x
