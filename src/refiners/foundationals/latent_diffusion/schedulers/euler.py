from refiners.foundationals.latent_diffusion.schedulers.scheduler import NoiseSchedule, Scheduler
from torch import Tensor, device as Device, dtype as Dtype, sqrt, float32, tensor, arange
import numpy as np
import torch


class EulerScheduler(Scheduler):

    def __init__(
        self,
        num_inference_steps: int,
        num_train_timesteps: int = 1_000,
        initial_diffusion_rate: float = 8.5e-4,
        final_diffusion_rate: float = 1.2e-2,
        noise_schedule: NoiseSchedule = NoiseSchedule.QUADRATIC,
        device: Device | str = "cpu",
        dtype: Dtype = float32,
    ):
        super().__init__(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=num_train_timesteps,
            initial_diffusion_rate=initial_diffusion_rate,
            final_diffusion_rate=final_diffusion_rate,
            noise_schedule=noise_schedule,
            device=device,
            dtype=dtype,
        )
        self.sigmas = self.noise_std / self.cumulative_scale_factors
        self.sigmas[0] = 0.0

    def _generate_timesteps(self) -> Tensor:
        step_ratio = self.num_train_timesteps / self.num_inference_steps
        timesteps = (np.arange(self.num_train_timesteps, 0,
                               -step_ratio)).round().copy().astype(int)
        return timesteps - 1

    def __call__(self, x: Tensor, noise: Tensor, step: int) -> Tensor:
        timestep, previous_timestep = (
            self.timesteps[step],
            (self.timesteps[step + 1] if step < self.num_inference_steps -
             1 else tensor(data=[0], device=self.device, dtype=self.dtype)),
        )
        current_sigma, previous_sigma = self.sigmas[timestep], (
            self.sigmas[previous_timestep]
            if previous_timestep > 0 else self.sigmas[0])

        gamma = 0.0  # modify with inputs(?)
        eps = noise * 1.0  # modify with inputs(?)

        # with the hardcoded values sigma_hat is always current_scale_factor
        sigma_hat = current_sigma * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - current_sigma**2)**0.5

        predicted_x = x - sigma_hat * noise

        derivative = (x - predicted_x) / sigma_hat
        dt = previous_sigma - sigma_hat
        denoised_x = x + derivative * dt

        return denoised_x
