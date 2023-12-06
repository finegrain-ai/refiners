from refiners.foundationals.latent_diffusion.schedulers.scheduler import NoiseSchedule, Scheduler
from numpy import arange
from torch import Tensor, device as Device, dtype as Dtype, sqrt, float32, tensor, from_numpy


class EulerScheduler(Scheduler):

    def __init__(
        self,
        num_inference_steps: int,
        num_train_timesteps: int = 1_000,
        initial_diffusion_rate: float = 8.5e-4,
        final_diffusion_rate: float = 1.2e-2,
        noise_schedule: NoiseSchedule = NoiseSchedule.QUADRATIC,
        device: Device | str = "cpu",
        dtype: DType = float32,
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

    def _generate_timesteps(self) -> Tensor:
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (arange(0, self.num_inference_steps) *
                     step_ratio).round()[::-1]
        return from_numpy(timesteps).to(device=self.device)

    def __call__(self, x: Tensor, noise: Tensor, step: int) -> Tensor:
        timestep, previous_timestep = (
            self.timesteps[step],
            (self.timesteps[step + 1] if step < self.num_inference_steps -
             1 else tensor(data=[0], device=self.device, dtype=self.dtype)),
        )
        current_scale_factor, previous_scale_factor = self.cumulative_scale_factors[
            timestep], (self.cumulative_scale_factors[previous_timestep]
                        if previous_timestep > 0 else
                        self.cumulative_scale_factors[0])

        gamma = 0.0  # modify with inputs(?)
        eps = noise * 1.0  # modify with inputs(?)

        # with the hardcoded values sigma_hat is always current_scale_factor
        sigma_hat = current_scale_factor * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - current_scale_factor**2)**0.5

        predicted_x = (x - sqrt(1 - current_scale_factor**2) *
                       noise) / current_scale_factor

        derivative = (x - predicted_x) / sigma_hat
        dt = previous_scale_factor - sigma_hat
        denoised_x = x + derivative * dt

        return denoised_x
