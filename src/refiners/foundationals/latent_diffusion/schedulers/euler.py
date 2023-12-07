from refiners.foundationals.latent_diffusion.schedulers.scheduler import NoiseSchedule, Scheduler
from torch import Tensor, device as Device, dtype as Dtype, float32, tensor, arange, int64


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
        timesteps = arange(1000, 0, -step_ratio).round().type(int64)
        return timesteps - 1

    def __call__(self, x: Tensor, noise: Tensor, step: int) -> Tensor:
        timestep, previous_timestep = (
            self.timesteps[step],
            (
                self.timesteps[step + 1]
                if step < self.num_inference_steps - 1
                else tensor(data=[0], device=self.device, dtype=self.dtype)
            ),
        )
        current_sigma, previous_sigma = self.sigmas[timestep], (
            self.sigmas[previous_timestep] if previous_timestep > 0 else self.sigmas[0]
        )

        predicted_x = x - current_sigma * noise

        derivative = (x - predicted_x) / current_sigma
        dt = previous_sigma - current_sigma
        denoised_x = x + derivative * dt

        return denoised_x
