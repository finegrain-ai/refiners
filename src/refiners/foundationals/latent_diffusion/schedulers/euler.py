from refiners.foundationals.latent_diffusion.schedulers.scheduler import NoiseSchedule, Scheduler
from torch import Tensor, device as Device, dtype as Dtype, float32, tensor, arange
import torch
import numpy as np


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
        self.sigmas = self._generate_sigmas().to(device=self.device, dtype=self.dtype)

    @property
    def init_noise_sigma(self) -> Tensor:
        return (self.sigmas.max() ** 2 + 1) ** 0.5

    def _generate_timesteps(self) -> Tensor:
        # using "leading" timestep
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (arange(start=0, end=self.num_inference_steps, step=1) * step_ratio).flip(0)
        return timesteps + 1

    def _generate_sigmas(self) -> Tensor:
        sigmas = self.noise_std / self.cumulative_scale_factors
        sigmas = torch.from_numpy(
            np.interp(self.timesteps.cpu().numpy(), np.arange(0, len(sigmas)), sigmas.cpu().numpy())
        )
        sigmas = torch.cat([sigmas, tensor([0.0])])
        return sigmas

    def scale_model_input(self, x: Tensor, step: int) -> Tensor:
        sigma = self.sigmas[step]
        x = x / ((sigma**2 + 1) ** 0.5)
        return x

    def __call__(
        self,
        x: Tensor,
        noise: Tensor,
        step: int,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
    ) -> Tensor:
        sigma = self.sigmas[step]

        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0

        alt_noise = torch.randn_like(noise)
        eps = alt_noise * s_noise
        sigma_hat = sigma * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5

        predicted_x = x - sigma_hat * noise

        # 1st order Euler
        derivative = (x - predicted_x) / sigma_hat
        dt = self.sigmas[step + 1] - sigma_hat
        denoised_x = x + derivative * dt

        return denoised_x
