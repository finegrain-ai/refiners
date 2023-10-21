from torch import Tensor, device as Device, randn, arange, Generator, tensor
from refiners.foundationals.latent_diffusion.schedulers.scheduler import Scheduler


class DDPM(Scheduler):
    """
    The Denoising Diffusion Probabilistic Models (DDPM) is a specific type of diffusion model,
    which uses a specific strategy to generate the timesteps and applies the diffusion process in a specific way.
    """

    def __init__(
        self,
        num_inference_steps: int,
        num_train_timesteps: int = 1_000,
        initial_diffusion_rate: float = 8.5e-4,
        final_diffusion_rate: float = 1.2e-2,
        device: Device | str = "cpu",
    ) -> None:
        super().__init__(num_inference_steps, num_train_timesteps, initial_diffusion_rate, final_diffusion_rate, device)

    def _generate_timesteps(self) -> Tensor:
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = arange(start=0, end=self.num_inference_steps, step=1, device=self.device) * step_ratio
        return timesteps.flip(0)

    def __call__(self, x: Tensor, noise: Tensor, step: int, generator: Generator | None = None) -> Tensor:
        """
        Generate the next step in the diffusion process.

        This method adjusts the input data using added noise and an estimate of the denoised data, based on the current
        step in the diffusion process. This adjusted data forms the next step in the diffusion process.

        1. It uses current and previous timesteps to calculate the current factor dictating the contribution of original
        data and noise to the new step.
        2. An estimate of the denoised data (`estimated_denoised_data`) is generated.
        3. It calculates coefficients for the estimated denoised data and current data (`original_data_coeff` and
        `current_data_coeff`) that balance their contribution to the denoised data for the next step.
        4. It calculates the denoised data for the next step (`denoised_x`), which is a combination of the estimated
        denoised data and current data, adjusted by their respective coefficients.
        5. Noise is then added to `denoised_x`. The magnitude of noise is controlled by a calculated variance based on
        the cumulative scaling factor and the current factor.

        The output is the new data step for the next stage in the diffusion process.
        """
        timestep, previous_timestep = (
            self.timesteps[step],
            (
                self.timesteps[step + 1]
                if step < len(self.timesteps) - 1
                else tensor(-(self.num_train_timesteps // self.num_inference_steps), device=self.device)
            ),
        )
        current_cumulative_factor, previous_cumulative_scale_factor = (self.scale_factors.cumprod(0))[timestep], (
            (self.scale_factors.cumprod(0))[previous_timestep]
            if step < len(self.timesteps) - 1
            else tensor(1, device=self.device)
        )
        current_factor = current_cumulative_factor / previous_cumulative_scale_factor
        estimated_denoised_data = (x - (1 - current_cumulative_factor) ** 0.5 * noise) / current_cumulative_factor**0.5
        estimated_denoised_data = estimated_denoised_data.clamp(-1, 1)
        original_data_coeff = (previous_cumulative_scale_factor**0.5 * (1 - current_factor)) / (
            1 - current_cumulative_factor
        )
        current_data_coeff = (
            current_factor**0.5 * (1 - previous_cumulative_scale_factor) / (1 - current_cumulative_factor)
        )
        denoised_x = original_data_coeff * estimated_denoised_data + current_data_coeff * x
        if step < len(self.timesteps) - 1:
            variance = (1 - previous_cumulative_scale_factor) / (1 - current_cumulative_factor) * (1 - current_factor)
            denoised_x = denoised_x + (variance.clamp(min=1e-20) ** 0.5) * randn(
                x.shape, device=x.device, dtype=x.dtype, generator=generator
            )
        return denoised_x
