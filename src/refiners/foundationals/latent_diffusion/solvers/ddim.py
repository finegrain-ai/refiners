from torch import Generator, Tensor, device as Device, dtype as Dtype, float32, sqrt, tensor

from refiners.foundationals.latent_diffusion.solvers.solver import NoiseSchedule, Solver, TimestepSpacing


class DDIM(Solver):
    """Denoising Diffusion Implicit Model (DDIM) solver.

    See [[arXiv:2010.02502] Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) for more details.
    """

    def __init__(
        self,
        num_inference_steps: int,
        num_train_timesteps: int = 1_000,
        timesteps_spacing: TimestepSpacing = TimestepSpacing.LEADING,
        timesteps_offset: int = 1,
        initial_diffusion_rate: float = 8.5e-4,
        final_diffusion_rate: float = 1.2e-2,
        noise_schedule: NoiseSchedule = NoiseSchedule.QUADRATIC,
        first_inference_step: int = 0,
        device: Device | str = "cpu",
        dtype: Dtype = float32,
    ) -> None:
        """Initializes a new DDIM solver.

        Args:
            num_inference_steps: The number of inference steps.
            num_train_timesteps: The number of training timesteps.
            timesteps_spacing: The spacing to use for the timesteps.
            timesteps_offset: The offset to use for the timesteps.
            initial_diffusion_rate: The initial diffusion rate.
            final_diffusion_rate: The final diffusion rate.
            noise_schedule: The noise schedule.
            first_inference_step: The first inference step.
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """
        super().__init__(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=num_train_timesteps,
            timesteps_spacing=timesteps_spacing,
            timesteps_offset=timesteps_offset,
            initial_diffusion_rate=initial_diffusion_rate,
            final_diffusion_rate=final_diffusion_rate,
            noise_schedule=noise_schedule,
            first_inference_step=first_inference_step,
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
        assert self.first_inference_step <= step < self.num_inference_steps, "invalid step {step}"

        timestep, previous_timestep = (
            self.timesteps[step],
            (
                self.timesteps[step + 1]
                if step < self.num_inference_steps - 1
                else tensor(data=[0], device=self.device, dtype=self.dtype)
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
        predicted_x = (x - sqrt(1 - current_scale_factor**2) * predicted_noise) / current_scale_factor
        noise_factor = sqrt(1 - previous_scale_factor**2)

        # Do not add noise at the last step to avoid visual artifacts.
        if step == self.num_inference_steps - 1:
            noise_factor = 0

        denoised_x = previous_scale_factor * predicted_x + noise_factor * predicted_noise

        return denoised_x
