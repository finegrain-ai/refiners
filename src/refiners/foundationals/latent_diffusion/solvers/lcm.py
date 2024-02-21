import numpy as np
import torch

from refiners.foundationals.latent_diffusion.solvers.dpm import DPMSolver
from refiners.foundationals.latent_diffusion.solvers.solver import NoiseSchedule, Solver


class LCMSolver(Solver):
    """Latent Consistency Model solver.

    This solver is designed for use either with
    [a specific base model][refiners.foundationals.latent_diffusion.stable_diffusion_xl.lcm.SDXLLcmAdapter]
    or [a specific LoRA][refiners.foundationals.latent_diffusion.stable_diffusion_xl.lcm_lora.add_lcm_lora].

    See [[arXiv:2310.04378] Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/abs/2310.04378)
    for details.
    """

    def __init__(
        self,
        num_inference_steps: int,
        num_train_timesteps: int = 1_000,
        num_orig_steps: int = 50,
        initial_diffusion_rate: float = 8.5e-4,
        final_diffusion_rate: float = 1.2e-2,
        noise_schedule: NoiseSchedule = NoiseSchedule.QUADRATIC,
        diffusers_mode: bool = False,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        assert (
            num_orig_steps >= num_inference_steps
        ), f"num_orig_steps ({num_orig_steps}) < num_inference_steps ({num_inference_steps})"

        self._dpm = [
            DPMSolver(
                num_inference_steps=num_orig_steps,
                num_train_timesteps=num_train_timesteps,
                device=device,
                dtype=dtype,
            )
        ]

        if diffusers_mode:
            # Diffusers recomputes the timesteps in LCMScheduler,
            # and it does it slightly differently than DPM Solver.
            # We provide this option to reproduce Diffusers' output.
            k = num_train_timesteps // num_orig_steps
            ts = np.asarray(list(range(1, num_orig_steps + 1))) * k - 1
            self.dpm.timesteps = torch.tensor(ts, device=device).flip(0)

        super().__init__(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=num_train_timesteps,
            initial_diffusion_rate=initial_diffusion_rate,
            final_diffusion_rate=final_diffusion_rate,
            noise_schedule=noise_schedule,
            device=device,
            dtype=dtype,
        )

    @property
    def dpm(self):
        return self._dpm[0]

    def _generate_timesteps(self) -> torch.Tensor:
        # Note: not the same as torch.linspace(start=0, end=num_train_timesteps, steps=5)[1:],
        # e.g. for 4 steps we use  [999, 759, 500, 260] instead of [999, 749, 499, 249].
        # This is due to the use of the Skipping-Steps technique during distillation,
        # see section 4.3 of the Latent Consistency Models paper (Luo 2023).
        # `k` in the paper is `num_train_timesteps / num_orig_steps`. In practice, SDXL
        # LCMs are distilled with DPM++.

        self.timestep_indices: list[int] = (
            torch.floor(
                torch.linspace(
                    start=0,
                    end=self.dpm.num_inference_steps,
                    steps=self.num_inference_steps + 1,
                )[:-1]
            )
            .int()
            .tolist()  # type: ignore
        )
        return self.dpm.timesteps[self.timestep_indices]

    def __call__(
        self,
        x: torch.Tensor,
        predicted_noise: torch.Tensor,
        step: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        current_timestep = self.timesteps[step]
        scale_factor = self.cumulative_scale_factors[current_timestep]
        noise_ratio = self.noise_std[current_timestep]
        estimated_denoised_data = (x - noise_ratio * predicted_noise) / scale_factor

        # To understand the values of c_skip and c_out,
        # see "Parameterization for Consistency Models" in appendix C
        # of the Consistency Models paper (Song 2023) and Karras 2022.
        #
        # However, note that there are two major differences:
        # - epsilon is unused (= 0);
        # - c_out is missing a `sigma` factor.
        #
        # This equation is the one used in the original implementation
        # (https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7)
        # and hence the one used to train all available models.
        #
        # See https://github.com/luosiallen/latent-consistency-model/issues/82
        # for more discussion regarding this.

        sigma = 0.5  # assume standard deviation of data distribution is 0.5
        t = current_timestep * 10  # make curve sharper
        c_skip = sigma**2 / (t**2 + sigma**2)
        c_out = t / torch.sqrt(sigma**2 + t**2)

        denoised_x = c_skip * x + c_out * estimated_denoised_data

        if step == self.num_inference_steps - 1:
            return denoised_x

        # re-noise intermediate steps
        noise = torch.randn(
            predicted_noise.shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        next_step = int(self.timestep_indices[step + 1])
        return self.dpm.add_noise(x=denoised_x, noise=noise, step=next_step)
