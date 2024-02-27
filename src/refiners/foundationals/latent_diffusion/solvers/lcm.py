import dataclasses

import torch

from refiners.foundationals.latent_diffusion.solvers.dpm import DPMSolver
from refiners.foundationals.latent_diffusion.solvers.solver import (
    ModelPredictionType,
    Solver,
    SolverParams,
    TimestepSpacing,
)


class LCMSolver(Solver):
    """Latent Consistency Model solver.

    This solver is designed for use either with
    [a specific base model][refiners.foundationals.latent_diffusion.stable_diffusion_xl.lcm.SDXLLcmAdapter]
    or [a specific LoRA][refiners.foundationals.latent_diffusion.stable_diffusion_xl.lcm_lora.add_lcm_lora].

    See [[arXiv:2310.04378] Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/abs/2310.04378)
    for details.
    """

    # The spacing parameter is actually the spacing of the underlying DPM solver.
    default_params = dataclasses.replace(Solver.default_params, timesteps_spacing=TimestepSpacing.TRAILING)

    def __init__(
        self,
        num_inference_steps: int,
        first_inference_step: int = 0,
        params: SolverParams | None = None,
        num_orig_steps: int = 50,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """Initializes a new LCM solver.

        Args:
            num_inference_steps: The number of inference steps to perform.
            first_inference_step: The first inference step to perform.
            params: The common parameters for solvers.
            num_orig_steps: The number of inference steps of the emulated DPM solver.
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """

        assert (
            num_orig_steps >= num_inference_steps
        ), f"num_orig_steps ({num_orig_steps}) < num_inference_steps ({num_inference_steps})"

        params = self.resolve_params(params)
        if params.model_prediction_type != ModelPredictionType.NOISE:
            raise NotImplementedError

        self._dpm = [
            DPMSolver(
                num_inference_steps=num_orig_steps,
                params=SolverParams(
                    num_train_timesteps=params.num_train_timesteps,
                    timesteps_spacing=params.timesteps_spacing,
                ),
                device=device,
                dtype=dtype,
            )
        ]
        super().__init__(
            num_inference_steps=num_inference_steps,
            first_inference_step=first_inference_step,
            params=params,
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
        self, x: torch.Tensor, predicted_noise: torch.Tensor, step: int, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Apply one step of the backward diffusion process.

        Args:
            x: The input tensor to apply the diffusion process to.
            predicted_noise: The predicted noise tensor for the current step.
            step: The current step of the diffusion process.
            generator: The random number generator to use for sampling noise.

        Returns:
            The denoised version of the input data `x`.
        """
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
