import numpy as np
import torch
from torch import Generator, Tensor, device as Device, dtype as Dtype, float32, tensor

from refiners.foundationals.latent_diffusion.solvers.solver import (
    ModelPredictionType,
    NoiseSchedule,
    Solver,
    SolverParams,
)


class Euler(Solver):
    """Euler solver.

    See [[arXiv:2206.00364] Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
    for more details.
    """

    def __init__(
        self,
        num_inference_steps: int,
        first_inference_step: int = 0,
        params: SolverParams | None = None,
        device: Device | str = "cpu",
        dtype: Dtype = float32,
    ):
        """Initializes a new Euler solver.

        Args:
            num_inference_steps: The number of inference steps to perform.
            first_inference_step: The first inference step to perform.
            params: The common parameters for solvers.
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """
        if params and params.noise_schedule not in (NoiseSchedule.QUADRATIC, None):
            raise NotImplementedError

        super().__init__(
            num_inference_steps=num_inference_steps,
            first_inference_step=first_inference_step,
            params=params,
            device=device,
            dtype=dtype,
        )
        self.sigmas = self._generate_sigmas()

    @property
    def init_noise_sigma(self) -> Tensor:
        """The initial noise sigma."""
        return self.sigmas.max()

    def _generate_sigmas(self) -> Tensor:
        """Generate the sigmas used by the solver."""
        sigmas = self.noise_std / self.cumulative_scale_factors
        sigmas = torch.tensor(np.interp(self.timesteps.cpu().numpy(), np.arange(0, len(sigmas)), sigmas.cpu().numpy()))
        sigmas = torch.cat([sigmas, tensor([0.0])])
        return sigmas.to(device=self.device, dtype=self.dtype)

    def scale_model_input(self, x: Tensor, step: int) -> Tensor:
        """Scales the model input according to the current step.

        Args:
            x: The model input.
            step: The current step.

        Returns:
            The scaled model input.
        """
        sigma = self.sigmas[step]
        return x / ((sigma**2 + 1) ** 0.5)

    def __call__(self, x: Tensor, predicted_noise: Tensor, step: int, generator: Generator | None = None) -> Tensor:
        """Apply one step of the backward diffusion process.

        Args:
            x: The input tensor to apply the diffusion process to.
            predicted_noise: The predicted noise tensor for the current step (or x0 if the prediction type is SAMPLE).
            step: The current step of the diffusion process.
            generator: The random number generator to use for sampling noise (ignored, this solver is deterministic).

        Returns:
            The denoised version of the input data `x`.
        """
        assert self.first_inference_step <= step < self.num_inference_steps, "invalid step {step}"

        if self.params.model_prediction_type == ModelPredictionType.SAMPLE:
            x0 = predicted_noise  # the model does not actually predict the noise but x0
            ratio = self.sigmas[step + 1] / self.sigmas[step]
            return ratio * x + (1 - ratio) * x0

        assert self.params.model_prediction_type == ModelPredictionType.NOISE
        return x + predicted_noise * (self.sigmas[step + 1] - self.sigmas[step])
