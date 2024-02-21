import numpy as np
import torch
from torch import Generator, Tensor, device as Device, dtype as Dtype, float32, tensor
# from typing import Callable

from refiners.foundationals.latent_diffusion.solvers.solver import NoiseSchedule, Solver

from collections import deque
from collections.abc import Callable

class SASolver(Solver):
    """Stochastic Adams (SA) Solver.

    See [[arXiv:2309.05019] SA-Solver: Stochastic Adams Solver for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2309.05019)
    for more details.

    training-free sampler
    """

    def __init__(
        self,
        num_inference_steps: int,
        noise_prediction_function: Callable,
        num_train_timesteps: int = 1_000,
        initial_diffusion_rate: float = 8.5e-4,
        final_diffusion_rate: float = 1.2e-2,
        noise_schedule: NoiseSchedule = NoiseSchedule.QUADRATIC,
        first_inference_step: int = 0,
        predictor_steps: int = 2,
        corrector_steps: int = 2,
        variance: float | Tensor = 0.,
        device: Device | str = "cpu",
        dtype: Dtype = float32
    ):
        super().__init__(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=num_train_timesteps,
            initial_diffusion_rate=initial_diffusion_rate,
            final_diffusion_rate=final_diffusion_rate,
            noise_schedule=noise_schedule,
            first_inference_step=first_inference_step,
            device=device,
            dtype=dtype
        )

        self.noise_prediction_function = noise_prediction_function
        if isinstance(variance, float):
            self.variance : Tensor = torch.tensor([variance]*num_train_timesteps)
        else:
            self.variance : Tensor = variance

        self.predictor_steps = predictor_steps
        self.corrector_steps = corrector_steps

        self.previous_noise = deque(maxlen=max(self.predictor_steps, self.corrector_steps))
        
    # copied from refiners.foundationals.latent_diffusion.solvers.dpm.py
    def _generate_timesteps(self) -> Tensor:
        """Generate the timesteps used by the solver.

        Note:
            We need to use numpy here because:

            - numpy.linspace(0,999,31)[15] is 499.49999999999994
            - torch.linspace(0,999,31)[15] is 499.5

            and we want the same result as the original codebase.
        """
        return tensor(
            np.linspace(0, self.num_train_timesteps - 1, self.num_inference_steps + 1).round().astype(int)[1:],
        ).flip(0)

    def rebuild(
        self: "SASolver",
        num_inference_steps: int | None,
        first_inference_step: int | None = None,
    ) -> "SASolver":
        """Rebuilds the solver with new parameters.

        Args:
            num_inference_steps: The number of inference steps.
            first_inference_step: The first inference step.
        """
        r = super().rebuild(
            num_inference_steps=num_inference_steps,
            first_inference_step=first_inference_step,
        )
        # r.last_step_first_order = self.last_step_first_order
        return r

    def _integrate_variance(self, step: int) -> Tensor:
        # integration of squared variance between SNR_t+1 and SNR_t using trapezoidal rule 
        # probably need to find a more accurate way to do so
        # give snr as index, meaning ?

        timestep = self.timesteps[step]
        next_timestep = self.timesteps[step+1]
        snr = int(self.signal_to_noise_ratios[timestep])
        next_snr = int(self.signal_to_noise_ratios[next_timestep])
        integrated_variance = (next_snr-snr) * (self.variance[snr]**2 + self.variance[next_snr]**2) * 0.5
        return integrated_variance
    
    def _b_hat(self, timestep: int):
        # Implementation in Appendix D
        # works for order 2
        factor = self.cumulative_scale_factors[timestep-1] / (self.signal_to_noise_ratios[timestep]-self.signal_to_noise_ratios[timestep-1])
        h = self.signal_to_noise_ratios[timestep+1]-self.signal_to_noise_ratios[timestep]
        previous_b_hat = factor * 0.5 * (1 + self.variance[timestep]**2) * h**2

        current_b_hat = self.cumulative_scale_factors[timestep+1] * (1-torch.exp(-h*(1+self.variance[timestep]**2))) - previous_b_hat
        return current_b_hat


    def multistep_sa_predictor(self, step: int, pred_step: int, x: Tensor, noise: Tensor,) -> Tensor:
        # Apply s-step SA-Predictor on inputs at timestep t_i

        timestep = self.timesteps[step]
        next_timestep = self.timesteps[step+1]
        integreted_var=self._integrate_variance(step)
        next_x = (self.noise_std[next_timestep]/self.noise_std[timestep]) * torch.exp(-integreted_var) * x

        F = []
        for j in range(pred_step):
            b = self._b_hat(self.timesteps[step-j])
            F.append(torch.multiply(b, self.previous_noise[-j-1]))
        F_sum = torch.stack(F, dim=0).sum(dim=0)

        sigma_tilde = self.noise_std[next_timestep] * torch.sqrt(1 - torch.exp(-2*self._integrate_variance(step)))
        G = sigma_tilde * noise

        next_x += F_sum + G

        return next_x
    
    def multistep_sa_corrector(self, step: int, corr_step: int, x: Tensor, latent_sa_predictor: Tensor, noise: Tensor,) -> Tensor:
        # Apply s-step SA-Corrector on inputs at timestep t_i

        timestep = self.timesteps[step]
        next_timestep = self.timesteps[step+1]
        integreted_var=self._integrate_variance(step)
        next_x = (self.noise_std[next_timestep]/self.noise_std[timestep]) * torch.exp(-integreted_var) * x

        next_x += self._b_hat(self.timesteps[step]) * latent_sa_predictor

        F = []
        for j in range(corr_step):
            b = self._b_hat(self.timesteps[step-j])
            F.append(torch.multiply(b, self.previous_noise[-j-1]))
        F_sum = torch.stack(F, dim=0).sum(dim=0)

        sigma_tilde = self.noise_std[next_timestep] * torch.sqrt(1 - torch.exp(-2*self._integrate_variance(step)))
        G = sigma_tilde * noise

        next_x += F_sum + G

        return next_x
        

    def __call__(self, x: Tensor, predicted_noise: Tensor, step: int, generator: Generator | None = None) -> Tensor:
        """Apply one step of the backward diffusion process using SA method

        Args:
            x: The input data.
            predicted_noise: The predicted noise.
            step: The current step.
            generator: The random number generator.

        Returns:
            The denoised version of the input data `x`.
        """

        warmup_steps = max(self.predictor_steps, self.corrector_steps)
        self.previous_noise.append(self.remove_noise(x, predicted_noise, step))

        G_noise = torch.rand(x.shape)
        # warm up
        if step <= warmup_steps:
            pred_step = min(step, self.predictor_steps)
            corr_step = min(step, self.corrector_steps)

            x_sa_predictor = self.multistep_sa_predictor(step, pred_step, x, G_noise)
            noise_sa_predictor = self.noise_prediction_function(x_sa_predictor)
            latent_sa_predictor = self.remove_noise(x_sa_predictor, noise_sa_predictor, step) # paragraph below eq. 5
            corrected_latent = self.multistep_sa_corrector(step, corr_step, x, latent_sa_predictor, G_noise)

        else: # step>warmup_steps
            x_sa_predictor = self.multistep_sa_predictor(step, self.predictor_steps, x, G_noise)
            noise_sa_predictor = self.noise_prediction_function(x_sa_predictor)
            latent_sa_predictor = self.remove_noise(x_sa_predictor, noise_sa_predictor, step)
            corrected_latent = self.multistep_sa_corrector(step, self.corrector_steps, x_sa_predictor, latent_sa_predictor, G_noise)

        return corrected_latent


    