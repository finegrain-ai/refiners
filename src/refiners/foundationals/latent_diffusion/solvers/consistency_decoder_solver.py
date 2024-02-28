import math
import torch
from torch import Generator, Tensor, device as Device, dtype as Dtype, float32
from typing import List
from refiners.foundationals.latent_diffusion.solvers.solver import NoiseSchedule, Solver
from refiners.foundationals.latent_diffusion.solvers.solver import SolverParams


# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(
    num_diffusion_timesteps: int,
    max_beta :float =0.999,
    alpha_transform_type: str="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t: float):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t: float):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas : List[float] = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class ConsistencyDecoderSolver(Solver):
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
            num_inference_steps: The number of inference steps.
            num_train_timesteps: The number of training timesteps.
            initial_diffusion_rate: The initial diffusion rate.
            final_diffusion_rate: The final diffusion rate.
            noise_schedule: The noise schedule.
            first_inference_step: The first inference step.
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """

        super().__init__(
            num_inference_steps=num_inference_steps,
            first_inference_step=first_inference_step,
            params=params,
            device=device,
            dtype=dtype,
        )

        sigma_data: float = 0.5
        betas = betas_for_alpha_bar(self.params.num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)
        sigmas = torch.sqrt(1.0 / alphas_cumprod - 1)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.c_skip = sqrt_recip_alphas_cumprod * sigma_data**2 / (sigmas**2 + sigma_data**2).to(device)
        self.c_out = (sigmas * sigma_data / (sigmas**2 + sigma_data**2) ** 0.5).to(device)
        self.c_in = (sqrt_recip_alphas_cumprod / (sigmas**2 + sigma_data**2) ** 0.5).to(device)

    @property
    def init_noise_sigma(self) -> Tensor:
        return self.sqrt_one_minus_alphas_cumprod[self.timesteps[0]]

    def scale_model_input(self, x: Tensor, step: int) -> Tensor:
        return x * self.c_in[step]

    def _generate_timesteps(self) -> Tensor:
        assert self.num_inference_steps == 2
        timesteps = torch.tensor([1008, 512], dtype=torch.long, device=self.device)
        return timesteps

    def __call__(self, x: Tensor, predicted_noise: Tensor, step: int, generator: Generator | None = None) -> Tensor:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from the learned diffusion model.
            timestep (`float`):
                The current timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
        """
        x_0 = self.c_out[step] * predicted_noise + self.c_skip[step] * x

        timestep_idx = torch.where(torch.eq(self.timesteps, step))[0][0]

        if timestep_idx == len(self.timesteps) - 1:
            prev_sample = x_0
        else:
            noise = torch.randn(x_0.shape, device=self.device)
            prev_sample = (
                self.sqrt_alphas_cumprod[self.timesteps[timestep_idx + 1]].to(x_0.dtype) * x_0
                + self.sqrt_one_minus_alphas_cumprod[self.timesteps[timestep_idx + 1]].to(x_0.dtype) * noise
            )

        return prev_sample
