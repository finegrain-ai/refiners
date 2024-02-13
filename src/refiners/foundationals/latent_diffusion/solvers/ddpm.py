from torch import Generator, Tensor, arange, device as Device

from refiners.foundationals.latent_diffusion.solvers.solver import Solver


class DDPM(Solver):
    """Denoising Diffusion Probabilistic Model (DDPM) solver.

    Warning:
        Only used for training Latent Diffusion models.
        Cannot be called.

    See [[arXiv:2006.11239] Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) for more details.
    """

    def __init__(
        self,
        num_inference_steps: int,
        num_train_timesteps: int = 1_000,
        initial_diffusion_rate: float = 8.5e-4,
        final_diffusion_rate: float = 1.2e-2,
        first_inference_step: int = 0,
        device: Device | str = "cpu",
    ) -> None:
        """Initializes a new DDPM solver.

        Args:
            num_inference_steps: The number of inference steps.
            num_train_timesteps: The number of training timesteps.
            initial_diffusion_rate: The initial diffusion rate.
            final_diffusion_rate: The final diffusion rate.
            first_inference_step: The first inference step.
            device: The PyTorch device to use.
        """
        super().__init__(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=num_train_timesteps,
            initial_diffusion_rate=initial_diffusion_rate,
            final_diffusion_rate=final_diffusion_rate,
            first_inference_step=first_inference_step,
            device=device,
        )

    def _generate_timesteps(self) -> Tensor:
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = arange(start=0, end=self.num_inference_steps, step=1, device=self.device) * step_ratio
        return timesteps.flip(0)

    def __call__(self, x: Tensor, predicted_noise: Tensor, step: int, generator: Generator | None = None) -> Tensor:
        raise NotImplementedError
