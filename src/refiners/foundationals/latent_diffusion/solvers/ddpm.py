import dataclasses

from torch import Generator, Tensor, device as Device

from refiners.foundationals.latent_diffusion.solvers.solver import (
    BaseSolverParams,
    ModelPredictionType,
    Solver,
    TimestepSpacing,
)


class DDPM(Solver):
    """Denoising Diffusion Probabilistic Model (DDPM) solver.

    Warning:
        Only used for training Latent Diffusion models.
        Cannot be called.

    See [[arXiv:2006.11239] Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) for more details.
    """

    default_params = dataclasses.replace(
        Solver.default_params,
        timesteps_spacing=TimestepSpacing.LEADING,
    )

    def __init__(
        self,
        num_inference_steps: int,
        first_inference_step: int = 0,
        params: BaseSolverParams | None = None,
        device: Device | str = "cpu",
    ) -> None:
        """Initializes a new DDPM solver.

        Args:
            num_inference_steps: The number of inference steps to perform.
            first_inference_step: The first inference step to perform.
            params: The common parameters for solvers.
            device: The PyTorch device to use.
        """

        if params and params.model_prediction_type not in (ModelPredictionType.NOISE, None):
            raise NotImplementedError

        super().__init__(
            num_inference_steps=num_inference_steps,
            first_inference_step=first_inference_step,
            params=params,
            device=device,
        )

    def __call__(self, x: Tensor, predicted_noise: Tensor, step: int, generator: Generator | None = None) -> Tensor:
        raise NotImplementedError
