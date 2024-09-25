import itertools
from typing import cast
from warnings import warn

import pytest
import torch
from torch import Tensor, device as Device

from refiners.fluxion import manual_seed
from refiners.foundationals.latent_diffusion.solvers import (
    DDIM,
    DDPM,
    DPMSolver,
    Euler,
    FrankenSolver,
    LCMSolver,
    ModelPredictionType,
    NoiseSchedule,
    Solver,
    SolverParams,
    TimestepSpacing,
)


def test_ddpm_diffusers():
    from diffusers import DDPMScheduler  # type: ignore

    diffusers_scheduler = DDPMScheduler(beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012)
    diffusers_scheduler.set_timesteps(1000)
    solver = DDPM(num_inference_steps=1000)
    assert torch.equal(diffusers_scheduler.timesteps, solver.timesteps)


@pytest.mark.parametrize(
    "n_steps, last_step_first_order, sde_variance, use_karras_sigmas",
    list(itertools.product([5, 30], [False, True], [0.0, 1.0], [False, True])),
)
def test_dpm_solver_diffusers(n_steps: int, last_step_first_order: bool, sde_variance: float, use_karras_sigmas: bool):
    from diffusers import DPMSolverMultistepScheduler as DiffuserScheduler  # type: ignore

    manual_seed(0)

    diffusers_scheduler = DiffuserScheduler(
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        beta_end=0.012,
        lower_order_final=False,
        euler_at_final=last_step_first_order,
        final_sigmas_type="sigma_min",  # default before Diffusers 0.26.0
        algorithm_type="sde-dpmsolver++" if sde_variance == 1.0 else "dpmsolver++",
        use_karras_sigmas=use_karras_sigmas,
    )
    diffusers_scheduler.set_timesteps(n_steps)
    solver = DPMSolver(
        num_inference_steps=n_steps,
        last_step_first_order=last_step_first_order,
        params=SolverParams(
            sde_variance=sde_variance,
            sigma_schedule=NoiseSchedule.KARRAS if use_karras_sigmas else None,
        ),
    )
    assert torch.equal(solver.timesteps, diffusers_scheduler.timesteps)

    sample = torch.randn(1, 3, 32, 32)
    predicted_noise = torch.randn(1, 3, 32, 32)

    manual_seed(37)
    diffusers_outputs: list[Tensor] = [
        cast(Tensor, diffusers_scheduler.step(predicted_noise, timestep, sample).prev_sample)  # type: ignore
        for timestep in diffusers_scheduler.timesteps
    ]

    manual_seed(37)
    refiners_outputs = [solver(x=sample, predicted_noise=predicted_noise, step=step) for step in range(n_steps)]

    if use_karras_sigmas:
        atol = 1e-4
    elif sde_variance == 1.0:
        atol = 1e-6
    else:
        atol = 1e-8
    for step, (diffusers_output, refiners_output) in enumerate(zip(diffusers_outputs, refiners_outputs)):
        assert torch.allclose(diffusers_output, refiners_output, rtol=0.01, atol=atol), f"outputs differ at step {step}"


def test_ddim_diffusers():
    from diffusers import DDIMScheduler  # type: ignore

    manual_seed(0)

    diffusers_scheduler = DDIMScheduler(
        beta_end=0.012,
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        num_train_timesteps=1000,
        steps_offset=1,
        clip_sample=False,
    )
    diffusers_scheduler.set_timesteps(30)
    solver = DDIM(num_inference_steps=30)
    assert torch.equal(solver.timesteps, diffusers_scheduler.timesteps)

    sample = torch.randn(1, 4, 32, 32)
    predicted_noise = torch.randn(1, 4, 32, 32)

    for step, timestep in enumerate(diffusers_scheduler.timesteps):
        diffusers_output = cast(Tensor, diffusers_scheduler.step(predicted_noise, timestep, sample).prev_sample)  # type: ignore
        refiners_output = solver(x=sample, predicted_noise=predicted_noise, step=step)

        assert torch.allclose(diffusers_output, refiners_output, rtol=0.01), f"outputs differ at step {step}"


@pytest.mark.parametrize("model_prediction_type", [ModelPredictionType.NOISE, ModelPredictionType.SAMPLE])
def test_euler_diffusers(model_prediction_type: ModelPredictionType):
    from diffusers import EulerDiscreteScheduler  # type: ignore

    manual_seed(0)
    diffusers_prediction_type = "epsilon" if model_prediction_type == ModelPredictionType.NOISE else "sample"
    diffusers_scheduler = EulerDiscreteScheduler(
        beta_end=0.012,
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        num_train_timesteps=1000,
        steps_offset=1,
        timestep_spacing="linspace",
        use_karras_sigmas=False,
        prediction_type=diffusers_prediction_type,
    )
    diffusers_scheduler.set_timesteps(30)
    solver = Euler(num_inference_steps=30, params=SolverParams(model_prediction_type=model_prediction_type))
    assert torch.equal(solver.timesteps, diffusers_scheduler.timesteps)

    sample = torch.randn(1, 4, 32, 32)
    predicted_noise = torch.randn(1, 4, 32, 32)

    ref_init_noise_sigma = diffusers_scheduler.init_noise_sigma  # type: ignore
    assert isinstance(ref_init_noise_sigma, Tensor)
    assert torch.isclose(ref_init_noise_sigma, solver.init_noise_sigma), "init_noise_sigma differ"

    for step, timestep in enumerate(diffusers_scheduler.timesteps):
        diffusers_output = cast(Tensor, diffusers_scheduler.step(predicted_noise, timestep, sample).prev_sample)  # type: ignore
        refiners_output = solver(x=sample, predicted_noise=predicted_noise, step=step)

        assert torch.allclose(diffusers_output, refiners_output, rtol=0.02), f"outputs differ at step {step}"


def test_franken_diffusers():
    from diffusers import EulerDiscreteScheduler  # type: ignore

    manual_seed(0)
    params = {
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "num_train_timesteps": 1000,
        "steps_offset": 1,
        "timestep_spacing": "linspace",
        "use_karras_sigmas": False,
    }

    diffusers_scheduler = EulerDiscreteScheduler(**params)  # type: ignore
    diffusers_scheduler.set_timesteps(30)

    diffusers_scheduler_2 = EulerDiscreteScheduler(**params)  # type: ignore
    solver = FrankenSolver(lambda: diffusers_scheduler_2, num_inference_steps=30)
    assert torch.equal(solver.timesteps, diffusers_scheduler.timesteps)

    sample = torch.randn(1, 4, 32, 32)
    predicted_noise = torch.randn(1, 4, 32, 32)

    ref_init_noise_sigma = diffusers_scheduler.init_noise_sigma  # type: ignore
    assert isinstance(ref_init_noise_sigma, Tensor)
    init_noise_sigma = solver.scale_model_input(torch.tensor(1), step=-1)
    assert torch.equal(ref_init_noise_sigma, init_noise_sigma), "init_noise_sigma differ"

    for step, timestep in enumerate(diffusers_scheduler.timesteps):
        diffusers_output = cast(Tensor, diffusers_scheduler.step(predicted_noise, timestep, sample).prev_sample)  # type: ignore
        refiners_output = solver(x=sample, predicted_noise=predicted_noise, step=step)

        assert torch.equal(diffusers_output, refiners_output), f"outputs differ at step {step}"


def test_lcm_diffusers():
    from diffusers import LCMScheduler  # type: ignore

    manual_seed(0)

    # LCMScheduler is stochastic, make sure we use identical generators
    diffusers_generator = torch.Generator().manual_seed(42)
    refiners_generator = torch.Generator().manual_seed(42)

    diffusers_scheduler = LCMScheduler()
    diffusers_scheduler.set_timesteps(4)
    solver = LCMSolver(num_inference_steps=4)
    assert torch.equal(solver.timesteps, diffusers_scheduler.timesteps)

    sample = torch.randn(1, 4, 32, 32)
    predicted_noise = torch.randn(1, 4, 32, 32)

    for step, timestep in enumerate(diffusers_scheduler.timesteps):
        alpha_prod_t = diffusers_scheduler.alphas_cumprod[timestep]
        diffusers_noise_ratio = (1 - alpha_prod_t).sqrt()
        diffusers_scale_factor = alpha_prod_t.sqrt()

        refiners_scale_factor = solver.cumulative_scale_factors[timestep]
        refiners_noise_ratio = solver.noise_std[timestep]

        assert refiners_scale_factor == diffusers_scale_factor
        assert refiners_noise_ratio == diffusers_noise_ratio

        d_out = diffusers_scheduler.step(predicted_noise, timestep, sample, generator=diffusers_generator)  # type: ignore
        diffusers_output = cast(Tensor, d_out.prev_sample)  # type: ignore

        refiners_output = solver(
            x=sample,
            predicted_noise=predicted_noise,
            step=step,
            generator=refiners_generator,
        )

        assert torch.allclose(refiners_output, diffusers_output, rtol=0.01), f"outputs differ at step {step}"


def test_solver_remove_noise():
    from diffusers import DDIMScheduler  # type: ignore

    manual_seed(0)

    diffusers_scheduler = DDIMScheduler(
        beta_end=0.012,
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        num_train_timesteps=1000,
        steps_offset=1,
        clip_sample=False,
    )
    diffusers_scheduler.set_timesteps(30)
    solver = DDIM(num_inference_steps=30)

    sample = torch.randn(1, 4, 32, 32)
    noise = torch.randn(1, 4, 32, 32)

    for step, timestep in enumerate(diffusers_scheduler.timesteps):
        diffusers_output = cast(Tensor, diffusers_scheduler.step(noise, timestep, sample).pred_original_sample)  # type: ignore
        refiners_output = solver.remove_noise(x=sample, noise=noise, step=step)

        assert torch.allclose(diffusers_output, refiners_output, rtol=0.01), f"outputs differ at step {step}"


def test_solver_device(test_device: Device):
    if test_device.type == "cpu":
        warn("not running on CPU, skipping")
        pytest.skip()

    scheduler = DDIM(num_inference_steps=30, device=test_device)
    x = torch.randn(1, 4, 32, 32, device=test_device)
    noise = torch.randn(1, 4, 32, 32, device=test_device)
    noised = scheduler.add_noise(x, noise, scheduler.first_inference_step)
    assert noised.device == test_device


def test_solver_add_noise(test_device: Device):
    scheduler = DDIM(num_inference_steps=30, device=test_device)
    latent = torch.randn(1, 4, 32, 32, device=test_device)
    noise = torch.randn(1, 4, 32, 32, device=test_device)
    noised = scheduler.add_noise(
        x=latent,
        noise=noise,
        step=0,
    )
    noised_double = scheduler.add_noise(
        x=latent.repeat(2, 1, 1, 1),
        noise=noise.repeat(2, 1, 1, 1),
        step=[0, 0],
    )
    assert torch.allclose(noised, noised_double[0])
    assert torch.allclose(noised, noised_double[1])


@pytest.mark.parametrize("noise_schedule", [NoiseSchedule.UNIFORM, NoiseSchedule.QUADRATIC, NoiseSchedule.KARRAS])
def test_solver_noise_schedules(noise_schedule: NoiseSchedule, test_device: Device):
    scheduler = DDIM(
        num_inference_steps=30,
        params=SolverParams(noise_schedule=noise_schedule),
        device=test_device,
    )
    assert len(scheduler.scale_factors) == 1000
    assert scheduler.scale_factors[0] == 1 - scheduler.params.initial_diffusion_rate
    assert scheduler.scale_factors[-1] == 1 - scheduler.params.final_diffusion_rate


def test_solver_timestep_spacing():
    # Tests we get the results from [[arXiv:2305.08891] Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/abs/2305.08891) table 2.
    linspace_int = Solver.generate_timesteps(
        spacing=TimestepSpacing.LINSPACE_ROUNDED,
        num_inference_steps=10,
        num_train_timesteps=1000,
        offset=1,
    )
    assert torch.equal(linspace_int, torch.tensor([1000, 889, 778, 667, 556, 445, 334, 223, 112, 1]))

    leading = Solver.generate_timesteps(
        spacing=TimestepSpacing.LEADING,
        num_inference_steps=10,
        num_train_timesteps=1000,
        offset=1,
    )
    assert torch.equal(leading, torch.tensor([901, 801, 701, 601, 501, 401, 301, 201, 101, 1]))

    trailing = Solver.generate_timesteps(
        spacing=TimestepSpacing.TRAILING,
        num_inference_steps=10,
        num_train_timesteps=1000,
        offset=1,
    )
    assert torch.equal(trailing, torch.tensor([1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]))


def test_dpm_bfloat16(test_device: Device):
    if test_device.type == "cpu":
        warn("not running on CPU, skipping")
        pytest.skip()

    n_steps = 5
    manual_seed(0)

    solver_f32 = DPMSolver(num_inference_steps=n_steps, dtype=torch.float32)
    solver_bf16 = DPMSolver(num_inference_steps=n_steps, dtype=torch.bfloat16)
    assert torch.equal(solver_bf16.timesteps, solver_f32.timesteps)
