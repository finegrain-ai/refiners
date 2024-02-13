from typing import cast
from warnings import warn

import pytest
from torch import Tensor, allclose, device as Device, equal, isclose, randn

from refiners.fluxion import manual_seed
from refiners.foundationals.latent_diffusion.solvers import DDIM, DDPM, DPMSolver, Euler, NoiseSchedule


def test_ddpm_diffusers():
    from diffusers import DDPMScheduler  # type: ignore

    diffusers_scheduler = DDPMScheduler(beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012)
    diffusers_scheduler.set_timesteps(1000)
    refiners_scheduler = DDPM(num_inference_steps=1000)

    assert equal(diffusers_scheduler.timesteps, refiners_scheduler.timesteps)


@pytest.mark.parametrize("n_steps, last_step_first_order", [(5, False), (5, True), (30, False), (30, True)])
def test_dpm_solver_diffusers(n_steps: int, last_step_first_order: bool):
    from diffusers import DPMSolverMultistepScheduler as DiffuserScheduler  # type: ignore

    manual_seed(0)

    diffusers_scheduler = DiffuserScheduler(
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        beta_end=0.012,
        lower_order_final=False,
        euler_at_final=last_step_first_order,
        final_sigmas_type="sigma_min",  # default before Diffusers 0.26.0
    )
    diffusers_scheduler.set_timesteps(n_steps)
    refiners_scheduler = DPMSolver(num_inference_steps=n_steps, last_step_first_order=last_step_first_order)

    sample = randn(1, 3, 32, 32)
    predicted_noise = randn(1, 3, 32, 32)

    for step, timestep in enumerate(diffusers_scheduler.timesteps):
        diffusers_output = cast(Tensor, diffusers_scheduler.step(predicted_noise, timestep, sample).prev_sample)  # type: ignore
        refiners_output = refiners_scheduler(x=sample, predicted_noise=predicted_noise, step=step)
        assert allclose(diffusers_output, refiners_output, rtol=0.01), f"outputs differ at step {step}"


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
    refiners_scheduler = DDIM(num_inference_steps=30)

    sample = randn(1, 4, 32, 32)
    predicted_noise = randn(1, 4, 32, 32)

    for step, timestep in enumerate(diffusers_scheduler.timesteps):
        diffusers_output = cast(Tensor, diffusers_scheduler.step(predicted_noise, timestep, sample).prev_sample)  # type: ignore
        refiners_output = refiners_scheduler(x=sample, predicted_noise=predicted_noise, step=step)

        assert allclose(diffusers_output, refiners_output, rtol=0.01), f"outputs differ at step {step}"


def test_euler_diffusers():
    from diffusers import EulerDiscreteScheduler  # type: ignore

    manual_seed(0)
    diffusers_scheduler = EulerDiscreteScheduler(
        beta_end=0.012,
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        num_train_timesteps=1000,
        steps_offset=1,
        timestep_spacing="linspace",
        use_karras_sigmas=False,
    )
    diffusers_scheduler.set_timesteps(30)
    refiners_scheduler = Euler(num_inference_steps=30)

    sample = randn(1, 4, 32, 32)
    predicted_noise = randn(1, 4, 32, 32)

    ref_init_noise_sigma = diffusers_scheduler.init_noise_sigma  # type: ignore
    assert isinstance(ref_init_noise_sigma, Tensor)
    assert isclose(ref_init_noise_sigma, refiners_scheduler.init_noise_sigma), "init_noise_sigma differ"

    for step, timestep in enumerate(diffusers_scheduler.timesteps):
        diffusers_output = cast(Tensor, diffusers_scheduler.step(predicted_noise, timestep, sample).prev_sample)  # type: ignore
        refiners_output = refiners_scheduler(x=sample, predicted_noise=predicted_noise, step=step)

        assert allclose(diffusers_output, refiners_output, rtol=0.02), f"outputs differ at step {step}"


def test_scheduler_remove_noise():
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
    refiners_scheduler = DDIM(num_inference_steps=30)

    sample = randn(1, 4, 32, 32)
    noise = randn(1, 4, 32, 32)

    for step, timestep in enumerate(diffusers_scheduler.timesteps):
        diffusers_output = cast(Tensor, diffusers_scheduler.step(noise, timestep, sample).pred_original_sample)  # type: ignore
        refiners_output = refiners_scheduler.remove_noise(x=sample, noise=noise, step=step)

        assert allclose(diffusers_output, refiners_output, rtol=0.01), f"outputs differ at step {step}"


def test_scheduler_device(test_device: Device):
    if test_device.type == "cpu":
        warn("not running on CPU, skipping")
        pytest.skip()

    scheduler = DDIM(num_inference_steps=30, device=test_device)
    x = randn(1, 4, 32, 32, device=test_device)
    noise = randn(1, 4, 32, 32, device=test_device)
    noised = scheduler.add_noise(x, noise, scheduler.first_inference_step)
    assert noised.device == test_device


@pytest.mark.parametrize("noise_schedule", [NoiseSchedule.UNIFORM, NoiseSchedule.QUADRATIC, NoiseSchedule.KARRAS])
def test_scheduler_noise_schedules(noise_schedule: NoiseSchedule, test_device: Device):
    scheduler = DDIM(num_inference_steps=30, device=test_device, noise_schedule=noise_schedule)
    assert len(scheduler.scale_factors) == 1000
    assert scheduler.scale_factors[0] == 1 - scheduler.initial_diffusion_rate
    assert scheduler.scale_factors[-1] == 1 - scheduler.final_diffusion_rate
