import torch
from diffusers import StableDiffusionPipeline

# from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from refiners.fluxion.utils import manual_seed

# diffusers_solver = DDIMScheduler.from_config( # type: ignore
#     {
#         "_class_name": "PNDMScheduler",
#         "_diffusers_version": "0.6.0",
#         "beta_end": 0.012,
#         "beta_schedule": "scaled_linear",
#         "beta_start": 0.00085,
#         "num_train_timesteps": 1000,
#         "set_alpha_to_one": False,
#         "skip_prk_steps": True,
#         "steps_offset": 1,
#         "trained_betas": None,
#         "clip_sample": False,
#         # "use_karras_sigmas": True,
#         # "algorithm": "dpmsolver++",
#     }
# )


model_id = "botp/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, scheduler=diffusers_solver)
pipe = pipe.to("cuda:1")

prompt = "a cute cat, detailed high-quality professional image"
negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
manual_seed(2)
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("diffusers-ddim.png")
