import torch
from diffusers import StableDiffusionPipeline
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from refiners.fluxion.utils import manual_seed

model_id = "botp/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cuda:1")


config = {**pipe.scheduler.config}
config["use_karras_sigmas"] = True
config["algorithm_type"] = "sde-dpmsolver++"
pipe.scheduler = DPMSolverMultistepScheduler.from_config(config)

prompt = "a cute cat, detailed high-quality professional image"
negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
manual_seed(2)
image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=18, guidance_scale=7.5).images[0]

image.save("diffusers-sde-dpm-karras.png")
