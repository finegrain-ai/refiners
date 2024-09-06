import torch
from PIL import Image

from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline

weights_path = "weights"
device = torch.device("cuda:1")
n_steps = 30

pipe = StableDiffusionPipeline.from_pretrained(
    "botp/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None,
).to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

prompt = "a cute cat, detailed high-quality professional image"
negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"


torch.manual_seed(2)
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=n_steps,
    guidance_scale=7.5,
)

output.images[0].save("output-diffusers-diffusion.png")
