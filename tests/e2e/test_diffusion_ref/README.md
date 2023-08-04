# Note about this data

## Expected outputs

`expected_*.png` files are the output of the same diffusion run with a different codebase, usually diffusers with the same settings as us (`DPMSolverMultistepScheduler`, VAE [patched to remove randomness](#vae-without-randomness), same seed...).

For instance here is how we generate `expected_std_random_init.png`:

```py
import torch

from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
).to("cuda)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

prompt = "a cute cat, detailed high-quality professional image"
negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"

torch.manual_seed(2)
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
)

output.images[0].save("std_random_init_expected.png")
```

Special cases:

- `expected_refonly.png` has been generated [with Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
- `expected_inpainting_refonly.png` has been generated with refiners itself (and inspected so that it looks reasonable).

## Other images

- `cutecat_init.png` is generated with the same Diffusers script and prompt but with seed 1234.

- `kitchen_dog.png` is generated with the same Diffusers script and negative prompt, seed 12, positive prompt "a small brown dog, detailed high-quality professional image, sitting on a chair, in a kitchen".

- `kitchen_mask.png` is made manually.

- Controlnet guides have been manually generated using open source software and models, namely:
    - Canny: opencv-python
    - Depth: https://github.com/isl-org/ZoeDepth
    - Lineart: https://github.com/lllyasviel/ControlNet-v1-1-nightly/tree/main/annotator/lineart
    - Normals: https://github.com/baegwangbin/surface_normal_uncertainty/tree/fe2b9f1
    - SAM: https://huggingface.co/spaces/mfidabel/controlnet-segment-anything

- `cyberpunk_guide.png` [comes from Lexica](https://lexica.art/prompt/5ba40855-0d0c-4322-8722-51115985f573).

- `inpainting-mask.png`, `inpainting-scene.png` and `inpainting-target.png` have been generated as follows:
    - `inpainting-mask.png`: negated version of a mask computed with [SAM](https://github.com/facebookresearch/segment-anything) automatic mask generation using the `vit_h` checkpoint
    - `inpainting-scene.png`: cropped-to-square-and-resized version of https://unsplash.com/photos/RCz6eSVPGYU by @jannerboy62
    - `inpainting-target.png`: computed with `convert <(convert -size 512x512 xc:white png:-) kitchen_dog.png <(convert inpainting-mask.png -negate png:-) -compose Over -composite inpainting-target.png`

## VAE without randomness

```diff
--- a/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py
+++ b/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py
@@ -524,13 +524,8 @@ class StableDiffusionImg2ImgPipeline(DiffusionPipeline):
                 f" size of {batch_size}. Make sure the batch size matches the length of the generators."
             )

-        if isinstance(generator, list):
-            init_latents = [
-                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
-            ]
-            init_latents = torch.cat(init_latents, dim=0)
-        else:
-            init_latents = self.vae.encode(image).latent_dist.sample(generator)
+        init_latents = [self.vae.encode(image[i : i + 1]).latent_dist.mean for i in range(batch_size)]
+        init_latents = torch.cat(init_latents, dim=0)

         init_latents = self.vae.config.scaling_factor * init_latents
```
