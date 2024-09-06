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
).to("cuda")
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

- For self-attention guidance, `StableDiffusionSAGPipeline` has been used instead of the default pipeline.
- `expected_refonly.png` has been generated [with Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
- The following references have been generated with refiners itself (and inspected so that they look reasonable):
    - `expected_karras_random_init.png`,
    - `expected_inpainting_refonly.png`,
    - `expected_image_ip_adapter_woman.png`,
    - `expected_image_sdxl_ip_adapter_woman.png`
    - `expected_ip_adapter_controlnet.png`
    - `expected_t2i_adapter_xl_canny.png`
    - `expected_image_sdxl_ip_adapter_plus_woman.png`
    - `expected_cutecat_sdxl_ddim_random_init_sag.png`
    - `expected_cutecat_sdxl_euler_random_init.png`
    - `expected_restart.png`
    - `expected_freeu.png`
    - `expected_dropy_slime_9752.png`
    - `expected_sdxl_dpo_lora.png`
    - `expected_sdxl_multi_loras.png`
    - `expected_image_ip_adapter_multi.png`
    - `expected_controllora_CPDS.png`
    - `expected_controllora_PyraCanny.png`
    - `expected_controllora_PyraCanny+CPDS.png`
    - `expected_controllora_disabled.png`
    - `expected_style_aligned.png`
    - `expected_controlnet_canny_scale_decay.png` 
    - `expected_multi_diffusion_dpm.png`
    - `expected_multi_upscaler.png`
    - `expected_ic_light.png`

## Other images

- `cutecat_init.png` is generated with the same Diffusers script and prompt but with seed 1234.

- `kitchen_dog.png` is generated with the same Diffusers script and negative prompt, seed 12, positive prompt "a small brown dog, detailed high-quality professional image, sitting on a chair, in a kitchen".

- `expected_std_sde_random_init.png` is generated with the following code:

```python
import torch
from diffusers import StableDiffusionPipeline
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from refiners.fluxion.utils import manual_seed

diffusers_solver = DPMSolverMultistepScheduler.from_config(  # type: ignore
    {
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "algorithm_type": "sde-dpmsolver++",
        "use_karras_sigmas": False,
        "final_sigmas_type": "sigma_min",
        "euler_at_final": True,
    }
)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, scheduler=diffusers_solver)
pipe = pipe.to("cuda")
prompt = "a cute cat, detailed high-quality professional image"
negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
manual_seed(2)
image = pipe(prompt, negative_prompt=negative_prompt, guidance_scale=7.5).images[0]
```

- `expected_std_sde_karras_random_init.png` is generated with the following code (diffusers 0.30.2):

```python
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
```

- `kitchen_mask.png` is made manually.

- Controlnet guides have been manually generated (x) using open source software and models, namely:
    - Canny: opencv-python
    - Depth: https://github.com/isl-org/ZoeDepth
    - Lineart: https://github.com/lllyasviel/ControlNet-v1-1-nightly/tree/main/annotator/lineart
    - Normals: https://github.com/baegwangbin/surface_normal_uncertainty/tree/fe2b9f1
    - SAM: https://huggingface.co/spaces/mfidabel/controlnet-segment-anything

(x): excepted `fairy_guide_canny.png` which comes from [TencentARC/t2i-adapter-canny-sdxl-1.0](https://huggingface.co/TencentARC/t2i-adapter-canny-sdxl-1.0)

- `cyberpunk_guide.png` [comes from Lexica](https://lexica.art/prompt/5ba40855-0d0c-4322-8722-51115985f573).

- `inpainting-mask.png`, `inpainting-scene.png` and `inpainting-target.png` have been generated as follows:
    - `inpainting-mask.png`: negated version of a mask computed with [SAM](https://github.com/facebookresearch/segment-anything) automatic mask generation using the `vit_h` checkpoint
    - `inpainting-scene.png`: cropped-to-square-and-resized version of https://unsplash.com/photos/RCz6eSVPGYU by @jannerboy62
    - `inpainting-target.png`: computed with `convert <(convert -size 512x512 xc:white png:-) kitchen_dog.png <(convert inpainting-mask.png -negate png:-) -compose Over -composite inpainting-target.png`

- `woman.png` [comes from tencent-ailab/IP-Adapter](https://github.com/tencent-ailab/IP-Adapter/blob/8b96670cc5c8ef00278b42c0c7b62fe8a74510b9/assets/images/woman.png).

- `statue.png` [comes from tencent-ailab/IP-Adapter](https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/assets/images/statue.png).

- `cutecat_guide_PyraCanny.png` and `cutecat_guide_CPDS.png` were [generated inside Fooocus](https://github.com/lllyasviel/Fooocus/blob/e8d88d3e250e541c6daf99d6ef734e8dc3cfdc7f/extras/preprocessors.py).

- `low_res_dog.png` and `expected_controlnet_tile.png` are taken from Diffusers [documentation](https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile/tree/main/images), respectively named
`original.png` and `output.png`.

- `clarity_input_example.png` is taken from the [Replicate demo](https://replicate.com/philz1337x/clarity-upscaler/examples) of the Clarity upscaler.

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

## Textual Inversion

- `expected_textual_inversion_random_init.png` has been generated with StableDiffusionPipeline, e.g.:

```py
import torch

from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.load_textual_inversion("sd-concepts-library/gta5-artwork")

prompt = "a cute cat on a <gta5-artwork>"
negative_prompt = ""

torch.manual_seed(2)
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
)

output.images[0].save("expected_textual_inversion_random_init.png")
```
