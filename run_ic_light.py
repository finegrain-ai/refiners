from pathlib import Path

import torch
from PIL import Image

from refiners.fluxion.utils import load_from_safetensors, no_grad
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
from refiners.foundationals.latent_diffusion.stable_diffusion_1.ic_light import ICLightBackground
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet

no_grad().__enter__()

device = "cuda" if torch.cuda.is_available() else "cpu"
sd = ICLightBackground(
    patch_weights=load_from_safetensors("iclight_sd15_fbc-refiners.safetensors", device=device),
    unet=SD1UNet(in_channels=4, device=device).load_from_safetensors("realistic-vision-v51-unet.safetensors"),
    clip_text_encoder=CLIPTextEncoderL(device=device).load_from_safetensors(
        "realistic-vision-v51-text_encoder.safetensors"
    ),
    lda=SD1Autoencoder(device=device).load_from_safetensors("realistic-vision-v51-autoencoder.safetensors"),
    device=device,
)

prompt = "porcelaine mug, 4k high quality, soft lighting, high-quality professional image"
negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
clip_text_embedding = sd.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

data_test = Path.home() / "data_test"
uid = "ic_out"
foreground_image = Image.open(f"{uid}.png").convert("RGB")
mask = Image.open(f"{uid}.mask.png").resize(foreground_image.size).convert("L").point(lambda x: 255 if x > 0 else 0)


sd.set_foreground_condition(foreground_image, mask=mask, use_rescaled_image=True)
sd.set_background_condition(foreground_image)

x = torch.randn(1, 4, foreground_image.height // 8, foreground_image.width // 8, device=device)

for step in sd.steps:
    x = sd(
        x,
        step=step,
        clip_text_embedding=clip_text_embedding,
        condition_scale=3,
    )
predicted_image = sd.lda.latents_to_image(x)

predicted_image.save("ic-light-output.png")
