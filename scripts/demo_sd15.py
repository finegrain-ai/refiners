import argparse
import logging
import os
import random
import re

import torch

from refiners.fluxion.utils import manual_seed, download_diffusers_weights, default_device
from refiners.foundationals.latent_diffusion import StableDiffusion_1

logger = logging.getLogger(__name__)


def load_stable_diffusion_from_diffusers_repo(diffusers_repo="runwayml/stable-diffusion-v1-5", device=default_device()):
    sd = StableDiffusion_1(device=device, dtype=torch.float16)

    logger.debug("Loading VAE")
    vae_weights_path = download_diffusers_weights(
        repo=diffusers_repo, sub="vae", filename="diffusion_pytorch_model.safetensors"
    )
    sd.lda.load_from_safetensors(vae_weights_path)

    logger.debug("Loading text encoder")
    text_encoder_weights_path = download_diffusers_weights(
        repo=diffusers_repo, sub="text_encoder", filename="model.safetensors"
    )
    sd.clip_text_encoder.load_from_safetensors(text_encoder_weights_path)

    logger.debug("Loading UNet")
    unet_weights_path = download_diffusers_weights(
        repo=diffusers_repo, sub="unet", filename="diffusion_pytorch_model.safetensors"
    )
    sd.unet.load_from_safetensors(unet_weights_path)
    logger.debug(f"'{diffusers_repo}' Loaded")
    return sd


DEFAULT_NEGATIVE_PROMPT = (
    "Ugly, duplication, duplicates, mutilation, deformed, mutilated, mutation, twisted body, disfigured, bad anatomy,"
    " out of frame, extra fingers, mutated hands, poorly drawn hands, extra limbs, malformed limbs, missing arms, extra"
    " arms, missing legs, extra legs, mutated hands, extra hands, fused fingers, missing fingers, extra fingers, long"
    " neck, small head, closed eyes, rolling eyes, weird eyes, smudged face, blurred face, poorly drawn face, mutation,"
    " mutilation, cloned face, strange mouth, grainy, blurred, blurry, writing, calligraphy, signature, text,"
    " watermark, bad art,"
)
DEFAULT_PROMPT = "a cute cat, detailed high-quality professional image"


def make_images(
    prompt=DEFAULT_PROMPT,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    n_steps=50,
    diffusers_repo="runwayml/stable-diffusion-v1-5",
    seed=None,
    n_images=4,
    guidance_scale=7.5,
):
    sd = load_stable_diffusion_from_diffusers_repo(diffusers_repo=diffusers_repo)
    base_seed = random.randint(0, 2**32 - 1) if seed is None else seed
    with torch.no_grad():
        clip_text_embedding = sd.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)
        sd.set_num_inference_steps(n_steps)

        for image_num in range(n_images):
            print(f"Generating image {image_num + 1} of {n_images}")
            seed = base_seed + image_num
            manual_seed(seed)
            x = torch.randn(1, 4, 64, 64, device=default_device(), dtype=torch.float16)

            for step in sd.steps:
                x = sd(
                    x,
                    step=step,
                    clip_text_embedding=clip_text_embedding,
                    condition_scale=guidance_scale,
                )
            predicted_image = sd.lda.decode_latents(x)
            santized_repo = diffusers_repo.replace("/", "_")
            sanitized_prompt = re.sub(r"[^a-zA-Z0-9]+", "_", prompt)
            dest_path = f"outputs/{santized_repo}_{seed}_{sanitized_prompt[:50]}.jpg"
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            predicted_image.save(dest_path)


class Args(argparse.Namespace):
    prompt: str
    negative_prompt: str
    n_steps: int
    diffusers_repo: str
    seed: int
    n_images: int
    guidance_scale: float


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate some images")
    parser.add_argument(
        "--prompt",
        type=str,
        dest="prompt",
        default=DEFAULT_PROMPT,
        help="prompt to use",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        dest="negative_prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="negative prompt to use",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        dest="n_steps",
        default=50,
        help="number of steps",
    )
    parser.add_argument(
        "--diffusers-repo",
        type=str,
        dest="diffusers_repo",
        default="runwayml/stable-diffusion-v1-5",
        help="diffusers repo to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        dest="seed",
        default=None,
        help="seed to use",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        dest="n_images",
        default=4,
        help="number of images to generate",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        dest="guidance_scale",
        default=7.5,
        help="guidance scale to use",
    )
    args = parser.parse_args(namespace=Args())
    make_images(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        n_steps=args.n_steps,
        diffusers_repo=args.diffusers_repo,
        seed=args.seed,
        n_images=args.n_images,
        guidance_scale=args.guidance_scale,
    )
