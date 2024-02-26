# test image score
import os
import json
from refiners.foundationals.clip.image_encoder import CLIPImageEncoderL, CLIPImageEncoderH
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.text_encoder import TextEncoderWithPoolingL, CLIPTextEncoderL
import PIL
import PIL.Image
import torch
from refiners.fluxion.utils import image_to_tensor, normalize
from tqdm.auto import tqdm
from refiners.foundationals.latent_diffusion.stable_diffusion_1.image_prompt import SD1IPAdapter, get_sd1_image_proj
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder, SD1UNet, StableDiffusion_1
from refiners.foundationals.dinov2 import (
    DINOv2_small,
    DINOv2_small_reg,
    DINOv2_base,
    DINOv2_base_reg,
    DINOv2_large,
    DINOv2_large_reg,
    ViT,
)
from refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock2d
from refiners.fluxion.utils import load_from_safetensors
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder, SD1UNet, StableDiffusion_1
from refiners.foundationals.latent_diffusion.solvers.dpm import DPMSolver
from torch import Tensor, cat, randn
from refiners.fluxion.utils import manual_seed
import gc
import numpy as np
import argparse
def clip_transform(
    image: PIL.Image.Image,
    device: torch.device,
    dtype: torch.dtype,
    size: tuple[int, int] = (224, 224),
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> torch.Tensor:
    # Default mean and std are parameters from https://github.com/openai/CLIP
    return normalize(
        image_to_tensor(image.resize(size), device=device, dtype=dtype),
        mean=[0.48145466, 0.4578275, 0.40821073] if mean is None else mean,
        std=[0.26862954, 0.26130258, 0.27577711] if std is None else std,
    )
def calculate_clip_score(args):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    custom_concept_base_path = "/home/isamu/custom_concept_101"
    db_name =  "benchmark_dataset"
    prompts_and_config = "/home/isamu/custom_concept_101/custom-diffusion/customconcept101"
    single_concept_json = "dataset.json"
    clip_image_encoder_h_path = "/home/isamu/refiners/tests/weights/CLIPImageEncoderH.safetensors"
    text_encoder_path = "/home/isamu/refiners/tests/weights/CLIPLWithProjection.safetensors"
    image_encoder_path = "/home/isamu/refiners/tests/weights/clip_image_l_vision.safetensors"
    cond_resolution = 518

    num_prompts = args.num_prompts
    num_images_per_prompt = args.num_images_per_prompt
    condition_scale = args.condition_scale
    image_embedding_div_factor = args.image_embedding_div_factor
    generation_path = args.generation_path
    checkpoint_path = args.checkpoint_path
    use_pooled_text_embedding=args.use_pooled_text_embedding
    use_timestep_embedding=args.use_timestep_embedding
    with open(os.path.join(prompts_and_config, single_concept_json)) as f:
        data = json.load(f)
    text_encoder_l_with_projection = CLIPTextEncoderL(device=device, dtype=dtype)
    clip_text_encoder = TextEncoderWithPoolingL(target=text_encoder_l_with_projection).inject()
    text_encoder = clip_text_encoder.load_from_safetensors(text_encoder_path).to(device, dtype=dtype)
    clip_image_encoder = CLIPImageEncoderL().load_from_safetensors(image_encoder_path).to(device, dtype=dtype)
    base_data = {}
    for elem in data:
        class_prompt = elem['class_prompt']
        
        prompt_file = os.path.join(prompts_and_config, elem['prompt_filename'])
        benchmark_dataset = os.path.join(custom_concept_base_path, db_name)
        images_folder = elem['instance_data_dir'].replace("./benchmark_dataset", benchmark_dataset)
        validation_image_paths = []
        for image in os.listdir(images_folder):
            image_path = os.path.join(images_folder, image)
            validation_image_paths.append(image_path)
            break
        prompts = []
        with open(prompt_file) as f:
            for i, prompt in enumerate(f.readlines()):
                if i >= num_prompts:
                    break
                prompts.append(prompt.replace("{}", class_prompt))
        cond_image = PIL.Image.open(validation_image_paths[0])
        if cond_image.mode != "RGB":
            cond_image = cond_image.convert("RGB")
        base_data[class_prompt] = {}
        base_data[class_prompt]["image"] = cond_image
        base_data[class_prompt]["prompts"] = prompts
    clip_text = {}
    clip_image = {}
    w = 2.5
    with torch.no_grad():
        for scale in tqdm(scales):
            clip_text[scale] = []
            clip_image[scale] = []
            
            scale_dir = os.path.join(generation_path, str(scale))
            for elem in data:
                class_prompt = elem['class_prompt']
                cond_image = base_data[class_prompt]["image"]
                preprocessed_cond_image = clip_transform(cond_image, device, dtype)
                cond_embed = clip_image_encoder(preprocessed_cond_image)
                cond_embed = cond_embed / cond_embed.norm(p=2, dim=-1, keepdim=True)
                prompts = base_data[class_prompt]["prompts"]
                for idx, prompt in enumerate(prompts):
                    prompt_embeds = text_encoder(prompt)[1]
                    prompt_embeds = prompt_embeds / prompt_embeds.norm(p=2, dim=-1, keepdim=True)
                    for i in range(num_images_per_prompt):
                        generated_image = PIL.Image.open(os.path.join(scale_dir, f"{class_prompt}_{idx}_{i}.png"))
                        preprocessed_generated_image = clip_transform(generated_image, device, dtype)
                        generated_embeds = clip_image_encoder(preprocessed_generated_image)
                        generated_embeds = generated_embeds / generated_embeds.norm(p=2, dim=-1, keepdim=True)
                        clip_text_alignment = torch.sum(prompt_embeds*generated_embeds, dim=1)
                        clip_text_alignment = torch.clip(clip_text_alignment * w, 0)
                        clip_text_alignment = clip_text_alignment.float().cpu().detach().numpy()
                        clip_text[scale].append(clip_text_alignment)
                        clip_image_alignment = torch.sum(cond_embed*generated_embeds, dim=1)
                        clip_image_alignment = torch.clip(clip_image_alignment, 0)
                        clip_image_alignment = clip_image_alignment.float().cpu().detach().numpy()
                        clip_image[scale].append(clip_image_alignment)
    print(f"For {args.checkpoint_path}")
    for scale in scales:
        print(f"At scale: {scale}. Text alignment is {np.mean(clip_text[scale])/2.5} and Image alignment is {np.mean(clip_image[scale])}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converts a CLIPTextEncoder from the library transformers from the HuggingFace Hub to refiners."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/isamu/checkpoints_rescaler_fixed/step400000.safetensors",
        help=(
            "Path to checkpoint"
        ),
    )
    parser.add_argument(
        "--generation_path",
        type=str,
        default="/home/isamu/generation",
        help=(
            "Path for image generation"
        ),
    )
    parser.add_argument(
        "--use_pooled_text_embedding",
        action="store_true",
        help=(
            "Use pooled embed"
        ),
    )
    parser.add_argument(
        "--use_timestep_embedding",
        action="store_true",
        help=(
            "Use timestep embed"
        ),
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=5,
        help=(
            "Number of prompts"
        ),
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help=(
            "Number of images to generate per prompt"
        ),
    )
    parser.add_argument(
        "--condition_scale",
        type=float,
        default=7.5,
        help=(
            "Condition scale"
        ),
    )
    parser.add_argument(
        "--image_embedding_div_factor",
        type=float,
        default=1,
        help=(
            "Division factor by which to divide image embeddings"
        ),
    )
    parser.add_argument(
        "--clip_image_encoder",
        action="store_true",
        help=(
            "use clip image encoder"
        ),
    )
    args = parser.parse_args()
    calculate_clip_score(args)
if __name__ == "__main__":
    main()