"""
Download and convert weights for testing

To see what weights will be downloaded and converted, run:
DRY_RUN=1 python scripts/prepare_test_weights.py
"""
import hashlib
import os
import subprocess
import sys
from urllib.parse import urlparse

import requests
from tqdm import tqdm

# Set the base directory to the parent directory of the script
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
test_weights_dir = os.path.join(project_dir, "tests", "weights")

previous_line = "\033[F"

download_count = 0
bytes_count = 0


def die(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


def rel(path: str) -> str:
    return os.path.relpath(path, project_dir)


def calc_hash(filepath: str) -> str:
    with open(filepath, "rb") as f:
        data = f.read()
        found = hashlib.blake2b(data, digest_size=int(32 / 8)).hexdigest()
    return found


def check_hash(path: str, expected: str) -> str:
    found = calc_hash(path)
    if found != expected:
        die(f"❌ Invalid hash for {path} ({found} != {expected})")
    return found


def download_file(
    url: str,
    dest_folder: str,
    dry_run: bool | None = None,
    skip_existing: bool = True,
    expected_hash: str | None = None,
):
    """
    Downloads a file

    Features:
      - shows a progress bar
      - skips existing files
      - uses a temporary file to prevent partial downloads
      - can do a dry run to check the url is valid
      - displays the downloaded file hash

    """
    global download_count, bytes_count
    filename = os.path.basename(urlparse(url).path)
    dest_filename = os.path.join(dest_folder, filename)
    temp_filename = dest_filename + ".part"
    dry_run = bool(os.environ.get("DRY_RUN") == "1") if dry_run is None else dry_run

    is_downloaded = os.path.exists(dest_filename)
    if is_downloaded and skip_existing:
        skip_icon = "✖️ "
    else:
        skip_icon = "🔽"

    if dry_run:
        response = requests.head(url, allow_redirects=True)
        readable_size = ""

        if response.status_code == 200:
            content_length = response.headers.get("content-length")

            if content_length:
                size_in_bytes = int(content_length)
                readable_size = human_readable_size(size_in_bytes)
                download_count += 1
                bytes_count += size_in_bytes
            print(f"✅{skip_icon} {response.status_code} READY {readable_size:<8} {url}")

        else:
            print(f"❌{skip_icon} {response.status_code} ERROR {readable_size:<8} {url}")
        return

    if skip_existing and os.path.exists(dest_filename):
        print(f"{skip_icon}️ Skipping previously downloaded {url}")
        return

    os.makedirs(dest_folder, exist_ok=True)

    print(f"🔽 Downloading {url} => '{rel(dest_filename)}'", end="\n")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(response.content[:1000])
        die(f"Failed to download {url}. Status code: {response.status_code}")
    total = int(response.headers.get("content-length", 0))
    bar = tqdm(
        desc=filename,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        leave=False,
    )
    with open(temp_filename, "wb") as f, bar:
        for data in response.iter_content(chunk_size=1024 * 1000):
            size = f.write(data)
            bar.update(size)

    os.rename(temp_filename, dest_filename)
    calculated_hash = calc_hash(dest_filename)

    print(f"{previous_line}✅ Downloaded {calculated_hash} {url} => '{rel(dest_filename)}' ")
    if expected_hash is not None:
        check_hash(dest_filename, expected_hash)


def download_files(urls: list[str], dest_folder: str):
    for url in urls:
        download_file(url, dest_folder)


def human_readable_size(size: int | float, decimal_places: int = 2) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f}{unit}"  # type: ignore


def download_sd_text_encoder(hf_repo_id: str = "runwayml/stable-diffusion-v1-5", subdir: str = "text_encoder"):
    encoder_filename = "model.safetensors" if "inpainting" not in hf_repo_id else "model.fp16.safetensors"
    base_url = f"https://huggingface.co/{hf_repo_id}"
    download_files(
        urls=[
            f"{base_url}/raw/main/{subdir}/config.json",
            f"{base_url}/resolve/main/{subdir}/{encoder_filename}",
        ],
        dest_folder=os.path.join(test_weights_dir, hf_repo_id, subdir),
    )


def download_sd_tokenizer(hf_repo_id: str = "runwayml/stable-diffusion-v1-5", subdir: str = "tokenizer"):
    download_files(
        urls=[
            f"https://huggingface.co/{hf_repo_id}/raw/main/{subdir}/merges.txt",
            f"https://huggingface.co/{hf_repo_id}/raw/main/{subdir}/special_tokens_map.json",
            f"https://huggingface.co/{hf_repo_id}/raw/main/{subdir}/tokenizer_config.json",
            f"https://huggingface.co/{hf_repo_id}/raw/main/{subdir}/vocab.json",
        ],
        dest_folder=os.path.join(test_weights_dir, hf_repo_id, subdir),
    )


def download_sd_base(hf_repo_id: str = "runwayml/stable-diffusion-v1-5"):
    is_inpainting = "inpainting" in hf_repo_id
    ext = "safetensors" if not is_inpainting else "bin"
    base_folder = os.path.join(test_weights_dir, hf_repo_id)
    download_file(f"https://huggingface.co/{hf_repo_id}/raw/main/model_index.json", base_folder)
    download_file(
        f"https://huggingface.co/{hf_repo_id}/raw/main/scheduler/scheduler_config.json",
        os.path.join(base_folder, "scheduler"),
    )

    for subdir in ["unet", "vae"]:
        subdir_folder = os.path.join(base_folder, subdir)
        download_file(f"https://huggingface.co/{hf_repo_id}/raw/main/{subdir}/config.json", subdir_folder)
        download_file(
            f"https://huggingface.co/{hf_repo_id}/resolve/main/{subdir}/diffusion_pytorch_model.{ext}", subdir_folder
        )
    # we only need the unet for the inpainting model
    if not is_inpainting:
        download_sd_text_encoder(hf_repo_id, "text_encoder")
    download_sd_tokenizer(hf_repo_id, "tokenizer")


def download_sd15(hf_repo_id: str = "runwayml/stable-diffusion-v1-5"):
    download_sd_base(hf_repo_id)
    base_folder = os.path.join(test_weights_dir, hf_repo_id)

    subdir = "feature_extractor"
    download_file(
        f"https://huggingface.co/{hf_repo_id}/raw/main/{subdir}/preprocessor_config.json",
        os.path.join(base_folder, subdir),
    )

    if "inpainting" not in hf_repo_id:
        subdir = "safety_checker"
        subdir_folder = os.path.join(base_folder, subdir)
        download_file(f"https://huggingface.co/{hf_repo_id}/raw/main/{subdir}/config.json", subdir_folder)
        download_file(f"https://huggingface.co/{hf_repo_id}/resolve/main/{subdir}/model.safetensors", subdir_folder)


def download_sdxl(hf_repo_id: str = "stabilityai/stable-diffusion-xl-base-1.0"):
    download_sd_base(hf_repo_id)
    download_sd_text_encoder(hf_repo_id, "text_encoder_2")
    download_sd_tokenizer(hf_repo_id, "tokenizer_2")


def download_vae_fp16_fix():
    download_files(
        urls=[
            "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/raw/main/config.json",
            "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/diffusion_pytorch_model.safetensors",
        ],
        dest_folder=os.path.join(test_weights_dir, "madebyollin", "sdxl-vae-fp16-fix"),
    )


def download_vae_ft_mse():
    download_files(
        urls=[
            "https://huggingface.co/stabilityai/sd-vae-ft-mse/raw/main/config.json",
            "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors",
        ],
        dest_folder=os.path.join(test_weights_dir, "stabilityai", "sd-vae-ft-mse"),
    )


def download_loras():
    dest_folder = os.path.join(test_weights_dir, "loras", "pokemon-lora")
    download_file("https://huggingface.co/pcuenq/pokemon-lora/resolve/main/pytorch_lora_weights.bin", dest_folder)

    dest_folder = os.path.join(test_weights_dir, "loras", "dpo-lora")
    download_file(
        "https://huggingface.co/radames/sdxl-DPO-LoRA/resolve/main/pytorch_lora_weights.safetensors", dest_folder
    )

    dest_folder = os.path.join(test_weights_dir, "loras", "sliders")
    download_file("https://sliders.baulab.info/weights/xl_sliders/age.pt", dest_folder)
    download_file("https://sliders.baulab.info/weights/xl_sliders/cartoon_style.pt", dest_folder)
    download_file("https://sliders.baulab.info/weights/xl_sliders/eyesize.pt", dest_folder)


def download_preprocessors():
    dest_folder = os.path.join(test_weights_dir, "carolineec", "informativedrawings")
    download_file("https://huggingface.co/spaces/carolineec/informativedrawings/resolve/main/model2.pth", dest_folder)


def download_controlnet():
    base_folder = os.path.join(test_weights_dir, "lllyasviel")
    controlnets = [
        "control_v11p_sd15_canny",
        "control_v11f1p_sd15_depth",
        "control_v11p_sd15_normalbae",
        "control_v11p_sd15_lineart",
    ]
    for net in controlnets:
        net_folder = os.path.join(base_folder, net)
        urls = [
            f"https://huggingface.co/lllyasviel/{net}/raw/main/config.json",
            f"https://huggingface.co/lllyasviel/{net}/resolve/main/diffusion_pytorch_model.safetensors",
        ]
        download_files(urls, net_folder)

    mfidabel_folder = os.path.join(test_weights_dir, "mfidabel", "controlnet-segment-anything")
    urls = [
        "https://huggingface.co/mfidabel/controlnet-segment-anything/raw/main/config.json",
        "https://huggingface.co/mfidabel/controlnet-segment-anything/resolve/main/diffusion_pytorch_model.bin",
    ]
    download_files(urls, mfidabel_folder)


def download_unclip():
    base_folder = os.path.join(test_weights_dir, "stabilityai", "stable-diffusion-2-1-unclip")
    download_file(
        "https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip/raw/main/model_index.json", base_folder
    )
    image_encoder_folder = os.path.join(base_folder, "image_encoder")
    urls = [
        "https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip/raw/main/image_encoder/config.json",
        "https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip/resolve/main/image_encoder/model.safetensors",
    ]
    download_files(urls, image_encoder_folder)


def download_ip_adapter():
    base_folder = os.path.join(test_weights_dir, "h94", "IP-Adapter")
    models_folder = os.path.join(base_folder, "models")
    urls = [
        "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin",
        "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.bin",
    ]
    download_files(urls, models_folder)

    sdxl_models_folder = os.path.join(base_folder, "sdxl_models")
    urls = [
        "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.bin",
        "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin",
    ]
    download_files(urls, sdxl_models_folder)


def download_t2i_adapter():
    base_folder = os.path.join(test_weights_dir, "TencentARC", "t2iadapter_depth_sd15v2")
    urls = [
        "https://huggingface.co/TencentARC/t2iadapter_depth_sd15v2/raw/main/config.json",
        "https://huggingface.co/TencentARC/t2iadapter_depth_sd15v2/resolve/main/diffusion_pytorch_model.bin",
    ]
    download_files(urls, base_folder)

    canny_sdxl_folder = os.path.join(test_weights_dir, "TencentARC", "t2i-adapter-canny-sdxl-1.0")
    urls = [
        "https://huggingface.co/TencentARC/t2i-adapter-canny-sdxl-1.0/raw/main/config.json",
        "https://huggingface.co/TencentARC/t2i-adapter-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors",
    ]
    download_files(urls, canny_sdxl_folder)


def download_sam():
    weights_folder = os.path.join(test_weights_dir)
    download_file(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", weights_folder, expected_hash="06785e66"
    )


def download_dinov2():
    # For conversion
    weights_folder = os.path.join(test_weights_dir)
    urls = [
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
    ]
    download_files(urls, weights_folder)

    # For testing (note: versions with registers are not available yet on HuggingFace)
    for repo in ["dinov2-small", "dinov2-base", "dinov2-large"]:
        base_folder = os.path.join(test_weights_dir, "facebook", repo)
        urls = [
            f"https://huggingface.co/facebook/{repo}/raw/main/config.json",
            f"https://huggingface.co/facebook/{repo}/raw/main/preprocessor_config.json",
            f"https://huggingface.co/facebook/{repo}/resolve/main/pytorch_model.bin",
        ]
        download_files(urls, base_folder)


def printg(msg: str):
    """print in green color"""
    print("\033[92m" + msg + "\033[0m")


def run_conversion_script(
    script_filename: str,
    from_weights: str,
    to_weights: str,
    half: bool = False,
    expected_hash: str | None = None,
    additional_args: list[str] | None = None,
    skip_existing: bool = True,
):
    if skip_existing and expected_hash and os.path.exists(to_weights):
        found_hash = check_hash(to_weights, expected_hash)
        if expected_hash == found_hash:
            printg(f"✖️  Skipping converted from {from_weights} to {to_weights} (hash {found_hash} confirmed)   ")
            return

    msg = f"Converting {from_weights} to {to_weights}"
    printg(msg)

    args = ["python", f"scripts/conversion/{script_filename}", "--from", from_weights, "--to", to_weights]
    if half:
        args.append("--half")
    if additional_args:
        args.extend(additional_args)

    subprocess.run(args, check=True)
    if expected_hash is not None:
        found_hash = check_hash(to_weights, expected_hash)
        printg(f"✅  Converted from {from_weights} to {to_weights} (hash {found_hash} confirmed)   ")
    else:
        printg(f"✅⚠️  Converted from {from_weights} to {to_weights} (no hash check performed)")


def convert_sd15():
    run_conversion_script(
        script_filename="convert_transformers_clip_text_model.py",
        from_weights="tests/weights/runwayml/stable-diffusion-v1-5",
        to_weights="tests/weights/CLIPTextEncoderL.safetensors",
        half=True,
        expected_hash="6c9cbc59",
    )
    run_conversion_script(
        "convert_diffusers_autoencoder_kl.py",
        "tests/weights/runwayml/stable-diffusion-v1-5",
        "tests/weights/lda.safetensors",
        expected_hash="329e369c",
    )
    run_conversion_script(
        "convert_diffusers_unet.py",
        "tests/weights/runwayml/stable-diffusion-v1-5",
        "tests/weights/unet.safetensors",
        half=True,
        expected_hash="f81ac65a",
    )
    os.makedirs("tests/weights/inpainting", exist_ok=True)
    run_conversion_script(
        "convert_diffusers_unet.py",
        "tests/weights/runwayml/stable-diffusion-inpainting",
        "tests/weights/inpainting/unet.safetensors",
        half=True,
        expected_hash="c07a8c61",
    )


def convert_sdxl():
    run_conversion_script(
        "convert_transformers_clip_text_model.py",
        "tests/weights/stabilityai/stable-diffusion-xl-base-1.0",
        "tests/weights/DoubleCLIPTextEncoder.safetensors",
        half=True,
        expected_hash="7f99c30b",
        additional_args=["--subfolder2", "text_encoder_2"],
    )
    run_conversion_script(
        "convert_diffusers_autoencoder_kl.py",
        "tests/weights/stabilityai/stable-diffusion-xl-base-1.0",
        "tests/weights/sdxl-lda.safetensors",
        half=True,
        expected_hash="7464e9dc",
    )
    run_conversion_script(
        "convert_diffusers_unet.py",
        "tests/weights/stabilityai/stable-diffusion-xl-base-1.0",
        "tests/weights/sdxl-unet.safetensors",
        half=True,
        expected_hash="2e5c4911",
    )


def convert_vae_ft_mse():
    run_conversion_script(
        "convert_diffusers_autoencoder_kl.py",
        "tests/weights/stabilityai/sd-vae-ft-mse",
        "tests/weights/lda_ft_mse.safetensors",
        half=True,
        expected_hash="4d0bae7e",
    )


def convert_vae_fp16_fix():
    run_conversion_script(
        "convert_diffusers_autoencoder_kl.py",
        "tests/weights/madebyollin/sdxl-vae-fp16-fix",
        "tests/weights/sdxl-lda-fp16-fix.safetensors",
        additional_args=["--subfolder", "''"],
        half=True,
        expected_hash="98c7e998",
    )


def convert_preprocessors():
    subprocess.run(
        [
            "curl",
            "-L",
            "https://raw.githubusercontent.com/carolineec/informative-drawings/main/model.py",
            "-o",
            "src/model.py",
        ],
        check=True,
    )
    run_conversion_script(
        "convert_informative_drawings.py",
        "tests/weights/carolineec/informativedrawings/model2.pth",
        "tests/weights/informative-drawings.safetensors",
        expected_hash="93dca207",
    )
    os.remove("src/model.py")


def convert_controlnet():
    os.makedirs("tests/weights/controlnet", exist_ok=True)
    run_conversion_script(
        "convert_diffusers_controlnet.py",
        "tests/weights/lllyasviel/control_v11p_sd15_canny",
        "tests/weights/controlnet/lllyasviel_control_v11p_sd15_canny.safetensors",
        expected_hash="9a1a48cf",
    )
    run_conversion_script(
        "convert_diffusers_controlnet.py",
        "tests/weights/lllyasviel/control_v11f1p_sd15_depth",
        "tests/weights/controlnet/lllyasviel_control_v11f1p_sd15_depth.safetensors",
        expected_hash="bbe7e5a6",
    )
    run_conversion_script(
        "convert_diffusers_controlnet.py",
        "tests/weights/lllyasviel/control_v11p_sd15_normalbae",
        "tests/weights/controlnet/lllyasviel_control_v11p_sd15_normalbae.safetensors",
        expected_hash="9fa88ed5",
    )
    run_conversion_script(
        "convert_diffusers_controlnet.py",
        "tests/weights/lllyasviel/control_v11p_sd15_lineart",
        "tests/weights/controlnet/lllyasviel_control_v11p_sd15_lineart.safetensors",
        expected_hash="c29e8c03",
    )
    run_conversion_script(
        "convert_diffusers_controlnet.py",
        "tests/weights/mfidabel/controlnet-segment-anything",
        "tests/weights/controlnet/mfidabel_controlnet-segment-anything.safetensors",
        expected_hash="d536eebb",
    )


def convert_unclip():
    run_conversion_script(
        "convert_transformers_clip_image_model.py",
        "tests/weights/stabilityai/stable-diffusion-2-1-unclip",
        "tests/weights/CLIPImageEncoderH.safetensors",
        half=True,
        expected_hash="4ddb44d2",
    )


def convert_ip_adapter():
    run_conversion_script(
        "convert_diffusers_ip_adapter.py",
        "tests/weights/h94/IP-Adapter/models/ip-adapter_sd15.bin",
        "tests/weights/ip-adapter_sd15.safetensors",
        expected_hash="3fb0472e",
    )
    run_conversion_script(
        "convert_diffusers_ip_adapter.py",
        "tests/weights/h94/IP-Adapter/sdxl_models/ip-adapter_sdxl_vit-h.bin",
        "tests/weights/ip-adapter_sdxl_vit-h.safetensors",
        half=True,
        expected_hash="860518fe",
    )
    run_conversion_script(
        "convert_diffusers_ip_adapter.py",
        "tests/weights/h94/IP-Adapter/models/ip-adapter-plus_sd15.bin",
        "tests/weights/ip-adapter-plus_sd15.safetensors",
        half=True,
        expected_hash="aba8503b",
    )
    run_conversion_script(
        "convert_diffusers_ip_adapter.py",
        "tests/weights/h94/IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin",
        "tests/weights/ip-adapter-plus_sdxl_vit-h.safetensors",
        half=True,
        expected_hash="545d5ce7",
    )


def convert_t2i_adapter():
    os.makedirs("tests/weights/T2I-Adapter", exist_ok=True)
    run_conversion_script(
        "convert_diffusers_t2i_adapter.py",
        "tests/weights/TencentARC/t2iadapter_depth_sd15v2",
        "tests/weights/T2I-Adapter/t2iadapter_depth_sd15v2.safetensors",
        half=True,
        expected_hash="bb2b3115",
    )
    run_conversion_script(
        "convert_diffusers_t2i_adapter.py",
        "tests/weights/TencentARC/t2i-adapter-canny-sdxl-1.0",
        "tests/weights/T2I-Adapter/t2i-adapter-canny-sdxl-1.0.safetensors",
        half=True,
        expected_hash="f07249a6",
    )


def convert_sam():
    run_conversion_script(
        "convert_segment_anything.py",
        "tests/weights/sam_vit_h_4b8939.pth",
        "tests/weights/segment-anything-h.safetensors",
        expected_hash="b62ad5ed",
    )


def convert_dinov2():
    run_conversion_script(
        "convert_dinov2.py",
        "tests/weights/dinov2_vits14_pretrain.pth",
        "tests/weights/dinov2_vits14_pretrain.safetensors",
        expected_hash="b7f9b294",
    )
    run_conversion_script(
        "convert_dinov2.py",
        "tests/weights/dinov2_vitb14_pretrain.pth",
        "tests/weights/dinov2_vitb14_pretrain.safetensors",
        expected_hash="d72c767b",
    )
    run_conversion_script(
        "convert_dinov2.py",
        "tests/weights/dinov2_vitl14_pretrain.pth",
        "tests/weights/dinov2_vitl14_pretrain.safetensors",
        expected_hash="71eb98d1",
    )
    run_conversion_script(
        "convert_dinov2.py",
        "tests/weights/dinov2_vits14_reg4_pretrain.pth",
        "tests/weights/dinov2_vits14_reg4_pretrain.safetensors",
        expected_hash="89118b46",
    )
    run_conversion_script(
        "convert_dinov2.py",
        "tests/weights/dinov2_vitb14_reg4_pretrain.pth",
        "tests/weights/dinov2_vitb14_reg4_pretrain.safetensors",
        expected_hash="b0296f77",
    )
    run_conversion_script(
        "convert_dinov2.py",
        "tests/weights/dinov2_vitl14_reg4_pretrain.pth",
        "tests/weights/dinov2_vitl14_reg4_pretrain.safetensors",
        expected_hash="b3d877dc",
    )


def download_all():
    print(f"\nAll weights will be downloaded to {test_weights_dir}\n")
    download_sd15("runwayml/stable-diffusion-v1-5")
    download_sd15("runwayml/stable-diffusion-inpainting")
    download_sdxl("stabilityai/stable-diffusion-xl-base-1.0")
    download_vae_ft_mse()
    download_vae_fp16_fix()
    download_loras()
    download_preprocessors()
    download_controlnet()
    download_unclip()
    download_ip_adapter()
    download_t2i_adapter()
    download_sam()
    download_dinov2()


def convert_all():
    convert_sd15()
    convert_sdxl()
    convert_vae_ft_mse()
    convert_vae_fp16_fix()
    # Note: no convert loras: this is done at runtime by `SDLoraManager`
    convert_preprocessors()
    convert_controlnet()
    convert_unclip()
    convert_ip_adapter()
    convert_t2i_adapter()
    convert_sam()
    convert_dinov2()


def main():
    try:
        download_all()
        print(f"{download_count} files ({human_readable_size(bytes_count)})\n")
        if not bool(os.environ.get("DRY_RUN") == "1"):
            printg("Converting weights to refiners format\n")
            convert_all()
    except KeyboardInterrupt:
        print("Stopped")


if __name__ == "__main__":
    main()
