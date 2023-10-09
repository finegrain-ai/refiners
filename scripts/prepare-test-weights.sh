#!/bin/bash

# This script downloads source weights from Hugging Face using cURL.
# We want to convert from local directories (not use the network in conversion
# scripts) but we also do not want to clone full repositories to save space.
# This approach is more verbose but it lets us pick and choose.

set -x
cd "$(dirname "$0")/.."

die () { >&2 echo "$@" ; exit 1 ; }

check_hash () {  # (path, hash)
    _path="$1"; shift
    _expected="$1"
    _found="$(b2sum -l 32 "$_path" | cut -d' ' -f1)"
    [ "$_found" = "$_expected" ] || die "invalid hash for $_path ($_found != $_expected)"
}

download_sd_text_encoder () { # (base="runwayml/stable-diffusion-v1-5" subdir="text_encoder")
    _base="$1"; shift
    _subdir="$1"
    mkdir tests/weights/$_base/$_subdir
    pushd tests/weights/$_base/$_subdir
        curl -LO https://huggingface.co/$_base/raw/main/$_subdir/config.json
        curl -LO https://huggingface.co/$_base/resolve/main/$_subdir/model.safetensors
    popd
}

download_sd_tokenizer () { # (base="runwayml/stable-diffusion-v1-5" subdir="tokenizer")
    _base="$1"; shift
    _subdir="$1"
    mkdir tests/weights/$_base/$_subdir
    pushd tests/weights/$_base/$_subdir
        curl -LO https://huggingface.co/$_base/raw/main/$_subdir/merges.txt
        curl -LO https://huggingface.co/$_base/raw/main/$_subdir/special_tokens_map.json
        curl -LO https://huggingface.co/$_base/raw/main/$_subdir/tokenizer_config.json
        curl -LO https://huggingface.co/$_base/raw/main/$_subdir/vocab.json
    popd
}

download_sd_base () { # (base="runwayml/stable-diffusion-v1-5")
    _base="$1"

    # Inpainting source does not have safetensors.
    _ext="safetensors"
    grep -q "inpainting" <<< $_base && _ext="bin"

    mkdir -p tests/weights/$_base
    pushd tests/weights/$_base
        curl -LO https://huggingface.co/$_base/raw/main/model_index.json
        mkdir scheduler unet vae
        pushd scheduler
            curl -LO https://huggingface.co/$_base/raw/main/scheduler/scheduler_config.json
        popd
        pushd unet
            curl -LO https://huggingface.co/$_base/raw/main/unet/config.json
            curl -LO https://huggingface.co/$_base/resolve/main/unet/diffusion_pytorch_model.$_ext
        popd
        pushd vae
            curl -LO https://huggingface.co/$_base/raw/main/vae/config.json
            curl -LO https://huggingface.co/$_base/resolve/main/vae/diffusion_pytorch_model.$_ext
        popd
    popd
    download_sd_text_encoder $_base text_encoder
    download_sd_tokenizer $_base tokenizer
}

download_sd15 () { # (base="runwayml/stable-diffusion-v1-5")
    _base="$1"
    download_sd_base $_base
    pushd tests/weights/$_base
        mkdir feature_extractor safety_checker
        pushd feature_extractor
            curl -LO https://huggingface.co/$_base/raw/main/feature_extractor/preprocessor_config.json
        popd
        pushd safety_checker
            curl -LO https://huggingface.co/$_base/raw/main/safety_checker/config.json
            curl -LO https://huggingface.co/$_base/resolve/main/safety_checker/model.safetensors
        popd
    popd
}

download_sdxl () { # (base="stabilityai/stable-diffusion-xl-base-1.0")
    _base="$1"
    download_sd_base $_base
    download_sd_text_encoder $_base text_encoder_2
    download_sd_tokenizer $_base tokenizer_2
}

download_vae_ft_mse () {
    mkdir -p tests/weights/stabilityai/sd-vae-ft-mse
    pushd tests/weights/stabilityai/sd-vae-ft-mse
        curl -LO https://huggingface.co/stabilityai/sd-vae-ft-mse/raw/main/config.json
        curl -LO https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors
    popd
}

download_lora () {
    mkdir -p tests/weights/pcuenq/pokemon-lora
    pushd tests/weights/pcuenq/pokemon-lora
        curl -LO https://huggingface.co/pcuenq/pokemon-lora/resolve/main/pytorch_lora_weights.bin
    popd
}

download_preprocessors () {
    mkdir -p tests/weights/carolineec/informativedrawings
    pushd tests/weights/carolineec/informativedrawings
        curl -LO https://huggingface.co/spaces/carolineec/informativedrawings/resolve/main/model2.pth
    popd
}

download_controlnet () {
    mkdir -p tests/weights/lllyasviel
    pushd tests/weights/lllyasviel
        mkdir control_v11p_sd15_canny
        pushd control_v11p_sd15_canny
            curl -LO https://huggingface.co/lllyasviel/control_v11p_sd15_canny/raw/main/config.json
            curl -LO https://huggingface.co/lllyasviel/control_v11p_sd15_canny/resolve/main/diffusion_pytorch_model.safetensors
        popd
        mkdir control_v11f1p_sd15_depth
        pushd control_v11f1p_sd15_depth
            curl -LO https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth/raw/main/config.json
            curl -LO https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth/resolve/main/diffusion_pytorch_model.safetensors
        popd
        mkdir control_v11p_sd15_normalbae
        pushd control_v11p_sd15_normalbae
            curl -LO https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae/raw/main/config.json
            curl -LO https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae/resolve/main/diffusion_pytorch_model.safetensors
        popd
        mkdir control_v11p_sd15_lineart
        pushd control_v11p_sd15_lineart
            curl -LO https://huggingface.co/lllyasviel/control_v11p_sd15_lineart/raw/main/config.json
            curl -LO https://huggingface.co/lllyasviel/control_v11p_sd15_lineart/resolve/main/diffusion_pytorch_model.safetensors
        popd
    popd

    mkdir -p tests/weights/mfidabel/controlnet-segment-anything
    pushd tests/weights/mfidabel/controlnet-segment-anything
        curl -LO https://huggingface.co/mfidabel/controlnet-segment-anything/raw/main/config.json
        curl -LO https://huggingface.co/mfidabel/controlnet-segment-anything/resolve/main/diffusion_pytorch_model.bin
    popd
}

download_unclip () {
    mkdir -p tests/weights/stabilityai/stable-diffusion-2-1-unclip
    pushd tests/weights/stabilityai/stable-diffusion-2-1-unclip
        curl -LO https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip/raw/main/model_index.json
        mkdir image_encoder
        pushd image_encoder
            curl -LO https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip/raw/main/image_encoder/config.json
            curl -LO https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip/resolve/main/image_encoder/model.safetensors
        popd
    popd
}

download_ip_adapter () {
    mkdir -p tests/weights/h94/IP-Adapter
    pushd tests/weights/h94/IP-Adapter
        mkdir -p models
        pushd models
            curl -LO https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin
            curl -LO https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.bin
        popd
        mkdir -p sdxl_models
        pushd sdxl_models
            curl -LO https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.bin
            curl -LO https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin
        popd
    popd
}

download_t2i_adapter () {
    mkdir -p tests/weights/TencentARC/t2iadapter_depth_sd15v2
    pushd tests/weights/TencentARC/t2iadapter_depth_sd15v2
        curl -LO https://huggingface.co/TencentARC/t2iadapter_depth_sd15v2/raw/main/config.json
        curl -LO https://huggingface.co/TencentARC/t2iadapter_depth_sd15v2/resolve/main/diffusion_pytorch_model.bin
    popd
}

download_sam () {
    mkdir -p tests/weights
    pushd tests/weights
        curl -LO https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    popd
    check_hash "tests/weights/sam_vit_h_4b8939.pth" 06785e66
}

convert_sd15 () {
    python scripts/conversion/convert_transformers_clip_text_model.py \
        --from "tests/weights/runwayml/stable-diffusion-v1-5" \
        --to "tests/weights/CLIPTextEncoderL.safetensors" \
        --half
    check_hash "tests/weights/CLIPTextEncoderL.safetensors" bef71657

    python scripts/conversion/convert_diffusers_autoencoder_kl.py \
        --from "tests/weights/runwayml/stable-diffusion-v1-5" \
        --to "tests/weights/lda.safetensors"
    check_hash "tests/weights/lda.safetensors" 28f38b35

    python scripts/conversion/convert_diffusers_unet.py \
        --from "tests/weights/runwayml/stable-diffusion-v1-5" \
        --to "tests/weights/unet.safetensors" \
        --half
    check_hash "tests/weights/unet.safetensors" d283a9a5

    mkdir tests/weights/inpainting

    python scripts/conversion/convert_diffusers_unet.py \
        --from "tests/weights/runwayml/stable-diffusion-inpainting" \
        --to "tests/weights/inpainting/unet.safetensors" \
        --half
    check_hash "tests/weights/inpainting/unet.safetensors" 78069e20
}

convert_sdxl () {
    python scripts/conversion/convert_transformers_clip_text_model.py \
        --from "tests/weights/stabilityai/stable-diffusion-xl-base-1.0" \
        --to "tests/weights/DoubleCLIPTextEncoder.safetensors" \
        --subfolder2 text_encoder_2 \
        --half
    check_hash "tests/weights/DoubleCLIPTextEncoder.safetensors" a68fd375

    python scripts/conversion/convert_diffusers_autoencoder_kl.py \
        --from "tests/weights/stabilityai/stable-diffusion-xl-base-1.0" \
        --to "tests/weights/sdxl-lda.safetensors" \
        --half
    check_hash "tests/weights/sdxl-lda.safetensors" b00aaf87

    python scripts/conversion/convert_diffusers_unet.py \
        --from "tests/weights/stabilityai/stable-diffusion-xl-base-1.0" \
        --to "tests/weights/sdxl-unet.safetensors" \
        --half
    check_hash "tests/weights/sdxl-unet.safetensors" 861b57fd
}

convert_vae_ft_mse () {
    python scripts/conversion/convert_diffusers_autoencoder_kl.py \
        --from "tests/weights/stabilityai/sd-vae-ft-mse" \
        --to "tests/weights/lda_ft_mse.safetensors" \
        --half
    check_hash "tests/weights/lda_ft_mse.safetensors" 6cfb7776
}

convert_lora () {
    mkdir tests/weights/loras

    python scripts/conversion/convert_diffusers_lora.py \
        --from "tests/weights/pcuenq/pokemon-lora/pytorch_lora_weights.bin" \
        --base-model "tests/weights/runwayml/stable-diffusion-v1-5" \
        --to "tests/weights/loras/pcuenq_pokemon_lora.safetensors"
    check_hash "tests/weights/loras/pcuenq_pokemon_lora.safetensors" a9d7e08e
}

convert_preprocessors () {
    curl -L https://raw.githubusercontent.com/carolineec/informative-drawings/main/model.py \
        -o src/model.py
    python scripts/conversion/convert_informative_drawings.py \
        --from "tests/weights/carolineec/informativedrawings/model2.pth" \
        --to "tests/weights/informative-drawings.safetensors"
    rm -f src/model.py
    check_hash "tests/weights/informative-drawings.safetensors" 0294ac8a
}

convert_controlnet () {
    mkdir tests/weights/controlnet

    python scripts/conversion/convert_diffusers_controlnet.py \
        --from "tests/weights/lllyasviel/control_v11p_sd15_canny" \
        --to "tests/weights/controlnet/lllyasviel_control_v11p_sd15_canny.safetensors"
    check_hash "tests/weights/controlnet/lllyasviel_control_v11p_sd15_canny.safetensors" be9ffe47

    python scripts/conversion/convert_diffusers_controlnet.py \
        --from "tests/weights/lllyasviel/control_v11f1p_sd15_depth" \
        --to "tests/weights/controlnet/lllyasviel_control_v11f1p_sd15_depth.safetensors"
    check_hash "tests/weights/controlnet/lllyasviel_control_v11f1p_sd15_depth.safetensors" bbeaa1ba

    python scripts/conversion/convert_diffusers_controlnet.py \
        --from "tests/weights/lllyasviel/control_v11p_sd15_normalbae" \
        --to "tests/weights/controlnet/lllyasviel_control_v11p_sd15_normalbae.safetensors"
    check_hash "tests/weights/controlnet/lllyasviel_control_v11p_sd15_normalbae.safetensors" 24520c5b

    python scripts/conversion/convert_diffusers_controlnet.py \
        --from "tests/weights/lllyasviel/control_v11p_sd15_lineart" \
        --to "tests/weights/controlnet/lllyasviel_control_v11p_sd15_lineart.safetensors"
    check_hash "tests/weights/controlnet/lllyasviel_control_v11p_sd15_lineart.safetensors" 5bc4de82

    python scripts/conversion/convert_diffusers_controlnet.py \
        --from "tests/weights/mfidabel/controlnet-segment-anything" \
        --to "tests/weights/controlnet/mfidabel_controlnet-segment-anything.safetensors"
    check_hash "tests/weights/controlnet/mfidabel_controlnet-segment-anything.safetensors" ba7059fc
}

convert_unclip () {
    python scripts/conversion/convert_transformers_clip_image_model.py \
        --from "tests/weights/stabilityai/stable-diffusion-2-1-unclip" \
        --to "tests/weights/CLIPImageEncoderH.safetensors" \
        --half
    check_hash "tests/weights/CLIPImageEncoderH.safetensors" 654842e4
}

convert_ip_adapter () {
    python scripts/conversion/convert_diffusers_ip_adapter.py \
        --from "tests/weights/h94/IP-Adapter/models/ip-adapter_sd15.bin" \
        --to "tests/weights/ip-adapter_sd15.safetensors"
    check_hash "tests/weights/ip-adapter_sd15.safetensors" 9579b465

    python scripts/conversion/convert_diffusers_ip_adapter.py \
        --from "tests/weights/h94/IP-Adapter/sdxl_models/ip-adapter_sdxl_vit-h.bin" \
        --to "tests/weights/ip-adapter_sdxl_vit-h.safetensors" \
        --half
    check_hash "tests/weights/ip-adapter_sdxl_vit-h.safetensors" 739504c6

    python scripts/conversion/convert_diffusers_ip_adapter.py \
        --from "tests/weights/h94/IP-Adapter/models/ip-adapter-plus_sd15.bin" \
        --to "tests/weights/ip-adapter-plus_sd15.safetensors" \
        --half
    check_hash "tests/weights/ip-adapter-plus_sd15.safetensors" 9cea790f

    python scripts/conversion/convert_diffusers_ip_adapter.py \
        --from "tests/weights/h94/IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin" \
        --to "tests/weights/ip-adapter-plus_sdxl_vit-h.safetensors" \
        --half
    check_hash "tests/weights/ip-adapter-plus_sdxl_vit-h.safetensors" a090ab44
}

convert_t2i_adapter () {
    mkdir tests/weights/T2I-Adapter
    python scripts/conversion/convert_diffusers_t2i_adapter.py \
        --from "tests/weights/TencentARC/t2iadapter_depth_sd15v2" \
        --to "tests/weights/T2I-Adapter/t2iadapter_depth_sd15v2.safetensors" \
        --half
    check_hash "tests/weights/T2I-Adapter/t2iadapter_depth_sd15v2.safetensors" 809a355f
}

convert_sam () {
    python scripts/conversion/convert_segment_anything.py \
        --from "tests/weights/sam_vit_h_4b8939.pth" \
        --to "tests/weights/segment-anything-h.safetensors"
    check_hash "tests/weights/segment-anything-h.safetensors" e11e1ec5
}

download_all () {
    download_sd15 runwayml/stable-diffusion-v1-5
    download_sd15 runwayml/stable-diffusion-inpainting
    download_sdxl stabilityai/stable-diffusion-xl-base-1.0
    download_vae_ft_mse
    download_lora
    download_preprocessors
    download_controlnet
    download_unclip
    download_ip_adapter
    download_t2i_adapter
    download_sam
}

convert_all () {
    convert_sd15
    convert_sdxl
    convert_vae_ft_mse
    convert_lora
    convert_preprocessors
    convert_controlnet
    convert_unclip
    convert_ip_adapter
    convert_t2i_adapter
    convert_sam
}

main () {
    git lfs install || die "could not install git lfs"
    rm -rf tests/weights
    download_all
    convert_all
}

main
