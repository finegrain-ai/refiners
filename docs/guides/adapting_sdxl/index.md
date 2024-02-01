---
icon: material/castle
---

# Adapting Stable Diffusion XL

Stable Diffusion XL (SDXL) is a very popular text-to-image open source foundation model. This guide will show you how to boost its capabilities with Refiners, using iconic adapters the framework supports out-of-the-box, i.e. without the need for tedious prompt engineering. We'll follow a step by step approach, progressively increasing the number of adapters involved to showcase how simple adapter composition is using Refiners. Our use case will be the generation of an image with "a futuristic castle surrounded by a forest, mountains in the background".

## Prerequisites

Make sure Refiners is installed in your local environment - see [Getting started](/getting-started/recommended/) - and you have access to a decent GPU. 

!!! warning
    As the examples in this guide's code snippets use CUDA, a minimum of 24GB VRAM is needed. 

Before diving into the adapters themselves, let's establish a baseline by simply prompting SDXL with Refiners.

!!! note "Reminder"
    A StableDiffusion model is composed of three modules: 
    
    - An Autoencoder, responsible for embedding images into a latent space;
    - A UNet, responsible for the diffusion process;
    - A prompt encoder, such as CLIP, responsible for encoding the user prompt which will guide the diffusion process.

As Refiners comes with a new model representation - see [Chain](/concepts/chain/) - , you need to download and convert the weights of each module by calling our conversion scripts directly from your terminal (make sure you're in your local `refiners` directory, with your local environment active):

```bash
python scripts/conversion/convert_transformers_clip_text_model.py --from "stabilityai/stable-diffusion-xl-base-1.0" --subfolder2 text_encoder_2 --to DoubleCLIPTextEncoder.safetensors --half
python scripts/conversion/convert_diffusers_unet.py --from "stabilityai/stable-diffusion-xl-base-1.0" --to sdxl-unet.safetensors --half
python scripts/conversion/convert_diffusers_autoencoder_kl.py --from "madebyollin/sdxl-vae-fp16-fix" --subfolder "" --to sdxl-lda.safetensors --half
```

!!! note 
    This will download the original weights from https://huggingface.co/ which takes some time. If you already have this repo cloned locally, use the `--from /path/to/stabilityai/stable-diffusion-xl-base-1.0` option instead.

Now, we can write the Python script responsible for inference. Just create a simple `inference.py` file, and open it in your favorite editor.

Start by instantiating a [`StableDiffusion_XL`][refiners.foundationals.latent_diffusion.stable_diffusion_xl.StableDiffusion_XL] model and load it with the converted weights:

```py
import torch

from refiners.fluxion.utils import manual_seed, no_grad
from refiners.foundationals.latent_diffusion.stable_diffusion_xl import StableDiffusion_XL

# Load SDXL
sdxl = StableDiffusion_XL(device="cuda", dtype=torch.float16)  # Using half-precision for memory efficiency
sdxl.clip_text_encoder.load_from_safetensors("DoubleCLIPTextEncoder.safetensors")
sdxl.unet.load_from_safetensors("sdxl-unet.safetensors")
sdxl.lda.load_from_safetensors("sdxl-lda.safetensors")

```

Then, define the inference parameters by setting the appropriate prompt / seed / inference steps:

```py
# Hyperparameters
prompt = "a futuristic castle surrounded by a forest, mountains in the background"
seed = 42
sdxl.set_inference_steps(50, first_step=0)
sdxl.set_self_attention_guidance(
    enable=True, scale=0.75
)  # Enable self-attention guidance to enhance the quality of the generated images

# ... Inference process

```

You can now define and run the proper inference process:

```py
with no_grad():  # Disable gradient calculation for memory-efficient inference
    clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
        text=prompt + ", best quality, high quality",
        negative_text="monochrome, lowres, bad anatomy, worst quality, low quality",
    )
    time_ids = sdxl.default_time_ids

    manual_seed(seed=seed)

    # Using a higher latents inner dim to improve resolution of generated images
    x = torch.randn(size=(1, 4, 256, 256), device=sdxl.device, dtype=sdxl.dtype)

    # Diffusion process
    for step in sdxl.steps:
        if step % 10 == 0:
            print(f"Step {step}")
        x = sdxl(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            time_ids=time_ids,
        )
    predicted_image = sdxl.lda.decode_latents(x)

predicted_image.save("vanilla_sdxl.png")

```


??? example "Expand to see the entire end-to-end code"

    ```py
    import torch

    from refiners.fluxion.utils import manual_seed, no_grad
    from refiners.foundationals.latent_diffusion.stable_diffusion_xl import StableDiffusion_XL

    # Load SDXL
    sdxl = StableDiffusion_XL(device="cuda", dtype=torch.float16)
    sdxl.clip_text_encoder.load_from_safetensors("DoubleCLIPTextEncoder.safetensors")
    sdxl.unet.load_from_safetensors("sdxl-unet.safetensors")
    sdxl.lda.load_from_safetensors("sdxl-lda.safetensors")

    # Hyperparameters
    prompt = "a futuristic castle surrounded by a forest, mountains in the background"
    seed = 42
    sdxl.set_inference_steps(50, first_step=0)
    sdxl.set_self_attention_guidance(
        enable=True, scale=0.75
    )  # Enable self-attention guidance to enhance the quality of the generated images


    with no_grad():  # Disable gradient calculation for memory-efficient inference
        clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
            text=prompt + ", best quality, high quality",
            negative_text="monochrome, lowres, bad anatomy, worst quality, low quality",
        )
        time_ids = sdxl.default_time_ids

        manual_seed(seed=seed)

        # Using a higher latents inner dim to improve resolution of generated images
        x = torch.randn(size=(1, 4, 256, 256), device=sdxl.device, dtype=sdxl.dtype)

        # Diffusion process
        for step in sdxl.steps:
            if step % 10 == 0:
                print(f"Step {step}")
            x = sdxl(
                x,
                step=step,
                clip_text_embedding=clip_text_embedding,
                pooled_text_embedding=pooled_text_embedding,
                time_ids=time_ids,
            )
        predicted_image = sdxl.lda.decode_latents(x)

    predicted_image.save("vanilla_sdxl.png")

    ```


It's time to execute your code. The resulting image should look like this:

<figure markdown>
  <img src="vanilla_sdxl.webp" alt="Image title" width="400">
</figure>

It is not really what we prompted the model for, unfortunately. To get a more futuristic-looking castle, you can either go for tedious prompt engineering, or use a pretrainered LoRA tailored to our use case, like the [Sci-fi Environments](https://civitai.com/models/105945?modelVersionId=140624) LoRA available on Civitai. We'll now show you how the LoRA option works with Refiners. 

## Single LoRA

To use the [Sci-fi Environments](https://civitai.com/models/105945?modelVersionId=140624) LoRA, all you have to do is download its weights to disk as a `.safetensors`, and inject them into SDXL using [`SDLoraManager`][refiners.foundationals.latent_diffusion.lora.SDLoraManager] right after instantiating `StableDiffusion_XL`:

```py
from refiners.fluxion.utils import load_from_safetensors
from refiners.foundationals.latent_diffusion.lora import SDLoraManager

# Load LoRA weights from disk and inject them into target
manager = SDLoraManager(sdxl)
scifi_lora_weights = load_from_safetensors("Sci-fi_Environments_sdxl.safetensors")
manager.add_loras("scifi-lora", tensors=scifi_lora_weights)

```

??? example "Expand to see the entire end-to-end code"

    ```py
    import torch

    from refiners.fluxion.utils import load_from_safetensors, manual_seed, no_grad
    from refiners.foundationals.latent_diffusion.lora import SDLoraManager
    from refiners.foundationals.latent_diffusion.stable_diffusion_xl import StableDiffusion_XL

    # Load SDXL
    sdxl = StableDiffusion_XL(device="cuda", dtype=torch.float16)
    sdxl.clip_text_encoder.load_from_safetensors("DoubleCLIPTextEncoder.safetensors")
    sdxl.unet.load_from_safetensors("sdxl-unet.safetensors")
    sdxl.lda.load_from_safetensors("sdxl-lda.safetensors")

    # Load LoRA weights from disk and inject them into target
    manager = SDLoraManager(sdxl)
    scifi_lora_weights = load_from_safetensors("Sci-fi_Environments_sdxl.safetensors")
    manager.add_loras("scifi-lora", tensors=scifi_lora_weights)

    # Hyperparameters
    prompt = "a futuristic castle surrounded by a forest, mountains in the background"
    seed = 42
    sdxl.set_inference_steps(50, first_step=0)
    sdxl.set_self_attention_guidance(
        enable=True, scale=0.75
    )  # Enable self-attention guidance to enhance the quality of the generated images

    with no_grad():
        clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
            text=prompt + ", best quality, high quality",
            negative_text="monochrome, lowres, bad anatomy, worst quality, low quality",
        )
        time_ids = sdxl.default_time_ids

        manual_seed(seed=seed)

        # Using a higher latents inner dim to improve resolution of generated images
        x = torch.randn(size=(1, 4, 256, 256), device=sdxl.device, dtype=sdxl.dtype)

        # Diffusion process
        for step in sdxl.steps:
            if step % 10 == 0:
                print(f"Step {step}")
            x = sdxl(
                x,
                step=step,
                clip_text_embedding=clip_text_embedding,
                pooled_text_embedding=pooled_text_embedding,
                time_ids=time_ids,
            )
        predicted_image = sdxl.lda.decode_latents(x)

    predicted_image.save("scifi_sdxl.png")

    ```

You should get something like this - pretty neat, isn't it? 

<figure markdown>
  <img src="scifi_sdxl.webp" alt="Image title" width="400">
</figure>

## Multiple LoRAs

Continuing with our futuristic castle example, we might want to turn it, for instance, into a pixel art. 

Again, we could either try some tedious prompt engineering, 
or instead use another LoRA found on the web, such as [Pixel Art LoRA](https://civitai.com/models/120096/pixel-art-xl?modelVersionId=135931), found on Civitai. 
This is dead simple as [`SDLoraManager`][refiners.foundationals.latent_diffusion.lora.SDLoraManager] allows loading multiple LoRAs:

```py
# Load LoRAs weights from disk and inject them into target
manager = SDLoraManager(sdxl)
scifi_lora_weights = load_from_safetensors("Sci-fi_Environments_sdxl.safetensors")
pixel_art_lora_weights = load_from_safetensors("pixel-art-xl-v1.1.safetensors")
manager.add_multiple_loras(
    {"scifi-lora": scifi_lora_weights, "pixel-art-lora": pixel_art_lora_weights}
)
```

Adapters such as LoRAs also have a [scale](https://github.com/finegrain-ai/refiners/blob/fd01ba910efb764b4521254cded2530b6c31cbd4/src/refiners/fluxion/adapters/lora.py#L17) (roughly) quantifying the effect of this Adapter.
Refiners allows setting different scales for each Adapter, allowing the user to balance the effect of each Adapter:

```py
# Load LoRAs weights from disk and inject them into target
manager = SDLoraManager(sdxl)
scifi_lora_weights = load_from_safetensors("Sci-fi_Environments_sdxl.safetensors")
pixel_art_lora_weights = load_from_safetensors("pixel-art-xl-v1.1.safetensors")
manager.add_multiple_loras(
    tensors={"scifi-lora": scifi_lora_weights, "pixel-art-lora": pixel_art_lora_weights},
    scale={"scifi-lora": 1.0, "pixel-art-lora": 1.4},
)
```

??? example "Expand to see the entire end-to-end code"

    ```py 
    import torch

    from refiners.fluxion.utils import load_from_safetensors, manual_seed, no_grad
    from refiners.foundationals.latent_diffusion.lora import SDLoraManager
    from refiners.foundationals.latent_diffusion.stable_diffusion_xl import StableDiffusion_XL

    # Load SDXL
    sdxl = StableDiffusion_XL(device="cuda", dtype=torch.float16)
    sdxl.clip_text_encoder.load_from_safetensors("DoubleCLIPTextEncoder.safetensors")
    sdxl.unet.load_from_safetensors("sdxl-unet.safetensors")
    sdxl.lda.load_from_safetensors("sdxl-lda.safetensors")

    # Load LoRAs weights from disk and inject them into target
    manager = SDLoraManager(sdxl)
    scifi_lora_weights = load_from_safetensors("Sci-fi_Environments_sdxl.safetensors")
    pixel_art_lora_weights = load_from_safetensors("pixel-art-xl-v1.1.safetensors")
    manager.add_multiple_loras(
        tensors={"scifi-lora": scifi_lora_weights, "pixel-art-lora": pixel_art_lora_weights},
        scale={"scifi-lora": 1.0, "pixel-art-lora": 1.4},
    )

    # Hyperparameters
    prompt = "a futuristic castle surrounded by a forest, mountains in the background"
    seed = 42
    sdxl.set_inference_steps(50, first_step=0)
    sdxl.set_self_attention_guidance(
        enable=True, scale=0.75
    )  # Enable self-attention guidance to enhance the quality of the generated images

    with no_grad():
        clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
            text=prompt + ", best quality, high quality",
            negative_text="monochrome, lowres, bad anatomy, worst quality, low quality",
        )
        time_ids = sdxl.default_time_ids

        manual_seed(seed=seed)

        # Using a higher latents inner dim to improve resolution of generated images
        x = torch.randn(size=(1, 4, 256, 256), device=sdxl.device, dtype=sdxl.dtype)

        # Diffusion process
        for step in sdxl.steps:
            if step % 10 == 0:
                print(f"Step {step}")
            x = sdxl(
                x,
                step=step,
                clip_text_embedding=clip_text_embedding,
                pooled_text_embedding=pooled_text_embedding,
                time_ids=time_ids,
            )
        predicted_image = sdxl.lda.decode_latents(x)

    predicted_image.save("scifi_pixel_sdxl.png")

    ```

The results are looking great:

<figure markdown>
  <img src="scifi_pixel_sdxl.webp" alt="Image title" width="400">
</figure>

## Multiple LoRAs + IP-Adapter

Refiners really shines when it comes to composing different Adapters to fully exploit the possibilities of foundation models.

For instance, IP-Adapter (covered in [a previous blog post](https://blog.finegrain.ai/posts/supercharge-stable-diffusion-ip-adapter/)) is a common choice for practictioners wanting to guide the diffusion process towards a specific prompt image.

In our example, consider this image of the [Neuschwanstein Castle](https://en.wikipedia.org/wiki/Neuschwanstein_Castle):

<figure markdown>
  <img src="german-castle.jpg" alt="Image title" width="400">
  <figcaption>Credits: Bayerische Schlösserverwaltung, Anton Brandl</figcaption>
</figure>

We would like to guide the diffusion process to align with this image, using IP-Adapter. First, download the image as well as the weights of IP-Adapter by calling the following commands from your terminal (again, make sure in you're in your local `refiners` directory):

```bash
curl -O https://refine.rs/guides/adapting_sdxl/german-castle.jpg
python scripts/conversion/convert_transformers_clip_image_model.py --from "stabilityai/stable-diffusion-2-1-unclip" --to CLIPImageEncoderH.safetensors --half
curl -LO https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin
python scripts/conversion/convert_diffusers_ip_adapter.py --from ip-adapter-plus_sdxl_vit-h.bin --half
```

This will download and convert both IP-Adapter and CLIP Image Encoder pretrained weights.

Then, in your Python code, simply instantiate a [`SDXLIPAdapter`][refiners.foundationals.latent_diffusion.stable_diffusion_xl.image_prompt.SDXLIPAdapter] targetting our `sdxl.unet`, and inject it using a simple `.inject()` call:

```py
# IP-Adapter
ip_adapter = SDXLIPAdapter(
    target=sdxl.unet, 
    weights=load_from_safetensors("ip-adapter-plus_sdxl_vit-h.safetensors"),
    scale=1.0,
    fine_grained=True  # Use fine-grained IP-Adapter (i.e IP-Adapter Plus)
)
ip_adapter.clip_image_encoder.load_from_safetensors("CLIPImageEncoderH.safetensors")
ip_adapter.inject()

```

Then, at runtime, we simply compute the embedding of the image prompt through the `ip_adapter` object, and set its embedding calling `.set_clip_image_embedding()`:

```py
from PIL import Image
image_prompt = Image.open("german-castle.jpg")

with torch.no_grad():
    clip_image_embedding = ip_adapter.compute_clip_image_embedding(ip_adapter.preprocess_image(image_prompt))
    ip_adapter.set_clip_image_embedding(clip_image_embedding)

# And start the diffusion process
```

!!! note
    Be wary that composing Adapters (especially ones of different natures, such as LoRAs and IP-Adapter) can be tricky, as their respective effects can be adversarial. This is visible in our example below. In the code below, we tuned the LoRAs scales respectively to `1.5` and `1.55`. We invite you to try and test different seeds and scales to find the perfect combination!

??? example "Expand to see the entire end-to-end code"

    ```py
    import torch
    from PIL import Image

    from refiners.fluxion.utils import load_from_safetensors, manual_seed, no_grad
    from refiners.foundationals.latent_diffusion.lora import SDLoraManager
    from refiners.foundationals.latent_diffusion.stable_diffusion_xl import StableDiffusion_XL
    from refiners.foundationals.latent_diffusion.stable_diffusion_xl.image_prompt import SDXLIPAdapter

    # Load SDXL
    sdxl = StableDiffusion_XL(device="cuda", dtype=torch.float16)
    sdxl.clip_text_encoder.load_from_safetensors("DoubleCLIPTextEncoder.safetensors")
    sdxl.unet.load_from_safetensors("sdxl-unet.safetensors")
    sdxl.lda.load_from_safetensors("sdxl-lda.safetensors")

    # Load LoRAs weights from disk and inject them into target
    manager = SDLoraManager(sdxl)
    scifi_lora_weights = load_from_safetensors("Sci-fi_Environments_sdxl.safetensors")
    pixel_art_lora_weights = load_from_safetensors("pixel-art-xl-v1.1.safetensors")
    manager.add_multiple_loras(
        tensors={"scifi-lora": scifi_lora_weights, "pixel-art-lora": pixel_art_lora_weights},
        scale={"scifi-lora": 1.5, "pixel-art-lora": 1.55},
    )

    # Load IP-Adapter
    ip_adapter = SDXLIPAdapter(
        target=sdxl.unet,
        weights=load_from_safetensors("ip-adapter-plus_sdxl_vit-h.safetensors"),
        scale=1.0,
        fine_grained=True,  # Use fine-grained IP-Adapter (IP-Adapter Plus)
    )
    ip_adapter.clip_image_encoder.load_from_safetensors("CLIPImageEncoderH.safetensors")
    ip_adapter.inject()

    # Hyperparameters
    prompt = "a futuristic castle surrounded by a forest, mountains in the background"
    image_prompt = Image.open("german-castle.jpg")
    seed = 42
    sdxl.set_inference_steps(50, first_step=0)
    sdxl.set_self_attention_guidance(
        enable=True, scale=0.75
    )  # Enable self-attention guidance to enhance the quality of the generated images

    with no_grad():
        clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
            text=prompt + ", best quality, high quality",
            negative_text="monochrome, lowres, bad anatomy, worst quality, low quality",
        )
        time_ids = sdxl.default_time_ids

        clip_image_embedding = ip_adapter.compute_clip_image_embedding(ip_adapter.preprocess_image(image_prompt))
        ip_adapter.set_clip_image_embedding(clip_image_embedding)

        manual_seed(seed=seed)
        x = torch.randn(size=(1, 4, 128, 128), device=sdxl.device, dtype=sdxl.dtype)

        # Diffusion process
        for step in sdxl.steps:
            if step % 10 == 0:
                print(f"Step {step}")
            x = sdxl(
                x,
                step=step,
                clip_text_embedding=clip_text_embedding,
                pooled_text_embedding=pooled_text_embedding,
                time_ids=time_ids,
            )
        predicted_image = sdxl.lda.decode_latents(x)

    predicted_image.save("scifi_pixel_IP_sdxl.png")

    ```

The result looks convincing: we do get a *pixel-art, futuristic-looking Neuschwanstein castle*!

<figure markdown>
  <img src="scifi_pixel_IP_sdxl.webp" alt="Image title" width="400">
</figure>


## Wrap up

As you can see in this guide, composing Adapters on top of foundation models is pretty seamless in Refiners, allowing practitioners to quickly test out different combinations of Adapters for their needs. We encourage you to try out different ones, and even train some yourselves!
