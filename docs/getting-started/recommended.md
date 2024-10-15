---
icon: material/star-outline
---

# Recommended usage

Refiners is still a young project and development is active, so to use the latest and greatest version of the framework we recommend you use the `main` branch from our development repository.

Moreover, we recommend using [Rye](https://rye.astral.sh/) which simplifies several things related to Python package management, so start by following the instructions to install it on your system.

## Installing

To try Refiners, clone the GitHub repository and install it with all optional features:

```bash
git clone git@github.com:finegrain-ai/refiners.git
cd refiners
rye sync --all-features
```

## Converting weights

The format of state dicts used by Refiners is custom, so to use pretrained models you will need to convert weights.
We provide conversion tools and pre-converted weights on our [HuggingFace organization](https://huggingface.co/refiners) for popular models.

For instance, to use the autoencoder from Stable Diffusion 1.5:

### Use pre-converted weights

```py
from huggingface_hub import hf_hub_download
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder

# download the pre-converted weights from the hub
safetensors_path = hf_hub_download(
    repo_id="refiners/sd15.autoencoder",
    filename="model.safetensors",
    revision="9ce6af42e21fce64d74b1cab57a65aea82fd40ea",  # optional
)

# initialize the model
model = SD1Autoencoder()

# load the pre-converted weights
model.load_from_safetensors(safetensors_path)
```

### Convert the weights yourself

If you want to convert the weights yourself, you can use the conversion tools we provide.

```py
from refiners.conversion import autoencoder_sd15

# This function will:
#   - download the original weights from the internet, and save them to disk at a known location
#     (e.g. tests/weights/stable-diffusion-v1-5/stable-diffusion-v1-5/vae/diffusion_pytorch_model.safetensors)
#   - convert them to the refiners format, and save them to disk at a known location
#     (e.g. tests/weights/refiners/sd15.autoencoder/model.safetensors)
autoencoder_sd15.runwayml.convert()

# get the path to the converted weights
safetensors_path = autoencoder_sd15.runwayml.converted.local_path

# initialize the model
model = SD1Autoencoder()

# load the converted weights
model.load_from_safetensors(safetensors_path)
```

!!! note
    If you need to convert more model weights or all of them, check out the `refiners.conversion` module.

!!! warning
    Converting all the weights requires a lot of disk space and CPU time, so be prepared.
    Currently downloading all the original weights takes around ~100GB of disk space,
    and converting them all takes around ~70GB of disk space.

!!! warning
    Some conversion scripts may also require quite a bit of RAM, since they load the entire weights in memory,
    ~16GB of RAM should be enough for most models, but some models may require more.


### Testing the conversion

To quickly check that the weights you got from the hub or converted yourself are correct, you can run the following snippet:

```py
from PIL import Image
from refiners.fluxion.utils import no_grad

image = Image.open("input.png")

with no_grad():
    latents = model.image_to_latents(image)
    decoded = model.latents_to_image(latents)

decoded.save("output.png")
```

Inspect `output.png`, if the converted weights are correct, it should be similar to `input.png` (but have a few differences).

## Using Refiners in your own project

So far you used Refiners as a standalone package, but if you want to create your own project using it as a dependency here is how you can proceed:

```bash
rye init --py "3.11" myproject
cd myproject
rye add refiners@git+https://github.com/finegrain-ai/refiners
rye sync
```

If you intend to use Refiners for training, you can install the `training` feature:

```bash
rye add refiners[training]@git+https://github.com/finegrain-ai/refiners
```

Similarly, if you need to use the conversion tools we provide, you install the `conversion` feature:

```bash
rye add refiners[conversion]@git+https://github.com/finegrain-ai/refiners
```

!!! note
    You can install multiple features at once by separating them with a comma:

    ```bash
    rye add refiners[training,conversion]@git+https://github.com/finegrain-ai/refiners
    ```

## What's next?

We suggest you check out the [guides](/guides/) section to dive into the usage of Refiners, of the [Key Concepts](/concepts/chain/) section for a better understanding of how the framework works.
