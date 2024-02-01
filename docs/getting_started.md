# Getting Started

Refiners is a micro framework on top of PyTorch with first class citizen APIs for foundation model adaptation.

Refiners requires Python 3.10 or later, its main dependency is PyTorch.

## Recommended usage (development branch, with Rye)

Refiners is still a young project and development is active, so to use the latest and greatest version of the framework we recommend you use the `main` branch from our development repository.

Moreover, we recommend using [Rye](https://rye-up.com) which simplifies several things related to Python package management, so start by following the instructions to install it on your system.

### Trying Refiners, converting weights

To try Refiners, clone the GitHub repository and install it with all optional features:

```bash
git clone "git@github.com:finegrain-ai/refiners.git"
cd refiners
rye sync --all-features
```

The format of state dicts used by Refiners is custom and we do not redistribute model weights, but we provide conversion tools and working scripts for popular models. For instance, let us convert the autoencoder from Stable Diffusion 1.5:

```bash
python "scripts/conversion/convert_diffusers_autoencoder_kl.py" --to "lda.safetensors"
```

If you need to convert weights for all models, check out `script/prepare_test_weights.py` (warning: it requires a GPU with significant VRAM and a lot of disk space).

Now let to check that it works copy your favorite 512x512 picture in the current directory as `input.png` and create `ldatest.py` with this content:

```py
from PIL import Image
from refiners.fluxion.utils import no_grad
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder

with no_grad():
    lda = SD1Autoencoder()
    lda.load_from_safetensors("lda.safetensors")

    image = Image.open("input.png")
    latents = lda.encode_image(image)
    decoded = lda.decode_latents(latents)
    decoded.save("output.png")
```

Run it:

```bash
python ldatest.py
```

Inspect `output.png`: it should be similar to `input.png` but have a few differences. Latent Autoencoders are good compressors!

### Using refiners in your own project

So far you used Refiners as a standalone package, but if you want to create your own project using it as a dependency here is how you can proceed:

```bash
rye init --py "3.11" myproject
cd myproject
rye add --git "git@github.com:finegrain-ai/refiners.git" --features training refiners
rye sync
```

If you only intend to do inference and no training, you can drop `--features training`.

To convert weights, you can either use a copy of the `refiners` repository as described above or add the `conversion` feature as a development dependency:

```bash
rye add --dev --git "git@github.com:finegrain-ai/refiners.git" --features conversion refiners
```

Note that you will still need to download the conversion scripts independently if you go that route.

### What next?

We suggest you check out the [guides](/guides/) section to dive into the usage of Refiners, of the [Key Concepts](/concepts/chain/) section for a better understanding of how the framework works.

## Advanced usage

### Using other package managers (pip, Poetry...)

We use Rye to maintain and release Refiners but it conforms to the standard Python packaging guidelines and can be used with other package managers. Please refer to their respective documentation to figure out how to install a package from Git if you intend to use the development branch, as well as how to specify features.

### Using stable releases from PyPI

Although we recommend using our development branch, we do [publish more stable releases to PyPI](https://pypi.org/project/refiners/) and you are welcome to use them in your project. However, note that the format of weights can be different from the current state of the development branch, so you will need the conversion scripts from the corresponding tag in GitHub, for instance [here for v0.2.0](https://github.com/finegrain-ai/refiners/tree/v0.2.0).
