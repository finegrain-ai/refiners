<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/finegrain-ai/refiners/main/assets/logo_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/finegrain-ai/refiners/main/assets/logo_light.png">
  <img alt="Finegrain Refiners Library" width="352" height="128" style="max-width: 100%;">
</picture>

**The simplest way to train and run adapters on top of foundation models**

[**Manifesto**](https://refine.rs/home/why/) |
[**Docs**](https://refine.rs) |
[**Guides**](https://refine.rs/guides/adapting_sdxl/) |
[**Discussions**](https://github.com/finegrain-ai/refiners/discussions) |
[**Discord**](https://discord.gg/a4w4jXJ6)

______________________________________________________________________

[![dependencies - Rye](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/rye/main/artwork/badge.json)](https://github.com/astral-sh/rye)
[![linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![packaging - Hatch](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/refiners)](https://pypi.org/project/refiners/)
[![PyPI - Status](https://badge.fury.io/py/refiners.svg)](https://pypi.org/project/refiners/)
[![license](https://img.shields.io/badge/license-MIT-blue)](/LICENSE) \
[![code bounties](https://img.shields.io/badge/code-bounties-blue)](https://finegrain.ai/bounties)
[![Discord](https://img.shields.io/discord/1179456777406922913?logo=discord&logoColor=white&color=%235765F2)](https://discord.gg/a4w4jXJ6)
[![HuggingFace - Refiners](https://img.shields.io/badge/refiners-ffd21e?logo=huggingface&labelColor=555)](https://huggingface.co/refiners)
[![HuggingFace - Finegrain](https://img.shields.io/badge/finegrain-ffd21e?logo=huggingface&labelColor=555)](https://huggingface.co/finegrain)
[![ComfyUI Registry](https://img.shields.io/badge/ComfyUI_Registry-comfyui--refiners-1a56db)](https://registry.comfy.org/publishers/finegrain/nodes/comfyui-refiners)

</div>

## Latest News 🔥

- Added [ELLA](https://arxiv.org/abs/2403.05135) for better prompts handling (contributed by [@ily-R](https://github.com/ily-R))
- Added the Box Segmenter all-in-one solution ([model](https://huggingface.co/finegrain/finegrain-box-segmenter), [HF Space](https://huggingface.co/spaces/finegrain/finegrain-object-cutter))
- Added [MVANet](https://arxiv.org/abs/2404.07445) for high resolution segmentation
- Added [IC-Light](https://github.com/lllyasviel/IC-Light) to manipulate the illumination of images
- Added Multi Upscaler for high-resolution image generation, inspired from [Clarity Upscaler](https://github.com/philz1337x/clarity-upscaler) ([HF Space](https://huggingface.co/spaces/finegrain/enhancer))
- Added [HQ-SAM](https://arxiv.org/abs/2306.01567) for high quality mask prediction with Segment Anything
- ...see past [releases](https://github.com/finegrain-ai/refiners/releases)

## Installation

The current recommended way to install Refiners is from source using [Rye](https://rye-up.com/):

```bash
git clone "git@github.com:finegrain-ai/refiners.git"
cd refiners
rye sync --all-features
```

## Documentation

Refiners comes with a MkDocs-based documentation website available at https://refine.rs. You will find there a [quick start guide](https://refine.rs/getting-started/recommended/), a description of the [key concepts](https://refine.rs/concepts/chain/), as well as in-depth foundation model adaptation [guides](https://refine.rs/guides/adapting_sdxl/).

## Projects using Refiners

- [Finegrain Editor](https://editor.finegrain.ai/signup?utm_source=github&utm_campaign=refiners): use state-of-the-art visual AI skills to edit product photos
- [Visoid](https://www.visoid.com/): AI-powered architectural visualization
- [brycedrennan/imaginAIry](https://github.com/brycedrennan/imaginAIry): Pythonic AI generation of images and videos
- [chloedia/layerdiffuse](https://github.com/chloedia/layerdiffuse): an implementation of [LayerDiffuse](https://arxiv.org/abs/2402.17113v3) (foreground generation only)
- [Pinokio](https://pinokio.computer/): a browser for running AI apps locally (see [Clarity Refiners UI](https://pinokio.computer/item?uri=https://github.com/pinokiofactory/clarity-refiners-ui) and [announcement](https://x.com/cocktailpeanut/status/1858938348955443475))

## Awesome Adaptation Papers

If you're interested in understanding the diversity of use cases for foundation model adaptation (potentially beyond the specific adapters supported by Refiners), we suggest you take a look at these outstanding papers:

- [ControlNet](https://arxiv.org/abs/2302.05543)
- [T2I-Adapter](https://arxiv.org/abs/2302.08453)
- [IP-Adapter](https://arxiv.org/abs/2308.06721)
- [Medical SAM Adapter](https://arxiv.org/abs/2304.12620)
- [3DSAM-adapter](https://arxiv.org/abs/2306.13465)
- [SAM-adapter](https://arxiv.org/abs/2304.09148)
- [Cross Modality Attention Adapter](https://arxiv.org/abs/2307.01124)
- [UniAdapter](https://arxiv.org/abs/2302.06605)

## Credits

We took inspiration from these great projects:

- [tinygrad](https://github.com/tinygrad/tinygrad) - For something between PyTorch and [karpathy/micrograd](https://github.com/karpathy/micrograd)
- [Composer](https://github.com/mosaicml/composer) - A PyTorch Library for Efficient Neural Network Training
- [Keras](https://github.com/keras-team/keras) - Deep Learning for humans

## Citation

```bibtex
@misc{the-finegrain-team-2023-refiners,
  author = {Benjamin Trom and Pierre Chapuis and Cédric Deltheil},
  title = {Refiners: The simplest way to train and run adapters on top of foundation models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/finegrain-ai/refiners}}
}
```
