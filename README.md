<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/finegrain-ai/refiners/main/assets/logo_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/finegrain-ai/refiners/main/assets/logo_light.png">
  <img alt="Finegrain Refiners Library" width="352" height="128" style="max-width: 100%;">
</picture>

**The simplest way to train and run adapters on top of foundation models**

[**Manifesto**](https://refine.rs/home/why/) | [**Docs**](https://refine.rs) | [**Guides**](https://refine.rs/guides/adapting_sdxl/) | [**Discussions**](https://github.com/finegrain-ai/refiners/discussions) | [**Discord**](https://discord.gg/mCmjNUVV7d)

______________________________________________________________________

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/refiners)](https://pypi.org/project/refiners/)
[![PyPI Status](https://badge.fury.io/py/refiners.svg)](https://badge.fury.io/py/refiners)
[![license](https://img.shields.io/badge/license-MIT-blue)](/LICENSE)
[![code bounties](https://img.shields.io/badge/code-bounties-blue)](https://finegrain.ai/bounties)
[![chat](https://img.shields.io/discord/1179456777406922913?logo=discord&logoColor=white&color=%235765F2)](https://discord.gg/mCmjNUVV7d)
</div>

## Latest News 🔥

- Added [Euler's method](https://arxiv.org/abs/2206.00364) to solvers (contributed by [@israfelsr](https://github.com/israfelsr))
- Added [DINOv2](https://github.com/facebookresearch/dinov2) for high-performance visual features (contributed by [@Laurent2916](https://github.com/Laurent2916))
- Added [FreeU](https://github.com/ChenyangSi/FreeU) for improved quality at no cost (contributed by [@isamu-isozaki](https://github.com/isamu-isozaki))
- Added [Restart Sampling](https://github.com/Newbeeer/diffusion_restart_sampling) for improved image generation ([example](https://github.com/Newbeeer/diffusion_restart_sampling/issues/4))
- Added [Self-Attention Guidance](https://github.com/KU-CVLAB/Self-Attention-Guidance/) to avoid e.g. too smooth images ([example](https://github.com/SusungHong/Self-Attention-Guidance/issues/4))
- Added [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter) for extra guidance ([example](https://github.com/TencentARC/T2I-Adapter/discussions/93))
- Added [MultiDiffusion](https://github.com/omerbt/MultiDiffusion) for e.g. panorama images
- Added [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), aka image prompt ([example](https://github.com/tencent-ailab/IP-Adapter/issues/92))
- Added [Segment Anything](https://github.com/facebookresearch/segment-anything) to foundation models
- Added [SDXL 1.0](https://github.com/Stability-AI/generative-models) to foundation models
- Made possible to add new concepts to the CLIP text encoder, e.g. via [Textual Inversion](https://arxiv.org/abs/2208.01618)

## Installation

The current recommended way to install Refiners is from source using [Rye](https://rye-up.com/):

```bash
git clone "git@github.com:finegrain-ai/refiners.git"
cd refiners
rye sync --all-features
```

## Documentation

Refiners comes with a MkDocs-based documentation website available at https://refine.rs. You will find there a [quick start guide](https://refine.rs/getting-started/recommended/), a description of the [key concepts](https://refine.rs/concepts/chain/), as well as in-depth foundation model adaptation [guides](https://refine.rs/guides/adapting_sdxl/).

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

## Projects using Refiners

- https://github.com/brycedrennan/imaginAIry

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
