# Getting Started

Refiners is a micro framework on top of PyTorch with first class citizen APIs for foundation model adaptation.


## Installation

Refiners requires Python 3.10 or later, its main dependency is PyTorch.

### with git

To get the latest version of the code, clone the repository:

```bash
git clone https://github.com/finegrain-ai/refiners.git
```

Then install the package using pip:

```bash
cd refiners
pip install .
```


### with pip

Refiners is available on PyPI and can be installed using pip:

```bash
pip install refiners
```

## Run foundational models and adapters

If you want to understand how to use Refiners with existing foundational models, please refer to the specific [Models](models/index.md) page.

 - [Stable Diffusion](/models/stable_diffusion)
 - [Segment Anything](/models/segment_anything)

## Write new foundational models and adapters

To understand how to write new adapters or models with Refiners, please have a look at the [Fluxion](fluxion/index.md) documentation.