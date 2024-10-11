We are happy to accept contributions from everyone.
Feel free to browse our [bounty list](https://www.finegrain.ai/bounties) to find a task you would like to work on.

This document describes the process for contributing to Refiners.

## Licensing

Refiners is a library that is freely available to use and modify under the MIT License.
It's essential to exercise caution when using external code, as any code that can affect the licensing of Refiners, including proprietary code, should not be copied and pasted.
It's worth noting that some open-source licenses can also be problematic.
For instance, we'd need to redistribute an Apache 2.0 license if you're using code from an Apache 2.0 codebase.

## Design principles

We do not enforce strict rules on the design of the code, but we do have a few guidelines that we try to follow:

- No dead code. We keep the codebase clean and remove unnecessary code/functionality.
- No unnecessary dependencies. We keep the number of dependencies to a minimum and only add new ones if necessary.
- Separate concerns. We separate the code into different modules and avoid having too many dependencies between modules. In particular, we try not to revisit existing code/models when adding new functionality. Instead, we add new functionality in a separate module with the `Adapter` pattern.
- Declarative style. We make the code as declarative, self-documenting, and easily read as possible. By reading the model's `repr`, you should understand how it works. We use explicit names for the different components of the models or the variables in the code.

## Setting up your environment

We use [Rye](https://rye-up.com/guide/installation/) to manage our development environment.
Please follow the instructions on the Rye website to install it.

Once Rye is installed, you can clone the repository and run `rye sync` to install the dependencies.

You should regularly update your Rye version by running `rye self update`.

## Linting

We use the standard integration of [ruff](https://docs.astral.sh/ruff/) in Rye to lint and format our code.
You can lint your code by running:
```bash
rye fmt
rye lint --fix
```

You should also check for typos, by running:
```bash
rye run typos
```

We also enforce strict type checking with [pyright](https://github.com/microsoft/pyright). You can run the type checker with:
```bash
rye run pyright
```

## Getting the weights

Since refiners re-implements models using fluxion, we can't directly use the weights from their original implementations, we need to convert them to the correct format.
We provide already converted weights for some models on our huggingface organization: https://huggingface.co/refiners

If you need to convert the weights yourself (e.g. for running the tests), you may use the `get_weights` command (c.f. `project.scripts` in `pyproject.toml`) to convert the weights. This command will automatically download all original weights and convert them.

### Contributing a new model conversion

1. Add a new file in the `refiners.conversion.models` module, if necessary.
2. Instantiate a new `WeightRecipe` object to declare how to translate the keys from the original weights to the refiners keys.
3. Instantiate a new `Conversion` object to relate the original weights, the converted weights and the recipe.
4. Update the associated `__init__.py` and `cli.py` files.

Note: The conversion process is not always straightforward, we would prefer if you could use the above steps (since it's the most declarative way to convert the weights),
although alternative methods are acceptable if they are well documented (you may find some examples in the existing conversion scripts).

## Running the tests

Running end-to-end tests is pretty compute-intensive.

To test everything, you can either:
- Use `REFINERS_USE_LOCAL_WEIGHTS=1` and manually convert all the model weights to the correct format before running the tests. See the above section to learn how to convert the weights.
- Use `REFINER_USE_LOCAL_WEIGHTS=0` and automatically download the weights from the huggingface hub. This is the default behavior. Though since some weights aren't available on the hub, some tests may be skipped.

> [!WARNING]
> GPU tests are notoriously hard to reproduce across different hardware. (see https://pytorch.org/docs/stable/notes/randomness.html)
> Our tests are designed to work on a single NVIDIA GeForce RTX 3090 24GB, with nvidia drivers v545.23.08 and cuda v12.3.
> If you have a different GPU/drivers, the tests may fail or give different results.

First, install test dependencies with:
```bash
rye sync --all-features
```

Some tests require cloning the original implementation of the model as they use `torch.hub.load`:
```bash
git clone git@github.com:facebookresearch/dinov2.git tests/repos/dinov2
git clone git@github.com:microsoft/Swin-Transformer.git tests/repos/Swin-Transformer
```

Finally, run the tests:
```bash
rye run pytest
```

The `-k` option is handy to run a subset of tests that match a given expression, e.g.:
```bash
rye run pytest -k "diffusion_std_init_image" -v
rye run pytest -k "not e2e" -v
rye run pytest -k "e2e" -v
```

Some tests require a GPU, and may sometime OOM if you test a lot of models sequentially.
You can re-run the failed tests with:
```bash
pytest -k "e2e" -v --last-failed
```

You can modify the following environment variables to change the behavior of the tests:
```bash
export REFINERS_USE_LOCAL_WEIGHTS=1  # checkpoints will be loaded from the local hub, instead of the hf hub
export REFINERS_TEST_DEVICE=cpu  # tests that require a GPU will be skipped
export REFINERS_HUB_PATH=/path/to/hub  # default is ./tests/weights
export REFINERS_TEST_DATASETS_DIR=/path/to/datasets  # default it ./tests/datasets
export REFINERS_TEST_REPOS_DIR=/path/to/repos  # default is ./tests/repos
```

### Code coverage

You can collect [code coverage](https://github.com/nedbat/coveragepy) data while running tests with, e.g.:
```bash
rye run test-cov
```

Then, browse the corresponding HTML report with:
```bash
rye run serve-cov-report
```
