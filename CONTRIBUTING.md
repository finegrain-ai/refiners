We are happy to accept contributions from everyone. Feel free to browse our [bounty list](https://www.finegrain.ai/bounties) to find a task you would like to work on.

This document describes the process for contributing to Refiners.

## Licensing

Refiners is a library that is freely available to use and modify under the MIT License. It's essential to exercise caution when using external code, as any code that can affect the licensing of Refiners, including proprietary code, should not be copied and pasted. It's worth noting that some open-source licenses can also be problematic. For instance, we'd need to redistribute an Apache 2.0 license if you're using code from an Apache 2.0 codebase.

## Design principles

We do not enforce strict rules on the design of the code, but we do have a few guidelines that we try to follow:

- No dead code. We keep the codebase clean and remove unnecessary code/functionality.
- No unnecessary dependencies. We keep the number of dependencies to a minimum and only add new ones if necessary.
- Separate concerns. We separate the code into different modules and avoid having too many dependencies between modules. In particular, we try not to revisit existing code/models when adding new functionality. Instead, we add new functionality in a separate module with the `Adapter` pattern.
- Declarative style. We make the code as declarative, self-documenting, and easily read as possible. By reading the model's `repr`, you should understand how it works. We use explicit names for the different components of the models or the variables in the code.

## Setting up your environment

We use [Rye](https://rye-up.com/guide/installation/) to manage our development environment. Please follow the instructions on the Rye website to install it.

Once Rye is installed, you can clone the repository and run `rye sync` to install the dependencies.

## Linting

We use [ruff](https://docs.astral.sh/ruff/) to lint our code. You can lint your code by running.

```bash
rye run lint
```

We also enforce strict type checking with [pyright](https://github.com/microsoft/pyright). You can run the type checker with:

```bash
rye run pyright
```

## Running the tests

Running end-to-end tests is pretty compute-intensive, and you must convert all the model weights to the correct format before you can run them.

First, install test dependencies with:

```bash
rye sync --all-features
```

Then, download and convert all the necessary weights. Be aware that this will use around 50 GB of disk space:

```bash
python scripts/prepare_test_weights.py
```

Finally, run the tests:

```bash
rye run pytest
```

The `-k` option is handy to run a subset of tests that match a given expression, e.g.:

```bash
rye run pytest -k diffusion_std_init_image
```

You can enforce running tests on CPU. Tests that require a GPU will be skipped.

```bash
REFINERS_TEST_DEVICE=cpu rye run pytest
```

You can collect [code coverage](https://github.com/nedbat/coveragepy) data while running tests with, e.g.:

```bash
rye run test-cov
```

Then, browse the corresponding HTML report with:

```bash
rye run serve-cov-report
```
