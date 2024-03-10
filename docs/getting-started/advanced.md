---
icon: material/wrench-cog-outline
---

# Advanced usage

## Using other package managers (pip, Poetry...)

We use Rye to maintain and release Refiners but it conforms to the standard Python packaging guidelines and can be used with other package managers. Please refer to their respective documentation to figure out how to install a package from Git if you intend to use the development branch, as well as how to specify features.

## Using stable releases from PyPI

Although we recommend using our development branch, we do [publish more stable releases to PyPI](https://pypi.org/project/refiners/) and you are welcome to use them in your project. However, note that the format of weights can be different from the current state of the development branch, so you will need the conversion scripts from the corresponding tag in GitHub, for instance [here for v0.2.0](https://github.com/finegrain-ai/refiners/tree/v0.2.0).
