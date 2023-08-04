from importlib import import_module
from importlib.metadata import requires
import sys

refiners_requires = requires("refiners")
assert refiners_requires is not None

for dep in filter(lambda r: r.endswith('extra == "training"'), refiners_requires):
    try:
        import_module(dep.split(" ")[0])
    except ImportError:
        print(
            "Some dependencies are missing. Please install refiners with the `training` extra, e.g. `pip install"
            " refiners[training]`",
            file=sys.stderr,
        )
        sys.exit(1)
