import sys
from importlib import import_module
from importlib.metadata import requires

from packaging.requirements import Requirement

refiners_requires = requires("refiners")
assert refiners_requires is not None

for dep in refiners_requires:
    req = Requirement(dep)
    marker = req.marker
    if marker is None or not marker.evaluate({"extra": "training"}):
        continue
    try:
        import_module(req.name)
    except ImportError:
        print(
            "Some dependencies are missing. Please install refiners with the `training` extra, e.g. `pip install"
            " refiners[training]`",
            file=sys.stderr,
        )
        sys.exit(1)
