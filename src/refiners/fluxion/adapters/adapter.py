import contextlib
from typing import Any, Generic, Iterator, TypeVar

import refiners.fluxion.layers as fl

T = TypeVar("T", bound=fl.Module)
TAdapter = TypeVar("TAdapter", bound="Adapter[Any]")  # Self (see PEP 673)


class Adapter(Generic[T]):
    """Base class for adapters.

    An Adapter modifies the structure of a [`Module`][refiners.fluxion.layers.Module]
    (typically by adding, removing or replacing layers), to adapt it to a new task.
    """

    # we store _target into a one element list to avoid pytorch thinking it is a submodule
    _target: "list[T]"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        assert issubclass(cls, fl.Chain), f"Adapter {cls.__name__} must be a Chain"

    @property
    def target(self) -> T:
        """The target of the adapter."""
        return self._target[0]

    @contextlib.contextmanager
    def setup_adapter(self, target: T) -> Iterator[None]:
        """Setup the adapter.

        This method should be called by the constructor of the adapter.
        It sets the target of the adapter and ensures that the adapter
        is not a submodule of the target.

        Args:
            target: The target of the adapter.
        """
        assert isinstance(self, fl.Chain)
        assert (not hasattr(self, "_modules")) or (
            len(self) == 0
        ), "Call the Chain constructor in the setup_adapter context."
        self._target = [target]

        if isinstance(target, fl.ContextModule):
            assert isinstance(target, fl.ContextModule)
            with target.no_parent_refresh():
                yield
        else:
            yield

    def inject(self: TAdapter, parent: fl.Chain | None = None) -> TAdapter:
        """Inject the adapter.

        This method replaces the target of the adapter by the adapter inside the parent of the target.

        Args:
            parent: The parent to inject the adapter into, if the target doesn't have a parent.
        """
        assert isinstance(self, fl.Chain)

        if (parent is None) and isinstance(self.target, fl.ContextModule):
            parent = self.target.parent
            if parent is not None:
                assert isinstance(parent, fl.Chain), f"{self.target} has invalid parent {parent}"

        target_parent = self.find_parent(self.target)

        if parent is None:
            if isinstance(self.target, fl.ContextModule):
                self.target._set_parent(target_parent)  # type: ignore[reportPrivateUsage]
            return self

        # In general, `true_parent` is `parent`. We do this to support multiple adaptation,
        # i.e. initializing two adapters before injecting them.
        true_parent = parent.ensure_find_parent(self.target)
        true_parent.replace(
            old_module=self.target,
            new_module=self,
            old_module_parent=target_parent,
        )
        return self

    def eject(self) -> None:
        """Eject the adapter.

        This method is the inverse of [`inject`][refiners.fluxion.adapters.Adapter.inject],
        and should leave the target in the same state as before the injection.
        """
        assert isinstance(self, fl.Chain)

        # In general, the "actual target" is the target.
        # Here we deal with the edge case where the target
        # is part of the replacement block and has been adapted by
        # another adapter after this one. For instance, this is the
        # case when stacking Controlnets.
        actual_target = lookup_top_adapter(self, self.target)

        if (parent := self.parent) is None:
            if isinstance(actual_target, fl.ContextModule):
                actual_target._set_parent(None)  # type: ignore[reportPrivateUsage]
        else:
            parent.replace(old_module=self, new_module=actual_target)

    def _pre_structural_copy(self) -> None:
        if isinstance(self.target, fl.Chain):
            raise RuntimeError(f"Chain adapters ({self}) typically cannot be copied, eject them first.")

    def _post_structural_copy(self: TAdapter, source: TAdapter) -> None:
        self._target = [source.target]


def lookup_top_adapter(top: fl.Chain, target: fl.Module) -> fl.Module:
    """Lookup and return last adapter in parents tree (or target if none)."""

    target_parent = top.find_parent(target)
    if (target_parent is None) or (target_parent == top):
        return target

    r, p = target, target_parent
    while p != top:
        if isinstance(p, Adapter):
            r = p
        assert p.parent, f"parent tree of {top} is broken"
        p = p.parent
    return r
