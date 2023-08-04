import contextlib
import refiners.fluxion.layers as fl
from typing import Any, Generic, TypeVar, Iterator


T = TypeVar("T", bound=fl.Module)
TAdapter = TypeVar("TAdapter", bound="Adapter[fl.Module]")


class Adapter(Generic[T]):
    # we store _target into a one element list to avoid pytorch thinking it is a submodule
    _target: "list[T]"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        assert issubclass(cls, fl.Chain), f"Adapter {cls.__name__} must be a Chain"

    @property
    def target(self) -> T:
        return self._target[0]

    @contextlib.contextmanager
    def setup_adapter(self, target: T) -> Iterator[None]:
        assert isinstance(self, fl.Chain)
        assert (not hasattr(self, "_modules")) or (
            len(self) == 0
        ), "Call the Chain constructor in the setup_adapter context."
        self._target = [target]

        if not isinstance(self.target, fl.ContextModule):
            yield
            return

        _old_can_refresh_parent = target._can_refresh_parent
        target._can_refresh_parent = False
        yield
        target._can_refresh_parent = _old_can_refresh_parent

    def inject(self, parent: fl.Chain | None = None) -> None:
        assert isinstance(self, fl.Chain)

        if parent is None:
            if isinstance(self.target, fl.ContextModule):
                parent = self.target.parent
            else:
                raise ValueError(f"parent of {self.target} is mandatory")
        assert isinstance(parent, fl.Chain), f"{self.target} has invalid parent {parent}"
        if self.target not in iter(parent):
            raise ValueError(f"{self.target} is not in {parent}")

        parent.replace(
            old_module=self.target,
            new_module=self,
            old_module_parent=self.find_parent(self.target),
        )

    def eject(self) -> None:
        assert isinstance(self, fl.Chain)
        self.ensure_parent.replace(old_module=self, new_module=self.target)

    def _pre_structural_copy(self) -> None:
        if isinstance(self.target, fl.Chain):
            raise RuntimeError("Chain adapters typically cannot be copied, eject them first.")

    def _post_structural_copy(self: TAdapter, source: TAdapter) -> None:
        self._target = [source.target]
