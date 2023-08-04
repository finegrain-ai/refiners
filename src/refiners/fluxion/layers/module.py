from pathlib import Path
from typing import Any, Generator, TypeVar

from torch import device as Device, dtype as DType
from torch.nn.modules.module import Module as TorchModule

from refiners.fluxion.utils import load_from_safetensors
from refiners.fluxion.context import Context, ContextProvider

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from refiners.fluxion.layers.chain import Chain

T = TypeVar("T", bound="Module")
TContextModule = TypeVar("TContextModule", bound="ContextModule")


class Module(TorchModule):
    _parameters: dict[str, Any]
    _buffers: dict[str, Any]

    __getattr__: Callable[["Module", str], Any]  # type: ignore
    __setattr__: Callable[["Module", str, Any], None]  # type: ignore

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, *kwargs)  # type: ignore

    def load_from_safetensors(self, tensors_path: str | Path, strict: bool = True) -> "Module":
        state_dict = load_from_safetensors(tensors_path)
        self.load_state_dict(state_dict, strict=strict)
        return self

    def named_modules(self, *args: Any, **kwargs: Any) -> "Generator[tuple[str, Module], None, None]":  # type: ignore
        return super().named_modules(*args)  # type: ignore

    def to(self: T, device: Device | str | None = None, dtype: DType | None = None) -> T:  # type: ignore
        return super().to(device=device, dtype=dtype)  # type: ignore


class ContextModule(Module):
    # we store parent into a one element list to avoid pytorch thinking it's a submodule
    _parent: "list[Chain]"
    _can_refresh_parent: bool = True  # see usage in Adapter and Chain

    # Contains simple attributes set on the instance by `__init__` in subclasses
    # and copied by `structural_copy`. Note that is not the case of `device` since
    # Chain's __init__ takes care of it.
    structural_attrs: list[str] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, *kwargs)
        self._parent = []

    @property
    def parent(self) -> "Chain | None":
        return self._parent[0] if self._parent else None

    @property
    def ensure_parent(self) -> "Chain":
        assert self._parent, "module is not bound to a Chain"
        return self._parent[0]

    def _set_parent(self, parent: "Chain | None") -> None:
        if parent is None:
            self._parent = []
            return
        # Always insert the module in the Chain first to avoid inconsistencies.
        assert self in iter(parent), f"{self} not in {parent}"
        self._parent = [parent]

    @property
    def provider(self) -> ContextProvider:
        return self.ensure_parent.provider

    def get_parents(self) -> "list[Chain]":
        return self._parent + self._parent[0].get_parents() if self._parent else []

    def use_context(self, context_name: str) -> Context:
        """Retrieve the context object from the module's context provider."""
        context = self.provider.get_context(context_name)
        assert context is not None, f"Context {context_name} not found."
        return context

    def structural_copy(self: TContextModule) -> TContextModule:
        clone = object.__new__(self.__class__)
        for k in self.__class__.structural_attrs:
            setattr(clone, k, getattr(self, k))
        ContextModule.__init__(clone)
        return clone


class WeightedModule(Module):
    @property
    def device(self) -> Device:
        return self.weight.device

    @property
    def dtype(self) -> DType:
        return self.weight.dtype
