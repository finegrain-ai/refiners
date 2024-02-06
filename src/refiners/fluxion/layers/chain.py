import inspect
import re
import sys
import traceback
from collections import defaultdict
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar, cast, get_origin, overload

import torch
from torch import Tensor, cat, device as Device, dtype as DType

from refiners.fluxion.context import ContextProvider, Contexts
from refiners.fluxion.layers.module import ContextModule, Module, ModuleTree, WeightedModule
from refiners.fluxion.utils import summarize_tensor

T = TypeVar("T", bound=Module)
TChain = TypeVar("TChain", bound="Chain")  # because Self (PEP 673) is not in 3.10


def generate_unique_names(
    modules: tuple[Module, ...],
) -> dict[str, Module]:
    """Generate unique names for each Module in a sequence.

    Args:
        modules: The sequence of Modules to name.
    """
    class_counts: dict[str, int] = {}
    unique_names: list[tuple[str, Module]] = []
    for module in modules:
        class_name = module.__class__.__name__
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    name_counter: dict[str, int] = {}
    for module in modules:
        class_name = module.__class__.__name__
        name_counter[class_name] = name_counter.get(class_name, 0) + 1
        unique_name = f"{class_name}_{name_counter[class_name]}" if class_counts[class_name] > 1 else class_name
        unique_names.append((unique_name, module))
    return dict(unique_names)


def structural_copy(m: T) -> T:
    """Helper function to copy a Module's tree, only if it is a ContextModule instance."""
    return m.structural_copy() if isinstance(m, ContextModule) else m


class ChainError(RuntimeError):
    """Exception raised when an error occurs during the execution of a Chain."""

    def __init__(self, message: str, /) -> None:
        super().__init__(message)


class Chain(ContextModule):
    """Chain layer.

    This layer is the main building block of Fluxion.
    It is used to compose other layers in a sequential manner.
    Similary to [`torch.nn.Sequential`][torch.nn.Sequential],
    it calls each of its sub-layers in order, chaining their outputs as inputs to the next sublayer.
    However, it also provides additional methods to manipulate its sub-layers and their context.

    Example:
        ```py
        chain = fl.Chain(
            fl.Linear(32, 64),
            fl.ReLU(),
            fl.Linear(64, 128),
        )

        tensor = torch.randn(2, 32)
        output = chain(tensor)

        assert output.shape == (2, 128)
        ```
    """

    _modules: dict[str, Module]
    _provider: ContextProvider
    _tag = "CHAIN"

    def __init__(self, *args: Module | Iterable[Module]) -> None:
        super().__init__()
        self._provider = ContextProvider()
        modules = cast(
            tuple[Module],
            (
                tuple(args[0])
                if len(args) == 1 and isinstance(args[0], Iterable) and not isinstance(args[0], Chain)
                else tuple(args)
            ),
        )

        for module in modules:
            # Violating this would mean a ContextModule ends up in two chains,
            # with a single one correctly set as its parent.
            assert (
                (not isinstance(module, ContextModule))
                or (not module._can_refresh_parent)
                or (module.parent is None)
                or (module.parent == self)
            ), f"{module.__class__.__name__} already has parent {module.parent.__class__.__name__}"

        self._regenerate_keys(modules)
        self._reset_context()

        for module in self:
            if isinstance(module, ContextModule) and module.parent != self:
                module._set_parent(self)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, torch.nn.Module):
            raise ValueError(
                "Chain does not support setting modules by attribute. Instead, use a mutation method like `append` or"
                " wrap it within a single element list to prevent pytorch from registering it as a submodule."
            )
        super().__setattr__(name, value)

    @property
    def provider(self) -> ContextProvider:
        """The [`ContextProvider`][refiners.fluxion.context.ContextProvider] of the Chain."""
        return self._provider

    def init_context(self) -> Contexts:
        """Initialize the context provider with some default values.

        This method is called when the Chain is created, and when it is reset.
        This method may be overridden by subclasses to provide default values for the context provider.
        """
        return {}

    def _register_provider(self, context: Contexts | None = None) -> None:  # TODO: rename me ?
        """Recursively update the context provider to all sub-modules.

        Args:
            context: The context to use to update the provider.
        """
        if context:
            self._provider.update_contexts(context)

        for module in self:
            if isinstance(module, Chain):
                module._register_provider(context=self._provider.contexts)

    def _reset_context(self) -> None:
        """Reset the context provider to its initial state."""
        self._register_provider(self.init_context())

    def set_context(self, context: str, value: Any) -> None:
        """Set a value in the context provider.

        Args:
            context: The context to update.
            value: The value to set.
        """
        self._provider.set_context(context, value)
        self._register_provider()

    def _show_error_in_tree(self, name: str, /, max_lines: int = 20) -> str:
        tree = ModuleTree(module=self)
        classname_counter: dict[str, int] = defaultdict(int)
        first_ancestor = self.get_parents()[-1] if self.get_parents() else self

        def find_state_dict_key(module: Module, /) -> str | None:
            for key, layer in module.named_modules():
                if layer == self:
                    return ".".join((key, name))
            return None

        for child in tree:
            classname, count = name.rsplit(sep="_", maxsplit=1) if "_" in name else (name, "1")
            if child["class_name"] == classname:
                classname_counter[classname] += 1
                if classname_counter[classname] == int(count):
                    state_dict_key = find_state_dict_key(first_ancestor)
                    child["value"] = f">>> {child['value']} | {state_dict_key}"
                    break

        tree_repr = tree._generate_tree_repr(tree.root, depth=3)  # type: ignore[reportPrivateUsage]

        lines = tree_repr.split(sep="\n")
        error_line_idx = next((idx for idx, line in enumerate(iterable=lines) if line.startswith(">>>")), 0)

        return ModuleTree.shorten_tree_repr(tree_repr, line_index=error_line_idx, max_lines=max_lines)

    @staticmethod
    def _pretty_print_args(*args: Any) -> str:
        """
        Flatten nested tuples and print tensors with their shape and other information.
        """

        def _flatten_tuple(t: Tensor | tuple[Any, ...], /) -> list[Any]:
            if isinstance(t, tuple):
                return [item for subtuple in t for item in _flatten_tuple(subtuple)]
            else:
                return [t]

        flat_args = _flatten_tuple(args)

        return "\n".join(
            [
                f"{idx}: {summarize_tensor(arg) if isinstance(arg, Tensor) else arg}"
                for idx, arg in enumerate(iterable=flat_args)
            ]
        )

    def _filter_traceback(self, *frames: traceback.FrameSummary) -> list[traceback.FrameSummary]:
        patterns_to_exclude = [
            (r"torch/nn/modules/", r"^_call_impl$"),
            (r"torch/nn/functional\.py", r""),
            (r"refiners/fluxion/layers/", r"^_call_layer$"),
            (r"refiners/fluxion/layers/", r"^forward$"),
            (r"refiners/fluxion/layers/chain\.py", r""),
            (r"", r"^_"),
        ]

        def should_exclude(frame: traceback.FrameSummary, /) -> bool:
            for filename_pattern, name_pattern in patterns_to_exclude:
                if re.search(pattern=filename_pattern, string=frame.filename) and re.search(
                    pattern=name_pattern, string=frame.name
                ):
                    return True
            return False

        return [frame for frame in frames if not should_exclude(frame)]

    def _call_layer(self, layer: Module, name: str, /, *args: Any) -> Any:
        try:
            return layer(*args)
        except Exception as e:
            exc_type, _, exc_traceback = sys.exc_info()
            assert exc_type
            tb_list = traceback.extract_tb(tb=exc_traceback)
            filtered_tb_list = self._filter_traceback(*tb_list)
            formatted_tb = "".join(traceback.format_list(extracted_list=filtered_tb_list))
            pretty_args = Chain._pretty_print_args(args)
            error_tree = self._show_error_in_tree(name)

            exception_str = re.sub(pattern=r"\n\s*\n", repl="\n", string=str(object=e))
            message = f"{formatted_tb}\n{exception_str}\n---------------\n{error_tree}\n{pretty_args}"
            if "Error" not in exception_str:
                message = f"{exc_type.__name__}:\n {message}"

            raise ChainError(message) from None

    def forward(self, *args: Any) -> Any:
        result: tuple[Any] | Any = None
        intermediate_args: tuple[Any, ...] = args
        for name, layer in self._modules.items():
            result = self._call_layer(layer, name, *intermediate_args)
            intermediate_args = (result,) if not isinstance(result, tuple) else result

        self._reset_context()
        return result

    def _regenerate_keys(self, modules: Iterable[Module]) -> None:
        self._modules = generate_unique_names(tuple(modules))  # type: ignore

    @overload
    def __getitem__(self, key: int) -> Module:
        ...

    @overload
    def __getitem__(self, key: str) -> Module:
        ...

    @overload
    def __getitem__(self, key: slice) -> "Chain":
        ...

    def __getitem__(self, key: int | str | slice) -> Module:
        if isinstance(key, slice):
            copy = self.structural_copy()
            copy._regenerate_keys(modules=list(copy)[key])
            return copy
        elif isinstance(key, str):
            return self._modules[key]
        else:
            return list(self)[key]

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __len__(self) -> int:
        return len(self._modules)

    @property
    def device(self) -> Device | None:
        """The PyTorch device of the Chain's parameters."""
        wm = self.find(WeightedModule)
        return None if wm is None else wm.device

    @property
    def dtype(self) -> DType | None:
        """The PyTorch dtype of the Chain's parameters."""
        wm = self.find(WeightedModule)
        return None if wm is None else wm.dtype

    def _walk(
        self,
        predicate: Callable[[Module, "Chain"], bool] | None = None,
        recurse: bool = False,
    ) -> Iterator[tuple[Module, "Chain"]]:
        """Walk the Chain's sub-module tree and yield each module that matches the predicate.

        The predicate is a (Module, Chain) -> bool function.
        """
        if predicate is None:
            # if no predicate is given, yield all modules
            predicate = lambda _m, _p: True
        for module in self:
            try:
                p = predicate(module, self)
            except StopIteration:
                continue
            if p:
                yield (module, self)
                if not recurse:
                    continue
            if isinstance(module, Chain):
                yield from module.walk(predicate, recurse)

    @overload
    def walk(
        self,
        predicate: Callable[[Module, "Chain"], bool] | None = None,
        recurse: bool = False,
    ) -> Iterator[tuple[Module, "Chain"]]:
        ...

    @overload
    def walk(
        self,
        predicate: type[T],
        recurse: bool = False,
    ) -> Iterator[tuple[T, "Chain"]]:
        ...

    def walk(
        self,
        predicate: type[T] | Callable[[Module, "Chain"], bool] | None = None,
        recurse: bool = False,
    ) -> Iterator[tuple[T, "Chain"]] | Iterator[tuple[Module, "Chain"]]:
        """Walk the Chain's sub-module tree and yield each module that matches the predicate.

        Args:
            predicate: The predicate to match.
            recurse: Whether to recurse into sub-Chains.

        Yields:
            Each module that matches the predicate.
        """

        if get_origin(predicate) is not None:
            raise ValueError(f"subscripted generics cannot be used as predicates")

        if isinstance(predicate, type):
            # if the predicate is a Module type
            # build a predicate function that matches the type
            return self._walk(
                predicate=lambda m, _: isinstance(m, predicate),
                recurse=recurse,
            )
        else:
            return self._walk(
                predicate=predicate,
                recurse=recurse,
            )

    def layer(self, key: str | int | Sequence[str | int], layer_type: type[T] = Module) -> T:
        """Access a layer of the Chain given its type.

        Example:
            ```py
            # same as my_chain["Linear_2"], asserts it is a Linear
            my_chain.layer("Linear_2", fl.Linear)


            # same as my_chain[3], asserts it is a Linear
            my_chain.layer(3, fl.Linear)

            # probably won't work
            my_chain.layer("Conv2d", fl.Linear)


            # same as my_chain["foo"][42]["bar"],
            # assuming bar is a MyType and all parents are Chains
            my_chain.layer(("foo", 42, "bar"), fl.MyType)
            ```

        Args:
            key: The key or path of the layer.
            layer_type: The type of the layer.

        Yields:
            The layer.

        Raises:
            AssertionError: If the layer doesn't exist or the type is invalid.
        """
        if isinstance(key, (str, int)):
            r = self[key]
            assert isinstance(r, layer_type), f"layer {key} is {type(r)}, not {layer_type}"
            return r
        if len(key) == 0:
            assert isinstance(self, layer_type), f"layer is {type(self)}, not {layer_type}"
            return self
        if len(key) == 1:
            return self.layer(key[0], layer_type)
        return self.layer(key[0], Chain).layer(key[1:], layer_type)

    def layers(
        self,
        layer_type: type[T],
        recurse: bool = False,
    ) -> Iterator[T]:
        """Walk the Chain's sub-module tree and yield each layer of the given type.

        Args:
            layer_type: The type of layer to yield.
            recurse: Whether to recurse into sub-Chains.

        Yields:
            Each module of the given layer_type.
        """
        for module, _ in self.walk(layer_type, recurse):
            yield module

    def find(self, layer_type: type[T]) -> T | None:
        """Walk the Chain's sub-module tree and return the first layer of the given type.

        Args:
            layer_type: The type of layer to find.

        Returns:
            The first module of the given layer_type, or None if it doesn't exist.
        """
        return next(self.layers(layer_type=layer_type), None)

    def ensure_find(self, layer_type: type[T]) -> T:
        """Walk the Chain's sub-module tree and return the first layer of the given type.

        Args:
            layer_type: The type of layer to find.

        Returns:
            The first module of the given layer_type.

        Raises:
            AssertionError: If the module doesn't exist.
        """
        r = self.find(layer_type)
        assert r is not None, f"could not find {layer_type} in {self}"
        return r

    def find_parent(self, module: Module) -> "Chain | None":
        """Walk the Chain's sub-module tree and return the parent of the given module.

        Args:
            module: The module whose parent to find.

        Returns:
            The parent of the given module, or None if it doesn't exist.
        """
        if module in self:  # avoid DFS-crawling the whole tree
            return self
        for _, parent in self.walk(lambda m, _: m == module):
            return parent
        return None

    def ensure_find_parent(self, module: Module) -> "Chain":
        """Walk the Chain's sub-module tree and return the parent of the given module.

        Args:
            module: The module whose parent to find.

        Returns:
            The parent of the given module.

        Raises:
            AssertionError: If the module doesn't exist.
        """
        r = self.find_parent(module)
        assert r is not None, f"could not find {module} in {self}"
        return r

    def insert(self, index: int, module: Module) -> None:
        """Insert a new module in the chain.

        Args:
            index: The index at which to insert the module.
            module: The module to insert.

        Raises:
            IndexError: If the index is out of range.
        """
        if index < 0:
            index = max(0, len(self._modules) + index + 1)
        modules = list(self)
        modules.insert(index, module)
        self._regenerate_keys(modules)
        if isinstance(module, ContextModule):
            module._set_parent(self)
        self._register_provider()

    def insert_before_type(self, module_type: type[Module], new_module: Module) -> None:
        """Insert a new module in the chain, right before the first module of the given type.

        Args:
            module_type: The type of module to insert before.
            new_module: The module to insert.

        Raises:
            ValueError: If no module of the given type exists in the chain.
        """
        for i, module in enumerate(self):
            if isinstance(module, module_type):
                self.insert(i, new_module)
                return
        raise ValueError(f"No module of type {module_type.__name__} found in the chain.")

    def insert_after_type(self, module_type: type[Module], new_module: Module) -> None:
        """Insert a new module in the chain, right after the first module of the given type.

        Args:
            module_type: The type of module to insert after.
            new_module: The module to insert.

        Raises:
            ValueError: If no module of the given type exists in the chain.
        """
        for i, module in enumerate(self):
            if isinstance(module, module_type):
                self.insert(i + 1, new_module)
                return
        raise ValueError(f"No module of type {module_type.__name__} found in the chain.")

    def append(self, module: Module) -> None:
        """Append a new module to the chain.

        Args:
            module: The module to append.
        """
        self.insert(-1, module)

    def pop(self, index: int = -1) -> Module:
        """Pop a module from the chain at the given index.

        Args:
            index: The index of the module to pop.

        Returns:
            The popped module.

        Raises:
            IndexError: If the index is out of range.
        """
        modules = list(self)
        if index < 0:
            index = len(modules) + index
        if index < 0 or index >= len(modules):
            raise IndexError("Index out of range.")
        removed_module = modules.pop(index)
        if isinstance(removed_module, ContextModule):
            removed_module._set_parent(None)
        self._regenerate_keys(modules)
        return removed_module

    def remove(self, module: Module) -> None:
        """Remove a module from the chain.

        Args:
            module: The module to remove.

        Raises:
            ValueError: If the module is not in the chain.
        """
        modules = list(self)
        try:
            modules.remove(module)
        except ValueError:
            raise ValueError(f"{module} is not in {self}")
        self._regenerate_keys(modules)
        if isinstance(module, ContextModule):
            module._set_parent(None)

    def replace(
        self,
        old_module: Module,
        new_module: Module,
        old_module_parent: "Chain | None" = None,
    ) -> None:
        """Replace a module in the chain with a new module.

        Args:
            old_module: The module to replace.
            new_module: The module to replace with.
            old_module_parent: The parent of the old module.
                If None, the old module is orphanized.

        Raises:
            ValueError: If the module is not in the chain.
        """
        modules = list(self)
        try:
            modules[modules.index(old_module)] = new_module
        except ValueError:
            raise ValueError(f"{old_module} is not in {self}")
        self._regenerate_keys(modules)
        if isinstance(new_module, ContextModule):
            new_module._set_parent(self)
        if isinstance(old_module, ContextModule):
            old_module._set_parent(old_module_parent)

    def structural_copy(self: TChain) -> TChain:
        """Copy the structure of the Chain tree.

        This method returns a recursive copy of the Chain tree where all inner nodes
        (instances of Chain and its subclasses) are duplicated and all leaves
        (regular Modules) are not.

        Such copies can be adapted without disrupting the base model, but do not
        require extra GPU memory since the weights are in the leaves and hence not copied.
        """
        if hasattr(self, "_pre_structural_copy"):
            assert callable(self._pre_structural_copy)
            self._pre_structural_copy()

        modules = [structural_copy(m) for m in self]
        clone = super().structural_copy()
        clone._provider = ContextProvider.create(clone.init_context())

        for module in modules:
            clone.append(module=module)

        if hasattr(clone, "_post_structural_copy"):
            assert callable(clone._post_structural_copy)
            clone._post_structural_copy(self)

        return clone

    def _show_only_tag(self) -> bool:
        return self.__class__ == Chain


class UseContext(ContextModule):
    """UseContext layer.

    This layer reads from the [`ContextProvider`][refiners.fluxion.context.ContextProvider]
    of its parent [`Chain`][refiners.fluxion.layers.chain.Chain].

    Note: When called, it will
        - Retrieve a value from the context using the given key
        - Transform the value with the given function (optional)
        - Return the value
    """

    def __init__(self, context: str, key: str) -> None:
        super().__init__()
        self.context = context
        self.key = key
        self.func: Callable[[Any], Any] = lambda x: x

    def __call__(self, *args: Any) -> Any:
        context = self.use_context(self.context)
        assert context, f"context {self.context} is unset"
        value = context.get(self.key)
        assert value is not None, f"context entry {self.context}.{self.key} is unset"
        return self.func(value)

    def __repr__(self):
        return f"{self.__class__.__name__}(context={repr(self.context)}, key={repr(self.key)})"

    def compose(self, func: Callable[[Any], Any]) -> "UseContext":
        self.func = func
        return self


class SetContext(ContextModule):
    """SetContext layer.

    This layer writes to the [`ContextProvider`][refiners.fluxion.context.ContextProvider]
    of its parent [`Chain`][refiners.fluxion.layers.chain.Chain].

    Note: When called (without a callback), it will
        - Update the context with the given key and the input value
        - Return the input value

    Note: When called (with a callback), it will
        - Call the callback with the current value and the input value
          (the callback may update the context with a new value, or not)
        - Return the input value

    Warning:
        The context needs to already exist in the [`ContextProvider`][refiners.fluxion.context.ContextProvider]
    """

    # TODO: Create the context if it doesn't exist

    def __init__(
        self,
        context: str,
        key: str,
        callback: Callable[[Any, Any], Any] | None = None,
    ) -> None:
        super().__init__()
        self.context = context
        self.key = key
        self.callback = callback

    def __call__(self, x: Tensor) -> Tensor:
        if context := self.use_context(self.context):
            if not self.callback:
                context.update({self.key: x})
            else:
                self.callback(context[self.key], x)

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(context={repr(self.context)}, key={repr(self.key)})"


class Lambda(Module):
    """Lambda layer.

    This layer wraps a [`Callable`][typing.Callable].

    Note: When called, it will
        - Execute the [`Callable`][typing.Callable] with the given arguments
        - Return the output of the [`Callable`][typing.Callable])

    Example:
        ```py
        lambda_layer = fl.Lambda(lambda x: x + 1)

        tensor = torch.tensor([1, 2, 3])
        output = lambda_layer(tensor)

        expected_output = torch.tensor([2, 3, 4])
        assert torch.allclose(output, expected_output)
        ```
    """

    def __init__(self, func: Callable[..., Any]) -> None:
        super().__init__()
        self.func = func

    def forward(self, *args: Any) -> Any:
        return self.func(*args)

    def __str__(self) -> str:
        func_name = getattr(self.func, "__name__", "partial_function")
        return f"Lambda({func_name}{str(inspect.signature(self.func))})"


class Parallel(Chain):
    """Parallel layer.

    This layer calls its sub-modules in parallel with the same inputs, and returns a tuple of their outputs.

    Example:
        ```py
        parallel = fl.Parallel(
            fl.Linear(32, 64),
            fl.Identity(),
            fl.Linear(32, 128),
        )

        tensor = torch.randn(2, 32)
        outputs = parallel(tensor)

        assert len(outputs) == 3
        assert outputs[0].shape == (2, 64)
        assert torch.allclose(outputs[1], tensor)
        assert outputs[2].shape == (2, 128)
        ```
    """

    _tag = "PAR"

    def forward(self, *args: Any) -> tuple[Tensor, ...]:
        return tuple(
            [
                self._call_layer(
                    module,
                    name,
                    *args,  # same input for all sub-modules
                )
                for name, module in self._modules.items()
            ],
        )

    def _show_only_tag(self) -> bool:
        return self.__class__ == Parallel


class Distribute(Chain):
    """Distribute layer.

    This layer calls its sub-modules in parallel with their respective input, and returns a tuple of their outputs.

    Example:
        ```py
        distribute = fl.Distribute(
            fl.Linear(32, 128),
            fl.Linear(64, 256),
        )

        tensor1 = torch.randn(2, 32)
        tensor2 = torch.randn(4, 64)
        outputs = distribute(tensor1, tensor2)

        assert len(outputs) == 2
        assert outputs[0].shape == (2, 128)
        assert outputs[1].shape == (4, 256)
        ```
    """

    _tag = "DISTR"

    def forward(self, *args: Any) -> tuple[Tensor, ...]:
        n, m = len(args), len(self._modules)
        assert n == m, f"Number of positional arguments ({n}) must match number of sub-modules ({m})."
        return tuple(
            [
                self._call_layer(
                    module,
                    name,
                    arg,  # each sub-module has its own input
                )
                for arg, (name, module) in zip(args, self._modules.items())
            ]
        )

    def _show_only_tag(self) -> bool:
        return self.__class__ == Distribute


class Passthrough(Chain):
    """Passthrough layer.

    This layer call its sub-modules sequentially, and returns its original inputs,
    like an [`Identity`][refiners.fluxion.layers.Identity] layer.

    Example:
        ```py
        passthrough = fl.Passthrough(
            fl.Linear(32, 128),
            fl.ReLU(),
            fl.Linear(128, 128),
        )

        tensor = torch.randn(2, 32)
        output = passthrough(tensor)

        assert torch.allclose(output, tensor)
        ```
    """

    _tag = "PASS"

    def forward(self, *inputs: Any) -> Any:
        super().forward(*inputs)
        return inputs

    def _show_only_tag(self) -> bool:
        return self.__class__ == Passthrough


class Sum(Chain):
    """Summation layer.

    This layer calls its sub-modules in parallel with the same inputs, and returns the sum of their outputs.

    Example:
        ```py
        summation = fl.Sum(
            fl.Multiply(scale=2, bias=1),
            fl.Multiply(scale=3, bias=0),
        )

        tensor = torch.ones(1)
        output = summation(tensor)

        assert torch.allclose(output, torch.tensor([6.0]))
        ```
    """

    _tag = "SUM"

    def forward(self, *inputs: Any) -> Any:
        output = None
        for layer in self:
            layer_output: Any = layer(*inputs)
            if isinstance(layer_output, tuple):
                layer_output = sum(layer_output)  # type: ignore
            output = layer_output if output is None else output + layer_output
        return output

    def _show_only_tag(self) -> bool:
        return self.__class__ == Sum


class Residual(Chain):
    """Residual layer.

    This layer calls its sub-modules sequentially, and adds the original input to the output.

    Example:
        ```py
        residual = fl.Residual(
            fl.Multiply(scale=10),
        )

        tensor = torch.ones(2, 32)
        output = residual(tensor)

        assert output.shape == (2, 32)
        assert torch.allclose(output, 10 * tensor + tensor)
        ```
    """

    _tag = "RES"

    def forward(self, *inputs: Any) -> Any:
        assert len(inputs) == 1, "Residual connection can only be used with a single input."
        return super().forward(*inputs) + inputs[0]


class Concatenate(Chain):
    """Concatenation layer.

    This layer calls its sub-modules in parallel with the same inputs, and returns the concatenation of their outputs.

    Example:
        ```py
        concatenate = fl.Concatenate(
            fl.Linear(32, 128),
            fl.Linear(32, 128),
            dim=1,
        )

        tensor = torch.randn(2, 32)
        output = concatenate(tensor)

        assert output.shape == (2, 256)
        ```
    """

    _tag = "CAT"

    def __init__(self, *modules: Module, dim: int = 0) -> None:
        super().__init__(*modules)
        self.dim = dim

    def forward(self, *args: Any) -> Tensor:
        outputs = [module(*args) for module in self]
        return cat(
            [output for output in outputs if output is not None],
            dim=self.dim,
        )

    def _show_only_tag(self) -> bool:
        return self.__class__ == Concatenate


class Matmul(Chain):
    """Matrix multiplication layer.

    This layer returns the matrix multiplication of the outputs of its two sub-modules.

    Example:
        ```py
        matmul = fl.Matmul(
            fl.Identity(),
            fl.Multiply(scale=2),
        )

        tensor = torch.randn(10, 10)
        output = matmul(tensor)

        expected_output = tensor @ (2 * tensor)
        assert torch.allclose(output, expected_output)
        ```
    """

    _tag = "MATMUL"

    def __init__(self, input: Module, other: Module) -> None:
        super().__init__(
            input,
            other,
        )

    def forward(self, *args: Tensor) -> Tensor:
        return torch.matmul(
            input=self[0](*args),
            other=self[1](*args),
        )


class ReturnException(Exception):
    """Exception raised when a Return module is encountered."""

    def __init__(self, value: Tensor):
        self.value = value


class Return(Module):
    """Return layer.

    This layer stops the execution of a Chain when encountered.
    """

    def forward(self, x: Tensor):
        raise ReturnException(x)


class Breakpoint(ContextModule):
    """Breakpoint layer.

    This layer pauses the execution when encountered, and opens a debugger.
    """

    def __init__(self, vscode: bool = True):
        super().__init__()
        self.vscode = vscode

    def forward(self, *args: Any):
        if self.vscode:
            import debugpy  # type: ignore

            debugpy.breakpoint()  # type: ignore
        else:
            breakpoint()
        return args[0] if len(args) == 1 else args
