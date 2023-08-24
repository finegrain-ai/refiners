from collections import defaultdict
from enum import Enum, auto
from pathlib import Path
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
import torch
from typing import Any, DefaultDict, TypedDict

from refiners.fluxion.utils import norm, save_to_safetensors

TORCH_BASIC_LAYERS: list[type[nn.Module]] = [
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.Embedding,
    nn.MaxPool2d,
    nn.AvgPool2d,
    nn.AdaptiveAvgPool2d,
]


ModelTypeShape = tuple[str, tuple[torch.Size, ...]]


class ModuleArgsDict(TypedDict):
    """Represents positional and keyword arguments passed to a module.

    - `positional`: A tuple of positional arguments.
    - `keyword`: A dictionary of keyword arguments.
    """

    positional: tuple[Any, ...]
    keyword: dict[str, Any]


class ConversionStage(Enum):
    """Represents the current stage of the conversion process.

    - `INIT`: The conversion process has not started.
    - `BASIC_LAYERS_MATCH`: The source and target models have the same number of basic layers.
    """

    INIT = auto()
    BASIC_LAYERS_MATCH = auto()
    SHAPE_AND_LAYERS_MATCH = auto()
    MODELS_OUTPUT_AGREE = auto()


class ModelConverter:
    ModuleArgs = tuple[Any, ...] | dict[str, Any] | ModuleArgsDict
    stage: ConversionStage = ConversionStage.INIT
    _stored_mapping: dict[str, str] | None = None

    def __init__(
        self,
        source_model: nn.Module,
        target_model: nn.Module,
        source_keys_to_skip: list[str] | None = None,
        target_keys_to_skip: list[str] | None = None,
        custom_layer_mapping: dict[type[nn.Module], type[nn.Module]] | None = None,
        threshold: float = 1e-5,
        skip_output_check: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Create a ModelConverter.

        - `source_model`: The model to convert from.
        - `target_model`: The model to convert to.
        - `source_keys_to_skip`: A list of keys to skip when tracing the source model.
        - `target_keys_to_skip`: A list of keys to skip when tracing the target model.
        - `custom_layer_mapping`: A dictionary mapping custom layer types between the source and target models.
        - `threshold`: The threshold for comparing outputs between the source and target models.
        - `skip_output_check`: Whether to skip comparing the outputs of the source and target models.
        - `verbose`: Whether to print messages during the conversion process.

        The conversion process consists of three stages:

        1. Verify that the source and target models have the same number of basic layers.
        2. Find matching shapes and layers between the source and target models.
        3. Convert the source model's state_dict to match the target model's state_dict.
        4. Compare the outputs of the source and target models.

        The conversion process can be run multiple times, and will resume from the last stage.

        ### Example:
        ```
        converter = ModelConverter(source_model=source, target_model=target, threshold=0.1, verbose=False)
        is_converted = converter(args)
        if is_converted:
            converter.save_to_safetensors(path="test.pt")
        ```
        """
        self.source_model = source_model
        self.target_model = target_model
        self.source_keys_to_skip = source_keys_to_skip or []
        self.target_keys_to_skip = target_keys_to_skip or []
        self.custom_layer_mapping = custom_layer_mapping or {}
        self.threshold = threshold
        self.skip_output_check = skip_output_check
        self.verbose = verbose

    def run(self, source_args: ModuleArgs, target_args: ModuleArgs | None = None) -> bool:
        """
        Run the conversion process.

        - `source_args`: The arguments to pass to the source model it can be either a tuple of positional arguments,
            a dictionary of keyword arguments, or a dictionary with `positional` and `keyword` keys. If `target_args`
            is not provided, these arguments will also be passed to the target model.
        - `target_args`: The arguments to pass to the target model it can be either a tuple of positional arguments,
            a dictionary of keyword arguments, or a dictionary with `positional` and `keyword` keys.

        ### Returns:

        - `True` if the conversion process is done and the models agree.

        The conversion process consists of three stages:

        1. Verify that the source and target models have the same number of basic layers.
        2. Find matching shapes and layers between the source and target models.
        3. Convert the source model's state_dict to match the target model's state_dict.
        4. Compare the outputs of the source and target models.

        The conversion process can be run multiple times, and will resume from the last stage.
        """
        if target_args is None:
            target_args = source_args

        match self.stage:
            case ConversionStage.MODELS_OUTPUT_AGREE:
                self._log(message="Conversion is done: you can export the converted model using `save_to_safetensors`")
                return True

            case ConversionStage.SHAPE_AND_LAYERS_MATCH if self._run_models_output_agree_stage():
                self.stage = ConversionStage.MODELS_OUTPUT_AGREE
                return True

            case ConversionStage.BASIC_LAYERS_MATCH if self._run_basic_layers_match_stage(
                source_args=source_args, target_args=target_args
            ):
                self.stage = (
                    ConversionStage.SHAPE_AND_LAYERS_MATCH
                    if not self.skip_output_check
                    else ConversionStage.MODELS_OUTPUT_AGREE
                )
                return self.run(source_args=source_args, target_args=target_args)

            case ConversionStage.INIT if self._run_init_stage():
                self.stage = ConversionStage.BASIC_LAYERS_MATCH
                return self.run(source_args=source_args, target_args=target_args)

            case _:
                return False

    def __repr__(self) -> str:
        return (
            f"ModelConverter(source_model={self.source_model.__class__.__name__},"
            f" target_model={self.target_model.__class__.__name__}, stage={self.stage})"
        )

    def __bool__(self) -> bool:
        return self.stage == ConversionStage.MODELS_OUTPUT_AGREE

    def get_state_dict(self) -> dict[str, Tensor]:
        """Get the converted state_dict."""
        if not self:
            raise ValueError("The conversion process is not done yet. Run `converter(args)` first.")
        return self.target_model.state_dict()

    def get_mapping(self) -> dict[str, str]:
        """Get the mapping between the source and target models' state_dicts."""
        if not self:
            raise ValueError("The conversion process is not done yet. Run `converter(args)` first.")
        assert self._stored_mapping is not None, "Mapping is not stored"
        return self._stored_mapping

    def save_to_safetensors(self, path: Path | str, metadata: dict[str, str] | None = None, half: bool = False) -> None:
        """Save the converted model to a SafeTensors file.

        This method can only be called after the conversion process is done.

        - `path`: The path to save the converted model to.
        - `metadata`: Metadata to save with the converted model.
        - `half`: Whether to save the converted model as half precision.

        ### Raises:
        - `ValueError` if the conversion process is not done yet. Run `converter(args)` first.
        """
        if not self:
            raise ValueError("The conversion process is not done yet. Run `converter(args)` first.")
        state_dict = self.get_state_dict()
        if half:
            state_dict = {key: value.half() for key, value in state_dict.items()}
        save_to_safetensors(path=path, tensors=state_dict, metadata=metadata)

    def map_state_dicts(
        self,
        source_args: ModuleArgs,
        target_args: ModuleArgs | None = None,
    ) -> dict[str, str] | None:
        """
        Find a mapping between the source and target models' state_dicts.

        - `source_args`: The arguments to pass to the source model it can be either a tuple of positional arguments,
            a dictionary of keyword arguments, or a dictionary with `positional` and `keyword` keys. If `target_args`
            is not provided, these arguments will also be passed to the target model.
        - `target_args`: The arguments to pass to the target model it can be either a tuple of positional arguments,
            a dictionary of keyword arguments, or a dictionary with `positional` and `keyword` keys.

        ### Returns:
        - A dictionary mapping keys in the target model's state_dict to keys in the source model's state_dict.
        """
        if target_args is None:
            target_args = source_args

        source_order = self._trace_module_execution_order(
            module=self.source_model, args=source_args, keys_to_skip=self.source_keys_to_skip
        )
        target_order = self._trace_module_execution_order(
            module=self.target_model, args=target_args, keys_to_skip=self.target_keys_to_skip
        )

        if not self._assert_shapes_aligned(source_order=source_order, target_order=target_order):
            return None

        mapping: dict[str, str] = {}
        for model_type_shape in source_order:
            source_keys = source_order[model_type_shape]
            target_keys = target_order[model_type_shape]
            mapping.update(zip(target_keys, source_keys))

        return mapping

    def compare_models(
        self,
        source_args: ModuleArgs,
        target_args: ModuleArgs | None = None,
        threshold: float = 1e-5,
    ) -> bool:
        """
        Compare the outputs of the source and target models.

        - `source_args`: The arguments to pass to the source model it can be either a tuple of positional arguments,
            a dictionary of keyword arguments, or a dictionary with `positional` and `keyword` keys. If `target_args`
            is not provided, these arguments will also be passed to the target model.
        - `target_args`: The arguments to pass to the target model it can be either a tuple of positional arguments,
            a dictionary of keyword arguments, or a dictionary with `positional` and `keyword` keys.
        - `threshold`: The threshold for comparing outputs between the source and target models.
        """
        if target_args is None:
            target_args = source_args

        source_outputs = self._collect_layers_outputs(
            module=self.source_model, args=source_args, keys_to_skip=self.source_keys_to_skip
        )
        target_outputs = self._collect_layers_outputs(
            module=self.target_model, args=target_args, keys_to_skip=self.target_keys_to_skip
        )

        prev_source_key, prev_target_key = None, None
        for (source_key, source_output), (target_key, target_output) in zip(source_outputs, target_outputs):
            diff = norm(source_output - target_output).item()
            if diff > threshold:
                self._log(
                    f"Models diverged between {prev_source_key} and {source_key}, and between {prev_target_key} and"
                    f" {target_key}, difference in norm: {diff}"
                )
                return False
            prev_source_key, prev_target_key = source_key, target_key

        return True

    def _run_init_stage(self) -> bool:
        """Run the init stage of the conversion process."""
        is_count_correct = self._verify_basic_layers_count()
        is_not_missing_layers = self._verify_missing_basic_layers()

        return is_count_correct and is_not_missing_layers

    def _run_basic_layers_match_stage(self, source_args: ModuleArgs, target_args: ModuleArgs | None) -> bool:
        """Run the basic layers match stage of the conversion process."""
        self._log(message="Finding matching shapes and layers...")

        mapping = self.map_state_dicts(source_args=source_args, target_args=target_args)
        self._stored_mapping = mapping
        if mapping is None:
            self._log(message="Models do not have matching shapes.")
            return False

        self._log(message="Found matching shapes and layers. Converting state_dict...")

        source_state_dict = self.source_model.state_dict()
        target_state_dict = self.target_model.state_dict()
        converted_state_dict = self._convert_state_dict(
            source_state_dict=source_state_dict, target_state_dict=target_state_dict, state_dict_mapping=mapping
        )
        self.target_model.load_state_dict(state_dict=converted_state_dict)

        return True

    def _run_shape_and_layers_match_stage(self, source_args: ModuleArgs, target_args: ModuleArgs | None) -> bool:
        """Run the shape and layers match stage of the conversion process."""
        if self.compare_models(source_args=source_args, target_args=target_args, threshold=self.threshold):
            self._log(message="Models agree. You can export the converted model using `save_to_safetensors`")
            return True
        else:
            self._log(message="Models do not agree. Try to increase the threshold or modify the models.")
            return False

    def _run_models_output_agree_stage(self) -> bool:
        """Run the models output agree stage of the conversion process."""
        self._log(message="Conversion is done: you can export the converted model using `save_to_safetensors`")
        return True

    def _log(self, message: str) -> None:
        """Print a message if `verbose` is `True`."""
        if self.verbose:
            print(message)

    def _debug_print_shapes(
        self,
        shape: ModelTypeShape,
        source_keys: list[str],
        target_keys: list[str],
    ) -> None:
        """Print the shapes of the sub-modules in `source_keys` and `target_keys`."""
        self._log(message=f"{shape}")
        max_len = max(len(source_keys), len(target_keys))
        for i in range(max_len):
            source_key = source_keys[i] if i < len(source_keys) else "---"
            target_key = target_keys[i] if i < len(target_keys) else "---"
            self._log(f"\t{source_key}\t{target_key}")

    @staticmethod
    def _unpack_module_args(module_args: ModuleArgs) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Unpack the positional and keyword arguments passed to a module."""
        match module_args:
            case tuple(positional_args):
                keyword_args: dict[str, Any] = {}
            case {"positional": positional_args, "keyword": keyword_args}:
                pass
            case _:
                positional_args = ()
                keyword_args = dict(**module_args)

        return positional_args, keyword_args

    def _infer_basic_layer_type(self, module: nn.Module) -> type[nn.Module] | None:
        """Infer the type of a basic layer."""
        for layer_type in TORCH_BASIC_LAYERS:
            if isinstance(module, layer_type):
                return layer_type

        for source_type in self.custom_layer_mapping.keys():
            if isinstance(module, source_type):
                return source_type

        return None

    def get_module_signature(self, module: nn.Module) -> ModelTypeShape:
        """Get the signature of a module."""
        layer_type = self._infer_basic_layer_type(module=module)
        assert layer_type is not None, f"Module {module} is not a basic layer"
        param_shapes = [p.shape for p in module.parameters()]
        return (str(object=layer_type), tuple(param_shapes))

    def _count_basic_layers(self, module: nn.Module) -> dict[type[nn.Module], int]:
        """Count the number of basic layers in a module."""
        count: DefaultDict[type[nn.Module], int] = defaultdict(int)
        for submodule in module.modules():
            layer_type = self._infer_basic_layer_type(module=submodule)
            if layer_type is not None:
                count[layer_type] += 1

        return count

    def _verify_basic_layers_count(self) -> bool:
        """Verify that the source and target models have the same number of basic layers."""
        source_layers = self._count_basic_layers(module=self.source_model)
        target_layers = self._count_basic_layers(module=self.target_model)

        diff: dict[type[nn.Module], tuple[int, int]] = {}
        for layer_type in set(source_layers.keys()) | set(target_layers.keys()):
            source_count = source_layers.get(layer_type, 0)
            target_count = target_layers.get(layer_type, 0)
            if source_count != target_count:
                diff[layer_type] = (source_count, target_count)

        if diff:
            message = "Models do not have the same number of basic layers:\n"
            for layer_type, counts in diff.items():
                message += f"  {layer_type}: Source {counts[0]} - Target {counts[1]}\n"
            self._log(message=message.rstrip())
            return False

        return True

    def _is_weighted_leaf_module(self, module: nn.Module) -> bool:
        """Check if a module is a leaf module with weights."""
        return next(module.parameters(), None) is not None and next(module.children(), None) is None

    def _check_for_missing_basic_layers(self, module: nn.Module) -> list[type[nn.Module]]:
        """Check if a module has weighted leaf modules that are not basic layers."""
        return [
            type(submodule)
            for submodule in module.modules()
            if self._is_weighted_leaf_module(module=submodule) and not self._infer_basic_layer_type(module=submodule)
        ]

    def _verify_missing_basic_layers(self) -> bool:
        """Verify that the source and target models do not have missing basic layers."""
        missing_source_layers = self._check_for_missing_basic_layers(module=self.source_model)
        missing_target_layers = self._check_for_missing_basic_layers(module=self.target_model)

        if missing_source_layers or missing_target_layers:
            self._log(
                message=(
                    "Models might have missing basic layers. You can either pass them into keys to skip or set"
                    f" `check_missing_basic_layer` to `False`: {missing_source_layers}, {missing_target_layers}"
                )
            )
            return False

        return True

    @torch.no_grad()
    def _trace_module_execution_order(
        self,
        module: nn.Module,
        args: ModuleArgs,
        keys_to_skip: list[str],
    ) -> dict[ModelTypeShape, list[str]]:
        """
        Execute a forward pass and store the order of execution of specific sub-modules.

        - `module`: The module to trace.
        - `args`: The arguments to pass to the module it can be either a tuple of positional arguments,
            a dictionary of keyword arguments, or a dictionary with `positional` and `keyword` keys.
        - `keys_to_skip`: A list of keys to skip when tracing the module.

        ### Returns:
        - A dictionary mapping the signature of each sub-module to a list of keys in the module's `named_modules`
        """
        submodule_to_key: dict[nn.Module, str] = {}
        execution_order: defaultdict[ModelTypeShape, list[str]] = defaultdict(list)

        def collect_execution_order_hook(layer: nn.Module, *_: Any) -> None:
            layer_signature = self.get_module_signature(module=layer)
            execution_order[layer_signature].append(submodule_to_key[layer])

        hooks: list[RemovableHandle] = []
        named_modules: list[tuple[str, nn.Module]] = module.named_modules()  # type: ignore
        for name, submodule in named_modules:
            if (self._infer_basic_layer_type(module=submodule) is not None) and name not in keys_to_skip:
                submodule_to_key[submodule] = name  # type: ignore
                hook = submodule.register_forward_hook(hook=collect_execution_order_hook)
                hooks.append(hook)

        positional_args, keyword_args = self._unpack_module_args(module_args=args)
        module(*positional_args, **keyword_args)

        for hook in hooks:
            hook.remove()

        return dict(execution_order)

    def _assert_shapes_aligned(
        self, source_order: dict[ModelTypeShape, list[str]], target_order: dict[ModelTypeShape, list[str]]
    ) -> bool:
        """Assert that the shapes of the sub-modules in `source_order` and `target_order` are aligned."""
        model_type_shapes = set(source_order.keys()) | set(target_order.keys())
        shape_missmatched = False

        for model_type_shape in model_type_shapes:
            source_keys = source_order.get(model_type_shape, [])
            target_keys = target_order.get(model_type_shape, [])

            if len(source_keys) != len(target_keys):
                shape_missmatched = True
                self._debug_print_shapes(shape=model_type_shape, source_keys=source_keys, target_keys=target_keys)

        return not shape_missmatched

    @staticmethod
    def _convert_state_dict(
        source_state_dict: dict[str, Tensor], target_state_dict: dict[str, Tensor], state_dict_mapping: dict[str, str]
    ) -> dict[str, Tensor]:
        """Convert the source model's state_dict to match the target model's state_dict."""
        converted_state_dict: dict[str, Tensor] = {}
        for target_key in target_state_dict:
            target_prefix, suffix = target_key.rsplit(sep=".", maxsplit=1)
            source_prefix = state_dict_mapping[target_prefix]
            source_key = ".".join([source_prefix, suffix])
            converted_state_dict[target_key] = source_state_dict[source_key]

        return converted_state_dict

    @torch.no_grad()
    def _collect_layers_outputs(
        self, module: nn.Module, args: ModuleArgs, keys_to_skip: list[str]
    ) -> list[tuple[str, Tensor]]:
        """
        Execute a forward pass and store the output of specific sub-modules.

        - `module`: The module to trace.
        - `args`: The arguments to pass to the module it can be either a tuple of positional arguments,
            a dictionary of keyword arguments, or a dictionary with `positional` and `keyword` keys.
        - `keys_to_skip`: A list of keys to skip when tracing the module.

        ### Returns:
        - A list of tuples containing the key of each sub-module and its output.

        ### Note:
        - The output of each sub-module is cloned to avoid memory leaks.
        """
        submodule_to_key: dict[nn.Module, str] = {}
        execution_order: list[tuple[str, Tensor]] = []

        def collect_execution_order_hook(layer: nn.Module, _: Any, output: Tensor) -> None:
            execution_order.append((submodule_to_key[layer], output.clone()))

        hooks: list[RemovableHandle] = []
        named_modules: list[tuple[str, nn.Module]] = module.named_modules()  # type: ignore
        for name, submodule in named_modules:
            if (self._infer_basic_layer_type(module=submodule) is not None) and name not in keys_to_skip:
                submodule_to_key[submodule] = name  # type: ignore
                hook = submodule.register_forward_hook(hook=collect_execution_order_hook)
                hooks.append(hook)

        positional_args, keyword_args = self._unpack_module_args(module_args=args)
        module(*positional_args, **keyword_args)

        for hook in hooks:
            hook.remove()

        return execution_order
