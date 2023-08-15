from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Literal, TypeVar
from PIL import Image
from numpy import array, float32
from pathlib import Path
from safetensors import safe_open as _safe_open  # type: ignore
from safetensors.torch import save_file as _save_file  # type: ignore
from torch import norm as _norm, manual_seed as _manual_seed  # type: ignore
from torch.nn.functional import pad as _pad, interpolate as _interpolate  # type: ignore
from torch import Size, Tensor, tensor, no_grad, device as Device, dtype as DType, nn
from torch.utils.hooks import RemovableHandle

if TYPE_CHECKING:
    from refiners.fluxion.layers.module import Module


T = TypeVar("T")
E = TypeVar("E")


def norm(x: Tensor) -> Tensor:
    return _norm(x)  # type: ignore


def manual_seed(seed: int) -> None:
    _manual_seed(seed)


def pad(x: Tensor, pad: Iterable[int], value: float = 0.0) -> Tensor:
    return _pad(input=x, pad=pad, value=value)  # type: ignore


def interpolate(x: Tensor, factor: float | Size, mode: str = "nearest") -> Tensor:
    return (
        _interpolate(x, scale_factor=factor, mode=mode)
        if isinstance(factor, float | int)
        else _interpolate(x, size=factor, mode=mode)
    )  # type: ignore


def bidirectional_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
    return {**mapping, **{value: key for key, value in mapping.items()}}


def image_to_tensor(image: Image.Image, device: Device | str | None = None, dtype: DType | None = None) -> Tensor:
    return tensor(array(image).astype(float32).transpose(2, 0, 1) / 255.0, device=device, dtype=dtype).unsqueeze(0)


def tensor_to_image(tensor: Tensor) -> Image.Image:
    return Image.fromarray((tensor.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))  # type: ignore


def safe_open(
    path: Path | str,
    framework: Literal["pytorch", "tensorflow", "flax", "numpy"],
    device: Device | str = "cpu",
) -> dict[str, Tensor]:
    framework_mapping = {
        "pytorch": "pt",
        "tensorflow": "tf",
        "flax": "flax",
        "numpy": "numpy",
    }
    return _safe_open(str(path), framework=framework_mapping[framework], device=str(device))  # type: ignore


def load_from_safetensors(path: Path | str, device: Device | str = "cpu") -> dict[str, Tensor]:
    with safe_open(path=path, framework="pytorch", device=device) as tensors:  # type: ignore
        return {key: tensors.get_tensor(key) for key in tensors.keys()}  # type: ignore


def load_metadata_from_safetensors(path: Path | str) -> dict[str, str] | None:
    with safe_open(path=path, framework="pytorch") as tensors:  # type: ignore
        return tensors.metadata()  # type: ignore


def save_to_safetensors(path: Path | str, tensors: dict[str, Tensor], metadata: dict[str, str] | None = None) -> None:
    _save_file(tensors, path, metadata)  # type: ignore


class BasicLayers(Enum):
    Conv1d = nn.Conv1d
    Conv2d = nn.Conv2d
    Conv3d = nn.Conv3d
    ConvTranspose1d = nn.ConvTranspose1d
    ConvTranspose2d = nn.ConvTranspose2d
    ConvTranspose3d = nn.ConvTranspose3d
    Linear = nn.Linear
    BatchNorm1d = nn.BatchNorm1d
    BatchNorm2d = nn.BatchNorm2d
    BatchNorm3d = nn.BatchNorm3d
    LayerNorm = nn.LayerNorm
    GroupNorm = nn.GroupNorm
    Embedding = nn.Embedding
    MaxPool2d = nn.MaxPool2d
    AvgPool2d = nn.AvgPool2d
    AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d


ModelTypeShape = tuple[str, tuple[Size, ...]]


def infer_basic_layer_type(module: nn.Module) -> BasicLayers | None:
    """Identify if the provided module matches any in the BasicLayers enum."""
    for layer_type in BasicLayers:
        if isinstance(module, layer_type.value):
            return layer_type
    return None


def get_module_signature(module: "Module") -> ModelTypeShape:
    """Return a tuple representing the module's type and parameter shapes."""
    layer_type = infer_basic_layer_type(module=module)
    assert layer_type is not None, f"Module {module} is not a basic layer"
    param_shapes = [p.shape for p in module.parameters()]
    return (str(object=layer_type), tuple(param_shapes))


def forward_order_of_execution(
    module: "Module",
    example_args: tuple[Any, ...],
    key_skipper: Callable[[str], bool] | None = None,
) -> dict[ModelTypeShape, list[str]]:
    """
    Determine the execution order of sub-modules during a forward pass.

    Optionally skips specific modules using `key_skipper`.
    """
    key_skipper = key_skipper or (lambda _: False)

    submodule_to_key: dict["Module", str] = {}
    execution_order: defaultdict[ModelTypeShape, list[str]] = defaultdict(list)

    def collect_execution_order_hook(layer: "Module", *_: Any) -> None:
        layer_signature = get_module_signature(module=layer)
        execution_order[layer_signature].append(submodule_to_key[layer])

    hooks: list[RemovableHandle] = []
    for name, submodule in module.named_modules():
        if (infer_basic_layer_type(module=submodule) is not None) and not key_skipper(name):
            submodule_to_key[submodule] = name
            hook = submodule.register_forward_hook(hook=collect_execution_order_hook)
            hooks.append(hook)

    with no_grad():
        module(*example_args)

    for hook in hooks:
        hook.remove()

    return dict(execution_order)


def print_side_by_side(
    shape: ModelTypeShape,
    source_keys: list[str],
    target_keys: list[str],
) -> None:
    """Print module keys side by side, useful for debugging shape mismatches."""
    print(f"{shape}")
    max_len = max(len(source_keys), len(target_keys))
    for i in range(max_len):
        source_key = source_keys[i] if i < len(source_keys) else "---"
        target_key = target_keys[i] if i < len(target_keys) else "---"
        print(f"\t{source_key}\t{target_key}")


def verify_shape_match(
    source_order: dict[ModelTypeShape, list[str]], target_order: dict[ModelTypeShape, list[str]]
) -> bool:
    """Check if the sub-modules in source and target have matching shapes."""
    model_type_shapes = set(source_order.keys()) | set(target_order.keys())
    shape_missmatched = False

    for model_type_shape in model_type_shapes:
        source_keys = source_order.get(model_type_shape, [])
        target_keys = target_order.get(model_type_shape, [])

        if len(source_keys) != len(target_keys):
            shape_missmatched = True
            print_side_by_side(shape=model_type_shape, source_keys=source_keys, target_keys=target_keys)

    return not shape_missmatched


def create_state_dict_mapping(
    source_model: "Module",
    target_model: "Module",
    source_args: tuple[Any, ...],
    target_args: tuple[Any, ...] | None = None,
    source_key_skipper: Callable[[str], bool] | None = None,
    target_key_skipper: Callable[[str], bool] | None = None,
) -> dict[str, str] | None:
    """
    Create a mapping between state_dict keys of the source and target models.

    This facilitates the transfer of weights when architectures have slight differences.
    """
    if target_args is None:
        target_args = source_args

    source_order = forward_order_of_execution(
        module=source_model, example_args=source_args, key_skipper=source_key_skipper
    )
    target_order = forward_order_of_execution(
        module=target_model, example_args=target_args, key_skipper=target_key_skipper
    )

    if not verify_shape_match(source_order=source_order, target_order=target_order):
        return None

    mapping: dict[str, str] = {}
    for model_type_shape in source_order:
        source_keys = source_order[model_type_shape]
        target_keys = target_order[model_type_shape]
        mapping.update(zip(target_keys, source_keys))

    return mapping


def convert_state_dict(
    source_state_dict: dict[str, Tensor], target_state_dict: dict[str, Tensor], state_dict_mapping: dict[str, str]
) -> dict[str, Tensor]:
    """Convert source state_dict based on the provided mapping to match target state_dict structure."""
    converted_state_dict: dict[str, Tensor] = {}
    for target_key in target_state_dict:
        target_prefix, suffix = target_key.rsplit(sep=".", maxsplit=1)
        source_prefix = state_dict_mapping[target_prefix]
        source_key = ".".join([source_prefix, suffix])
        converted_state_dict[target_key] = source_state_dict[source_key]

    return converted_state_dict


def forward_store_outputs(
    module: "Module",
    example_args: tuple[Any, ...],
    key_skipper: Callable[[str], bool] | None = None,
) -> list[tuple[str, Tensor]]:
    """Execute a forward pass and store outputs of specific sub-modules."""
    key_skipper = key_skipper or (lambda _: False)
    submodule_to_key: dict["Module", str] = {}
    execution_order: list[tuple[str, Tensor]] = []  # Store outputs in a list

    def collect_execution_order_hook(layer: "Module", _: Any, output: Tensor) -> None:
        execution_order.append((submodule_to_key[layer], output.clone()))  # Store a copy of the output

    hooks: list[RemovableHandle] = []
    for name, submodule in module.named_modules():
        if (infer_basic_layer_type(module=module) is not None) and not key_skipper(name):
            submodule_to_key[submodule] = name
            hook = submodule.register_forward_hook(hook=collect_execution_order_hook)
            hooks.append(hook)

    with no_grad():
        module(*example_args)

    for hook in hooks:
        hook.remove()

    return execution_order


def compare_models(
    source_model: "Module",
    target_model: "Module",
    source_args: tuple[Any, ...],
    target_args: tuple[Any, ...] | None = None,
    source_key_skipper: Callable[[str], bool] | None = None,
    target_key_skipper: Callable[[str], bool] | None = None,
    threshold: float = 1e-5,
) -> bool:
    """
    Compare the outputs of two models given the same inputs.

    Flag if any difference exceeds the given threshold.
    """
    if target_args is None:
        target_args = source_args

    source_order = forward_store_outputs(module=source_model, example_args=source_args, key_skipper=source_key_skipper)
    target_order = forward_store_outputs(module=target_model, example_args=target_args, key_skipper=target_key_skipper)

    prev_source_key, prev_target_key = None, None
    for (source_key, source_output), (target_key, target_output) in zip(source_order, target_order):
        diff = norm(source_output - target_output).item()
        if diff > threshold:
            print(
                f"Models diverged between {prev_source_key} and {source_key}, and between {prev_target_key} and"
                f" {target_key}, difference in norm: {diff}"
            )
            return False
        prev_source_key, prev_target_key = source_key, target_key

    return True
