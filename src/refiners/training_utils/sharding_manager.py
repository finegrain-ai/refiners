from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

from torch import Tensor, bfloat16, device as Device, dtype as DType, float16, float32
from torch.autograd import backward
from torch.nn import Module

from .config import ModelConfig, TrainingConfig

WrappableMethod = Callable[..., Any]


class ShardingManager(ABC):
    @abstractmethod
    def backward(self, tensor: Tensor) -> None:
        ...

    @abstractmethod
    def setup_model(self, model: Module, config: ModelConfig) -> None:
        ...

    @abstractmethod
    def wrap_device(self, method: WrappableMethod, device: Device) -> WrappableMethod:
        ...

    @abstractmethod
    def add_device_hook(self, module: Module, device: Device, method_name: str) -> None:
        ...

    @abstractmethod
    def add_device_hooks(self, module: Module, device: Device) -> None:
        ...

    @property
    @abstractmethod
    def device(self) -> Device:
        ...

    @property
    @abstractmethod
    def dtype(self) -> DType:
        ...


def str_to_dtype(dtype_str: str) -> DType:
    if dtype_str == "float32":
        return float32
    elif dtype_str == "float16":
        return float16
    elif dtype_str == "bfloat16":
        return bfloat16
    else:
        raise ValueError(f"Unknown dtype {dtype_str}")


class SimpleShardingManager(ShardingManager):
    def __init__(self, config: TrainingConfig) -> None:
        device_str = config.gpu_index if config.gpu_index >= 0 else "cpu"
        self.default_device = Device(device_str)
        self.default_dtype = str_to_dtype(config.dtype)

    def backward(self, tensor: Tensor):
        backward(tensor)

    def setup_model(self, model: Module, config: ModelConfig) -> None:
        if config.gpu_index is not None:
            device = Device(f"cuda:{config.gpu_index}")
        else:
            device = self.default_device

        if config.dtype is not None:
            dtype = str_to_dtype(config.dtype)
        else:
            dtype = self.default_dtype

        model = model.to(device=device, dtype=dtype)
        self.add_device_hooks(model, device)

    # inspired from https://github.com/huggingface/accelerate/blob/6f05bbd41a179cc9a86238c7c6f3f4eded70fbd8/src/accelerate/hooks.py#L159C1-L170C18
    def add_device_hooks(self, module: Module, device: Device) -> None:
        method_list: List[str] = []
        if hasattr(module, "forward") is True:
            method_list.append("forward")

        if hasattr(module, "set_context") is True:
            method_list.append("set_context")

        if hasattr(module, "encode") is True:
            method_list.append("encode")

        if hasattr(module, "decode") is True:
            method_list.append("decode")

        if hasattr(module, "add_noise") is True:
            method_list.append("add_noise")
            
        for method_name in method_list:
            self.add_device_hook(module, device, method_name)

    def recursive_to(self, obj: Any, device: Device) -> Any:
        if hasattr(obj, "to"):
            return obj.to(device)
        elif isinstance(obj, dict):  # type: ignore
            return {k: self.recursive_to(v, device) for k, v in obj.items()}  # type: ignore
        elif isinstance(obj, list):  # type: ignore
            return [self.recursive_to(v, device) for v in obj]  # type: ignore
        elif isinstance(obj, tuple):  # type: ignore
            return tuple(self.recursive_to(v, device) for v in obj)  # type: ignore
        else:
            return obj

    def add_device_hook(self, module: Module, device: Device, method_name: str) -> None:
        old_method = getattr(module, method_name)
        new_method = self.wrap_device(old_method, device)
        setattr(module, method_name, new_method)

    def wrap_device(self, method: WrappableMethod, device: Device) -> WrappableMethod:
        def new_method(*args: List[Any], **kwargs: Dict[Any, Any]) -> Any:
            args = self.recursive_to(args, device)
            kwargs = self.recursive_to(kwargs, device)
            return method(*args, **kwargs)

        return new_method

    @property
    def device(self) -> Device:
        return self.default_device

    @property
    def dtype(self) -> DType:
        return self.default_dtype
