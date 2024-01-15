from abc import ABC, abstractmethod
from functools import partial, update_wrapper
from typing import Any, Callable, Dict, List

from torch import Tensor, device as Device
from torch.autograd import backward
from torch.nn import Module

from refiners.foundationals.latent_diffusion.schedulers import Scheduler

from .config import ModelConfig, TrainingConfig

Hookable = Module | Scheduler
WrappableMethod = Callable[..., Any]


class ShardingManager(ABC):
    @abstractmethod
    def backward(self, tensor: Tensor) -> None:
        ...

    @abstractmethod
    def setup_model(self, model: Hookable, config: ModelConfig) -> None:
        ...

    @abstractmethod
    def wrap_device(self, method: WrappableMethod, device: Device) -> WrappableMethod:
        ...

    @abstractmethod
    def add_device_hook(self, module: Hookable, device: Device, method_name: str) -> None:
        ...

    @abstractmethod
    def add_device_hooks(self, module: Hookable, device: Device) -> None:
        ...

    @property
    @abstractmethod
    def device(self) -> Device:
        raise NotImplementedError("FabricTrainer does not support this property")


class SimpleShardingManager(ShardingManager):
    def __init__(self, config: TrainingConfig) -> None:
        device_str = config.gpu_index if config.gpu_index >= 0 else "cpu"
        self.default_device = Device(device_str)

    def backward(self, tensor: Tensor):
        backward(tensor)

    def setup_model(self, model: Hookable, config: ModelConfig) -> None:
        if config.gpu_index is not None:
            device = Device(f"cuda:{config.gpu_index}")
        else:
            device = self.default_device
        model = model.to(device=device)
        self.add_device_hooks(model, device)

    # inspired from https://github.com/huggingface/accelerate/blob/6f05bbd41a179cc9a86238c7c6f3f4eded70fbd8/src/accelerate/hooks.py#L159C1-L170C18
    def add_device_hooks(self, module: Hookable, device: Device) -> None:
        method_list: List[str] = []
        if hasattr(module, "forward") is True:
            method_list.append("forward")

        if hasattr(module, "set_context") is True:
            method_list.append("set_context")

        if hasattr(module, "encode") is True:
            method_list.append("encode")

        if hasattr(module, "decode") is True:
            method_list.append("decode")

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

    def add_device_hook(self, module: Hookable, device: Device, method_name: str) -> None:
        old_method = getattr(module, method_name)

        new_method = self.wrap_device(old_method, device)
        # new_method = update_wrapper(partial(new_method, module), old_method)

        new_method = self.wrap_device(old_method, device)

        new_method = update_wrapper(partial(new_method, module), old_method)

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
