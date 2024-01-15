from .config import ModelConfig, TrainingConfig
from torch.nn import Module
from torch import Tensor, device as Device
from torch.autograd import backward
from abc import ABC, abstractmethod
from functools import cached_property, partial, update_wrapper
from typing import Any, List, Callable

class ShardingManager(ABC):
    @abstractmethod
    def backward(self, tensor: Tensor) -> None:
        ...

    @abstractmethod
    def setup_model(self, model: Module, config: ModelConfig):
        ...

    @property
    @abstractmethod
    def device(self) -> Device:
        raise NotImplementedError("FabricTrainer does not support this property")

from refiners.fluxion.context import ContextProvider

class SimpleShardingManager(ShardingManager):
    def __init__(self, config: TrainingConfig) -> None:
        self.default_device = config.gpu_index if config.gpu_index is not None else "cpu"

    def backward(self, tensor: Tensor):
        backward(tensor)

    def setup_model(self, model: Module, config: ModelConfig) -> Module:
        if config.gpu_index is not None:
            device = f"cuda:{config.gpu_index}"
        else:
            device = self.default_device
        model = model.to(device=device)
        model = self.add_execution_hooks(model, device)
        return model

    # inspired from https://github.com/huggingface/accelerate/blob/6f05bbd41a179cc9a86238c7c6f3f4eded70fbd8/src/accelerate/hooks.py#L159C1-L170C18
    def add_execution_hooks(self, module: Module, device: Device) -> None:
        method_list = []
        if hasattr(module, "forward") is True:
            method_list.append("forward")
        
        if hasattr(module, "set_context") is True:
            method_list.append("set_context")
        
        if hasattr(module, "encode") is True:
            method_list.append("encode")
        
        if hasattr(module, "decode") is True:
            method_list.append("decode")  
        
        for method_name in method_list:
            self.add_execution_hook(module, device, method_name)

    
    def recursive_to(self, obj: Any, device: Device) -> Any:
        if hasattr(obj, "to"):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: self.recursive_to(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.recursive_to(v, device) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.recursive_to(v, device) for v in obj)
        else:
            return obj
        
    def add_execution_hook(self, module: Module, device: Device, method_name: str) -> None:
        
        old_method = getattr(module, method_name)

        new_method = self.bind_input_to_device(old_method, device)
        # new_method = update_wrapper(partial(new_method, module), old_method)
        
        def new_method(module, *args, **kwargs):
            args = self.recursive_to(args, device)
            kwargs = self.recursive_to(kwargs, device)
            output = old_method(*args, **kwargs)
            return output

        new_method = update_wrapper(partial(new_method, module), old_method)

        setattr(module, method_name, new_method)
    
    def bind_input_to_device(self, method: Callable, device: Device) -> Callable:
        def new_method(*args, **kwargs):
            args = self.recursive_to(args, device)
            kwargs = self.recursive_to(kwargs, device)
            return method(*args, **kwargs)
        
        return new_method
        
    
    @property
    def device(self) -> Device:
        return self.default_device
