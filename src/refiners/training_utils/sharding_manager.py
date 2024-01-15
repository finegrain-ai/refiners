from .config import ModelConfig, TrainingConfig
from torch.nn import Module
from torch import Tensor, device as Device
from torch.autograd import backward
from abc import ABC, abstractmethod
from functools import cached_property, partial, update_wrapper

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
        
        if(hasattr(module, "_tensor_methods") is False):
            method_list = ["forward"]
        else:
            method_list = module._tensor_methods
        
        for method_name in method_list:
            module = self.add_execution_hook(module, device, method_name)
        return module
    
    def add_execution_hook(self, module: Module, device: Device, method_name: str) -> None:
        
        old_method = getattr(module, method_name)
        
        def new_method(module, *args, **kwargs):
            args = [arg.to(device) if hasattr(arg, "to") else arg for arg in args]
            kwargs = {k: v.to(device) if hasattr(v, "to") else v for k, v in kwargs.items()}
            output = old_method(*args, **kwargs)
            return output
        
        new_method = update_wrapper(
            partial(new_method, module),
            old_method
        )
        
        setattr(module, method_name, new_method)
        return module

    @property
    def device(self) -> Device:
        return self.default_device
