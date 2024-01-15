from accelerate import Accelerator
from .trainer import Trainer 
from functools import cached_property
from torch.optim import Optimizer
from torch import Tensor, cuda
from typing import Sequence
from refiners.training_utils.config import BaseConfig
from typing import Generic, TypeVar, Any
from refiners.training_utils.callback import Callback
from torch import device as Device
from loguru import logger
from accelerate import Accelerator

Batch = TypeVar("Batch")
ConfigType = TypeVar("ConfigType", bound=BaseConfig)


class AccelerateTrainer(Trainer, Generic[ConfigType, Batch]):    
    def __init__(self, config: ConfigType, callbacks: list[Callback[Any]] | None = None) -> None:        
        self.accelerator = Accelerator()
        print(self.accelerator.distributed_type)
        super().__init__(config, callbacks)
        
    
    def _backward(self, tensors: Tensor | Sequence[Tensor]):
        
        # Check if the input is a single tensor
        if isinstance(tensors, Tensor):
            tensors = [tensors]  # Wrap the tensor in a list       
         
        for tensor in tensors:
            self.accelerator.backward(tensor)
    
    def __str__(self) -> str:
        return f"Trainer : \n"+ "\n".join([f"* {self.models[model_name]}:{self.models[model_name].device}" for model_name in self.models])
    
    @cached_property
    def optimizer(self) -> Optimizer:
        optimizer = super().optimizer
        return self.accelerator.prepare(optimizer)

    def setup_model(self, model, **kwargs) -> None:
        out_model = self.accelerator.prepare(model)
        return out_model

    @property
    def device(self) -> Device:
        return self.accelerator.device