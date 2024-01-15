from lightning.fabric import Fabric
from .trainer import Trainer 
from functools import cached_property
from torch.optim import Optimizer
from torch import Tensor, cuda
from typing import Sequence
from refiners.training_utils.config import TrainingConfig, BaseConfig
from typing import Generic, TypeVar, Any
from refiners.training_utils.callback import Callback
from torch import device as Device
from loguru import logger

class FabricTrainingConfig(TrainingConfig):
    devices: int = 1

class FabricBaseConfig(BaseConfig):
    training: FabricTrainingConfig


Batch = TypeVar("Batch")
ConfigType = TypeVar("ConfigType", bound=FabricBaseConfig)


class FabricTrainer(Trainer, Generic[ConfigType, Batch]):
    fabric_optimizer: Optimizer = None
    
    def __init__(self, config: ConfigType, callbacks: list[Callback[Any]] | None = None) -> None:        
        self.fabric = Fabric(strategy="fsdp")
        self.fabric.launch()
        self.fabric.seed_everything(42)

        super().__init__(config, callbacks)
        
    
    def _backward(self, tensors: Tensor | Sequence[Tensor]):
        
        # Check if the input is a single tensor
        if isinstance(tensors, Tensor):
            tensors = [tensors]  # Wrap the tensor in a list       
         
        for tensor in tensors:
            self.fabric.backward(tensor)
    
    def __str__(self) -> str:
        return f"Trainer : \n"+ "\n".join([f"* {self.models[model_name]}:{self.models[model_name].device}" for model_name in self.models])
    
    @cached_property
    def optimizer(self) -> Optimizer:
        optimizer = super().optimizer
        return fabric.setup_optimizers(optimizer)

    def prepare_model(self, model_name: str) -> None:
        self.fabric.print(model_name, cuda.memory_summary())

        model = self.fabric.setup(self.models[model_name])
        # self.fabric_optimizer = optimizer
        self.models[model_name] = model  
        
        if (checkpoint := self.config.models[model_name].checkpoint) is not None:
            model.load_from_safetensors(tensors_path=checkpoint)
        else:
            logger.info(f"No checkpoint found. Initializing model `{model_name}` from scratch.")
        model.requires_grad_(requires_grad=self.config.models[model_name].train)
        model.zero_grad()
        

    
    @cached_property
    def device(self) -> Device:
        raise NotImplementedError("FabricTrainer does not support this property")