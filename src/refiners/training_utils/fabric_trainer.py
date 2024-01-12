from .trainer import Trainer 

class FabricTrainer(Trainer):
    @cached_property
    def optimizer(self) -> Optimizer:
        optimizer = super().optimizer
        for model_name in self.models:
            model, optimizer = fabric.setup(self.models[model_name], optimizer)
            self.models[model_name] = model
        return optimizer
    
    def _backward(self, tensors: torch.Tensor | List[torch.Tensor]):
        
        # Check if the input is a single tensor
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = [input_tensor]  # Wrap the tensor in a list       
         
        for tensor in tensors:
            fabric.backward(tensor)
