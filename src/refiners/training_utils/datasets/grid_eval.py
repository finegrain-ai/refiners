from src.refiners.training_utils.datasets.latent_diffusion import BatchType
from src.refiners.training_utils.trainer import Batch
from torch.utils.data import Dataset

BatchType = TypeVar("BatchType", bound=Any)

class GridEvalDataset(Dataset[Dict[str, Any]]):
    
    BatchType = BatchType
    
    def __init__(self,
		config: HuggingfaceDatasetConfig,
		db_indexes: list[int],
  		prompts: list[str],
    ):
		self.db_indexes = db_indexes
		self.prompts = prompts
	
    def __len__(self):
        return len(self.db_indexes) * len(self.prompts)
    
    def __getitem__(self, index: int) -> BatchType:
		db_index = self.db_indexes[index // len(self.prompts)]
		prompt = self.prompts[index % len(self.prompts)]
		return db_index, prompt

	
	def collate_fn(self, batch: list[BatchType]) -> BatchType:
		
		return batch