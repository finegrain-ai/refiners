from functools import cached_property
from typing import Generic, TypeVar, Type, Any

from loguru import logger
from refiners.training_utils.wandb import WandbLoggable
from refiners.training_utils.metrics.color_palette import AbstractColorPrompt, AbstractColorResults
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
from torch import Tensor, randn, tensor

from torch.utils.data import DataLoader

from refiners.fluxion.adapters.histogram import (
    HistogramExtractor
)

from refiners.foundationals.latent_diffusion import (
    DPMSolver,
    StableDiffusion_1,
)

from refiners.training_utils.datasets.color_palette import ColorDatasetConfig, ColorPaletteDataset, TextEmbeddingColorPaletteLatentsBatch
from refiners.training_utils.trainers.latent_diffusion import (
    FinetuneLatentDiffusionBaseConfig,
    LatentDiffusionBaseTrainer,
    TestDiffusionBaseConfig,
)
from refiners.training_utils.datasets.color_palette import ColorPaletteDataset
from refiners.training_utils.trainers.trainer import scoped_seed
# def hash_tensor(image: Tensor) -> str:
#     str2 = ""
#     for i in range(image.shape[0]):
#         local_str = f"{image[i].sum()}-{image[i].mean()}-{image[i].std()}-{image[i].max()}-{image[i].min()}"
#         str2 += local_str
    
#     return str(hash(str2))

from torch.utils.data import Dataset

class ColorTrainerEvaluationConfig(TestDiffusionBaseConfig):
    db_indexes: list[int]
    batch_size: int = 1

class ColorTrainerConfig(FinetuneLatentDiffusionBaseConfig):
    evaluation: ColorTrainerEvaluationConfig
    dataset: ColorDatasetConfig # type: ignore
    eval_dataset: ColorDatasetConfig

PromptType = TypeVar("PromptType", bound=AbstractColorPrompt)
ResultType = TypeVar("ResultType", bound=AbstractColorResults) 
ConfigType = TypeVar("ConfigType", bound = ColorTrainerConfig)

class GridEvalDataset(Generic[PromptType], Dataset[PromptType]):
    
    __prompt_type__ : Type[PromptType]
    
    def __init__(self, db_indexes: list[int], hf_dataset: ColorPaletteDataset, source_prompts: list[str], text_encoder: CLIPTextEncoderL):
        self.db_indexes = db_indexes
        self.hf_dataset = hf_dataset
        self.source_prompts = source_prompts
        self.text_encoder = text_encoder
        self.text_embeddings : list[Tensor] = [self.text_encoder(prompt) for prompt in source_prompts]

    def __len__(self):
        return len(self.db_indexes) * len(self.source_prompts)

    def __getitem__(self, index: int) -> PromptType:
        db_index = self.db_indexes[index // len(self.source_prompts)]
        source_prompt = self.source_prompts[index % len(self.source_prompts)]
        batch = self.hf_dataset[db_index]
        args = self.process_item(batch)
        return self.__class__.__prompt_type__(
            db_indexes=[db_index], 
            source_prompts=[source_prompt],
            source_images=[batch[0].image],
            text_embeddings=self.text_embeddings[index % len(self.source_prompts)],
            **args
        )
        
    def process_item(self, items: TextEmbeddingColorPaletteLatentsBatch) -> dict[str, Any]:
        ...

class AbstractColorTrainer(
    Generic[PromptType, ResultType, ConfigType],
    LatentDiffusionBaseTrainer[ConfigType, TextEmbeddingColorPaletteLatentsBatch],
):
    def load_dataset(self) -> ColorPaletteDataset:
        return ColorPaletteDataset(
            config=self.config.dataset
        )

    @cached_property
    def dataset(self) -> ColorPaletteDataset:  # type: ignore
        return self.load_dataset() 
    @cached_property
    def sd(self) -> StableDiffusion_1:
        solver = DPMSolver(
            device=self.device, num_inference_steps=self.config.evaluation.num_inference_steps, dtype=self.dtype
        )

        self.sharding_manager.add_device_hooks(solver, solver.device)
        return StableDiffusion_1(
            unet=self.unet, lda=self.lda, clip_text_encoder=self.text_encoder, solver=solver)
    
    # @cached_property
    # def eval_dataset(self) -> list[tuple[str, Tensor]]:
    #     return [(prompt, self.text_encoder(prompt)) for prompt in self.config.evaluation.prompts]
    
    @cached_property
    def eval_dataloader(self) -> DataLoader[PromptType]:
                        
        return DataLoader(
            dataset=self.grid_eval_dataset, 
            batch_size=self.config.evaluation.batch_size, 
            shuffle=False,
            collate_fn=self.collate_prompts, 
            num_workers=self.config.training.num_workers
        )
    
    @cached_property
    def unconditionnal_text_embedding(self) -> Tensor:
        return self.text_encoder([""])
   
    @scoped_seed(5)
    def compute_batch_evaluation(self, batch: PromptType, same_seed: bool = True) -> ResultType:
        batch_size = len(batch.source_prompts)
        
        logger.info(f"Generating {batch_size} images for prompts/db_indexes: {batch.source_prompts}/{batch.db_indexes}")
        
        if same_seed:
            x = randn(1, 4, 64, 64, dtype=self.dtype, device=self.device)
            x = x.repeat(batch_size, 1, 1, 1)
        else: 
            x = randn(batch_size, 4, 64, 64, dtype=self.dtype, device=self.device)

        self.eval_set_adapter_values(batch)
        
        clip_text_embedding = self.sd.compute_clip_text_embedding(text=batch.source_prompts)
        
        for step in self.sd.steps:
            x = self.sd(
                x,
                step=step,
                clip_text_embedding=clip_text_embedding,
                condition_scale = self.config.evaluation.condition_scale
            )

        images = (self.sd.lda.decode(x) + 1 )/2
        # for i in range(batch_size):
        #     logger.info(f"eval_images/[{i}] {batch.source_prompts[i]}_{batch.db_indexes[i]} : "+
        #                 f"img hash : {hash_tensor(images[i])},"+
        #                 f"txt_hash: {clip_text_embedding.norm()},"+
        #                 f"histo_hash: {cfg_histogram_embedding.norm()}")
        return self.build_results(batch, images)
    
    def build_results(self, batch: PromptType, result_images: Tensor) -> ResultType:
        ...
    
    def image_distances(self, batch: ResultType) -> dict[str, float]:
        images = batch.result_images
        dist = tensor(0)
        for i in range(images.shape[0]):
            for j in range(i+1, images.shape[0]):
                dist = dist + self.mse_loss(images[i], images[j])
        
        return dist.item()
    
    def collate_results(self, batch: list[ResultType]) -> ResultType:
        ...
        
    def collate_prompts(self, batch: list[PromptType]) -> PromptType:
        ...
        
    def empty(self) -> ResultType:
        ...
    
    def compute_evaluation(
        self
    ) -> None:
        
        per_prompts : dict[str, ResultType] = {}
        images : dict[str, WandbLoggable] = {}
        
        all_results : ResultType = self.empty()
        
        
        for batch in self.eval_dataloader:
            results = self.compute_batch_evaluation(batch)
        
            for prompt in list(set(results.source_prompts)):
                batch = results.get_prompt(prompt)
                if prompt not in per_prompts:
                    per_prompts[prompt] = batch
                else:
                    per_prompts[prompt] = self.collate_results([
                        per_prompts[prompt],
                        batch
                    ])
        
        for prompt in per_prompts:
            self.log(data={f"inter_prompt_distance/{prompt}": self.image_distances(per_prompts[prompt])})
            image = self.draw_cover_image(per_prompts[prompt])
            image_name = f"eval_images/{prompt}"
            images[image_name] = image
            
        all_results = self.collate_results(list(per_prompts.values()))
        
        # images[f"eval_images/all"] = self.draw_cover_image(all_results)
        self.log(data=images)

        self.batch_metrics(all_results, prefix="eval")


    def batch_metrics(
        self, results: ResultType, prefix: str = ""
    ) -> None:
         ...
    
    @cached_property
    def histogram_extractor(self) -> HistogramExtractor:
        return HistogramExtractor(color_bits=self.config.histogram_auto_encoder.color_bits)


    @cached_property
    def grid_eval_dataset(self) -> GridEvalDataset[PromptType]:
        return GridEvalDataset(
            db_indexes=self.config.evaluation.db_indexes,
            hf_dataset=self.eval_dataset,
            source_prompts=self.config.evaluation.prompts,
            text_encoder=self.text_encoder
        )
    
    @cached_property
    def eval_dataset(self) -> ColorPaletteDataset:
        return ColorPaletteDataset(
            config=self.config.eval_dataset
        )
        
