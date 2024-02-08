from functools import cached_property
from typing import Any, TypedDict
from refiners.fluxion.utils import load_from_safetensors, tensor_to_images

from loguru import logger
from PIL import Image
from pydantic import BaseModel
from refiners.training_utils.trainers.abstract_color_trainer import AbstractColorTrainer, ColorTrainerEvaluationConfig
from refiners.training_utils.metrics.color_palette import BatchColorPalettePrompt
from refiners.training_utils.trainers.abstract_color_trainer import GridEvalDataset
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
from refiners.training_utils.datasets.color_palette import ColorPaletteDataset
from torch import Tensor
import numpy as np

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.color_palette import Color, ColorPaletteEncoder, SD1ColorPaletteAdapter, ColorPaletteExtractor
from refiners.fluxion.utils import save_to_safetensors
from refiners.foundationals.latent_diffusion import (
    SD1UNet
)
from refiners.training_utils.callback import Callback
from refiners.training_utils.metrics.color_palette import BatchColorPalettePrompt, BatchColorPaletteResults, ImageAndPalette, batch_image_palette_metrics
from refiners.training_utils.trainers.latent_diffusion import (
    FinetuneLatentDiffusionBaseConfig,
)
from refiners.training_utils.datasets.color_palette import ColorDatasetConfig, ColorPalette, TextEmbeddingColorPaletteLatentsBatch
from refiners.training_utils.callback import GradientNormLayerLogging

class ColorPaletteConfig(BaseModel):
    feedforward_dim: int = 3072
    num_attention_heads: int = 12
    num_layers: int = 12
    embedding_dim: int = 768
    trigger_phrase: str = ""
    use_only_trigger_probability: float = 0.0
    max_colors: int
    mode : str = "transformer"
    weighted_palette: bool = False
    without_caption_probability: float = 0.17

class ColorPalettePromptConfig(BaseModel):
    text: str
    color_palette: ColorPalette

class LatentPrompt(TypedDict):
    text: str
    color_palette_embedding: Tensor

class ColorPaletteLatentDiffusionConfig(FinetuneLatentDiffusionBaseConfig):
    color_palette: ColorPaletteConfig
    evaluation: ColorTrainerEvaluationConfig
    dataset: ColorDatasetConfig
    eval_dataset: ColorDatasetConfig

class GridEvalPaletteDataset(GridEvalDataset[BatchColorPalettePrompt]):
    __prompt_type__ = BatchColorPalettePrompt
    def __init__(self, db_indexes: list[int], hf_dataset: ColorPaletteDataset, source_prompts: list[str], text_encoder: CLIPTextEncoderL, color_palette_extractor: ColorPaletteExtractor):
        super().__init__(db_indexes, hf_dataset, source_prompts, text_encoder)
        self.color_palette_extractor = color_palette_extractor
    def process_item(self, items: TextEmbeddingColorPaletteLatentsBatch) -> dict[str, Any]:
        
        if len(items) != 1:
            raise ValueError("The items must have length 1.")
        
        source_palettes = [self.color_palette_extractor(item.image, size=len(item.color_palette)) for item in items]
        return {
            "source_palettes": source_palettes
        }

class ColorPaletteLatentDiffusionTrainer(AbstractColorTrainer[BatchColorPalettePrompt, BatchColorPaletteResults, ColorPaletteLatentDiffusionConfig]):
    @cached_property
    def color_palette_encoder(self) -> ColorPaletteEncoder:
        assert (
            self.config.models["color_palette_encoder"] is not None
        ), "The config must contain a color_palette_encoder entry."

        encoder = ColorPaletteEncoder(
            max_colors=self.config.color_palette.max_colors,
            embedding_dim=self.config.color_palette.embedding_dim,
            num_layers=self.config.color_palette.num_layers,
            mode=self.config.color_palette.mode,
            weighted_palette=self.config.color_palette.weighted_palette,
            num_attention_heads=self.config.color_palette.num_attention_heads,
            feedforward_dim=self.config.color_palette.feedforward_dim,
            device=self.device,
        )
        return encoder
    @cached_property
    def color_palette_extractor(self) -> ColorPaletteExtractor:
        return ColorPaletteExtractor(
            size=self.config.color_palette.max_colors,
            weighted_palette=self.config.color_palette.weighted_palette
        )
    @cached_property
    def color_palette_adapter(self) -> SD1ColorPaletteAdapter[Any]:
        
        weights : dict[str, Tensor] | None = None
        scale = 1.0
        
        if "color_palette_adapter" in self.config.adapters:
            if checkpoint := self.config.adapters["color_palette_adapter"].checkpoint:
                weights = load_from_safetensors(checkpoint)
            scale = self.config.adapters["color_palette_adapter"].scale
        
        adapter : SD1ColorPaletteAdapter[SD1UNet] = SD1ColorPaletteAdapter(
            target=self.unet,
            weights=weights,
            scale=scale,
            color_palette_encoder=self.color_palette_encoder
        )
        
        if weights is None:
            adapter.zero_init()
        
        return adapter

    def __init__(
        self,
        config: ColorPaletteLatentDiffusionConfig,
        callbacks: "list[Callback[Any]] | None" = None,
    ) -> None:
        super().__init__(config=config, callbacks=callbacks)
        self.callbacks.extend((LoadColorPalette(), SaveColorPalette(), GradientNormLayerLogging()))
    
    def load_models(self) -> dict[str, fl.Module]:
        return {
            "unet": self.unet,
            "text_encoder": self.text_encoder,
            "lda": self.lda,
            "color_palette_encoder": self.color_palette_encoder,
        }
    
    @cached_property
    def grid_eval_dataset(self) -> GridEvalDataset[BatchColorPalettePrompt]:
        return GridEvalPaletteDataset(
            db_indexes=self.config.evaluation.db_indexes,
            hf_dataset=self.eval_dataset,
            source_prompts=self.config.evaluation.prompts,
            text_encoder=self.text_encoder,
            color_palette_extractor=self.color_palette_extractor
        )
    
    # def eval_dataset(self) -> list[BatchColorPalettePrompt]:
    #     dataset = self.dataset
    #     indices = self.config.evaluation.db_indexes
    #     items = [dataset[i][0] for i in indices]
    #     print(f"color palette lenght : {[len(item.color_palette) for item in items]}")
    #     palette = [self.color_palette_extractor(item.image, size=len(item.color_palette)) for item in items]
    #     images = [item.image for item in items]
    #     eval_indices = list(zip(indices, palette, images))
        
    #     evaluations : list[BatchColorPalettePrompt] = []
    #     prompts_list = [(prompt, self.text_encoder(prompt)) for prompt in self.config.evaluation.prompts]

    #     for (prompt, prompt_embedding) in prompts_list:
    #         for db_index, palette, image in eval_indices:
    #             batch_prompt = BatchColorPalettePrompt(
    #                 source_prompts= [prompt],
    #                 db_indexes= [db_index],
    #                 source_palettes= [palette],
    #                 text_embeddings= prompt_embedding,
    #                 source_images= [image]
    #             )
    #             evaluations.append(batch_prompt)
        
    #     print(f"Eval dataset size: {len(evaluations)}")
    #     return evaluations
    
    def build_results(self, batch: BatchColorPalettePrompt, result_images: Tensor) -> BatchColorPaletteResults:
        
        return BatchColorPaletteResults(
            source_prompts=batch.source_prompts,
            db_indexes=batch.db_indexes,
            source_palettes=batch.source_palettes,
            result_images=result_images,
            source_images=batch.source_images,
            result_palettes=[self.color_palette_extractor(image, size=len(batch.source_palettes[i])) for i, image in enumerate(tensor_to_images(result_images))],
            text_embeddings=batch.text_embeddings
        )
    
    def collate_results(self, batch: list[BatchColorPaletteResults]) -> BatchColorPaletteResults:
        return BatchColorPaletteResults.collate_fn(batch)
    
    def empty(self) -> BatchColorPaletteResults:
        return BatchColorPaletteResults.empty()
    
    def collate_prompts(self, batch: list[BatchColorPalettePrompt]) -> BatchColorPalettePrompt:
        return BatchColorPalettePrompt.collate_fn(batch)
    
    def compute_loss(self, batch: TextEmbeddingColorPaletteLatentsBatch) -> Tensor:
        
        texts = [item.text for item in batch]
        text_embeddings = self.text_encoder(texts)
        
        latents = self.lda.images_to_latents([item.image for item in batch])
        color_palettes = [self.color_palette_extractor(item.image, size=len(item.color_palette)) for item in batch]
        
        color_palette_embeddings = self.color_palette_encoder(
            color_palettes
        )

        timestep = self.sample_timestep()
        noise = self.sample_noise(size=latents.shape, dtype=latents.dtype)
        noisy_latents = self.ddpm_solver.add_noise(x=latents, noise=noise, step=self.current_step)
        self.unet.set_timestep(timestep=timestep)

        self.unet.set_clip_text_embedding(clip_text_embedding=text_embeddings)
        self.color_palette_adapter.set_color_palette_embedding(color_palette_embeddings)

        prediction = self.unet(noisy_latents)
        loss = self.mse_loss(prediction, noise)
        
        return loss
    
    def eval_set_adapter_values(self, batch: BatchColorPalettePrompt) -> None:
        self.color_palette_adapter.set_color_palette_embedding(
            self.color_palette_encoder.compute_color_palette_embedding(
                batch.source_palettes
            )
        )
    
    def draw_palette(self, palette: ColorPalette, width: int, palette_img_size: int) -> Image.Image:
        palette_img = Image.new(mode="RGB", size=(width, palette_img_size))
        
        # sort the palette by weight
        current_x = 0
        for (color, weight) in palette:
            box_width = int(weight*width)            
            color_box = Image.fromarray(np.full((palette_img_size, box_width, 3), color, dtype=np.uint8)) # type: ignore
            palette_img.paste(color_box, box=(current_x, 0))
            current_x+=box_width
            
        return palette_img
    
    def draw_cover_image(self, batch: BatchColorPaletteResults) -> Image.Image:
        (batch_size, _, height, width) = batch.result_images.shape
        
        palette_img_size = width // self.config.color_palette.max_colors
        source_images = batch.source_images

        join_canvas_image: Image.Image = Image.new(
            mode="RGB", size=(2*width, (height+palette_img_size) * batch_size)
        )
        images = tensor_to_images(batch.result_images)
        for i, image in enumerate(images):
            join_canvas_image.paste(source_images[i], box=(0, i*(height+palette_img_size)))
            join_canvas_image.paste(image, box=(width, i*(height+palette_img_size)))
            palette_out = batch.result_palettes[i]
            palette_out_img = self.draw_palette(palette_out, width, palette_img_size)
            palette_in_img = self.draw_palette(batch.source_palettes[i], width, palette_img_size)
            
            join_canvas_image.paste(palette_in_img, box=(0, i*(height+palette_img_size) + height))
            join_canvas_image.paste(palette_out_img, box=(width, i*(height+palette_img_size) + height))
        return join_canvas_image
    
    # def palette_distance(self, source: list[ColorPalette], result: list[ColorPalette]) -> float:
    #     if len(source) != len(result):
    #         raise ValueError("The source and result palettes must have the same length.")
        
    #     distance = 0.0
    #     for i in range(len(source)):
    #         distance += self.color_palette_extractor.distance(source[i], result[i])
            
    #     return distance
    
    def batch_metrics(self, results: BatchColorPaletteResults, prefix: str = "palette-img") -> None:
        palettes : list[list[Color]] = []
        for p in results.source_palettes:
            palettes.append([cluster[0] for cluster in p])
        
        images = tensor_to_images(results.result_images)
        batch_image_palette_metrics(
            self.log, 
            [
                ImageAndPalette({"image": image, "palette": palette})
                for image, palette in zip(images, palettes)
            ], 
            prefix
        )
    
class LoadColorPalette(Callback[ColorPaletteLatentDiffusionTrainer]):
    def on_train_begin(self, trainer: ColorPaletteLatentDiffusionTrainer) -> None:
        adapter = trainer.color_palette_adapter
        adapter.inject()

class SaveColorPalette(Callback[ColorPaletteLatentDiffusionTrainer]):
    def on_checkpoint_save(self, trainer: ColorPaletteLatentDiffusionTrainer) -> None:
        tensors: dict[str, Tensor] = {}

        model = trainer.unet
        if model.parent is None:
            raise ValueError("The model must have a parent.")
        adapter = model.parent

        tensors = {f"color_palette_adapter.{i:03d}.{j:03d}": w for i, ws in enumerate(adapter.weights) for j, w in enumerate(ws)}
        encoder = trainer.color_palette_encoder

        state_dict = encoder.state_dict()
        for i in state_dict:
            tensors.update({f"color_palette_encoder.{i}": state_dict[i]})
        
        path = f"{trainer.ensure_checkpoints_save_folder}/step{trainer.clock.step}.safetensors"
        logger.info(
            f"Saving {len(tensors)} tensors to {path}"
        )
        save_to_safetensors(
            path=path, tensors=tensors
        )