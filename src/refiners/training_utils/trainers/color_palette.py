from functools import cached_property
from typing import Any, List, TypedDict
from refiners.fluxion.utils import load_from_safetensors, tensor_to_images

from loguru import logger
from PIL import Image
from pydantic import BaseModel
from src.refiners.training_utils.trainers.histogram import HistogramLatentDiffusionTrainer
from torch import Tensor, randn
import numpy as np

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.color_palette import ColorPaletteEncoder, SD1ColorPaletteAdapter, ColorPaletteExtractor
from refiners.fluxion.utils import save_to_safetensors
from refiners.foundationals.latent_diffusion import (
    DPMSolver,
    StableDiffusion_1,
    SD1UNet
)
from refiners.training_utils.callback import Callback
from refiners.training_utils.datasets.color_palette import ColorPaletteDataset
from refiners.training_utils.metrics.color_palette import BatchHistogramPrompt, ImageAndPalette, batch_image_palette_metrics
from refiners.training_utils.trainers.latent_diffusion import (
    FinetuneLatentDiffusionBaseConfig,
    LatentDiffusionBaseTrainer,
    TestDiffusionBaseConfig,
)
from refiners.training_utils.wandb import WandbLoggable
from refiners.training_utils.datasets.color_palette import ColorPalette, ColorPaletteDataset, TextEmbeddingColorPaletteLatentsBatch
from refiners.training_utils.callback import GradientNormLayerLogging

from refiners.training_utils.trainers.trainer import scoped_seed

class ColorPaletteConfig(BaseModel):
    feedforward_dim: int = 3072
    num_attention_heads: int = 12
    num_layers: int = 12
    embedding_dim: int = 768
    trigger_phrase: str = ""
    use_only_trigger_probability: float = 0.0
    max_colors: int
    mode : str = "transformer"
    without_caption_probability: float = 0.17

class ColorPalettePromptConfig(BaseModel):
    text: str
    color_palette: ColorPalette

class LatentPrompt(TypedDict):
    text: str
    color_palette_embedding: Tensor

class TestColorPaletteConfig(TestDiffusionBaseConfig):
    prompts: list[ColorPalettePromptConfig]
    num_palette_sample: int = 0

class ImageAndPalette(TypedDict):
    image: Image.Image
    palette: ColorPalette   

class ColorPaletteLatentDiffusionConfig(FinetuneLatentDiffusionBaseConfig):
    color_palette: ColorPaletteConfig
    test_color_palette: TestColorPaletteConfig



class ColorPaletteLatentDiffusionTrainer(HistogramLatentDiffusionTrainer):
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
            num_attention_heads=self.config.color_palette.num_attention_heads,
            feedforward_dim=self.config.color_palette.feedforward_dim,
            device=self.device,
        )
        return encoder

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
    def color_palette_extractor(self) -> ColorPaletteExtractor:
        return ColorPaletteExtractor(
            size=self.config.color_palette.max_colors
        )
        
    def compute_loss(self, batch: TextEmbeddingColorPaletteLatentsBatch) -> Tensor:
        
        texts = [item.text for item in batch]
        text_embeddings = self.text_encoder(texts)
        
        latents = self.lda.encode_images([item.image for item in batch])
        color_palettes = [self.color_palette_extractor(item.image) for item in batch]
        
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
    
    def set_adapter_values(self, batch: BatchHistogramPrompt) -> None:
        self.color_palette_adapter.set_color_palette_embedding(
            self.color_palette_encoder.compute_color_palette_embedding(
                batch.color_palette
            )
        )
    
    def draw_palette(self, palette: ColorPalette, width: int, height: int) -> Image.Image:
        palette_img = Image.new(mode="RGB", size=(width, height))
        for i, (color, _) in enumerate(palette):
            palette_img.paste(color, box=(i*height, 0))
        return palette_img
    
    def draw_cover_image(self, batch: BatchHistogramResults) -> Image.Image:
        (batch_size, channels, height, width) = batch.images.shape
        
        palette_img_size = width // self.config.color_palette.max_colors
        source_images = batch.source_images

        join_canvas_image: Image.Image = Image.new(
            mode="RGB", size=(2*width, (height+palette_img_size) * batch_size)
        )
        images = tensor_to_images(batch.images)
        for i, image in enumerate(images):
            join_canvas_image.paste(source_images[i], box=(0, i*(height+palette_img_size)))
            join_canvas_image.paste(image, box=(width, i*(height+palette_img_size)))
            palette_out = self.palette_extractor(image)
            palette_out_img = self.draw_palette(palette_out, width, palette_img_size)
            palette_in_img = self.draw_palette(batch.palettes[i], width, palette_img_size)
            
            join_canvas_image.paste(palette_in_img, box=(0, i*(height+palette_img_size) + height))
            join_canvas_image.paste(palette_out_img, box=(width, i*(height+palette_img_size) + height))
        return join_canvas_image
    def compute_edge_case_evaluation(
        self, prompts: List[ColorPalettePromptConfig], num_images_per_prompt: int
    ) -> List[ImageAndPalette]:
        images: dict[str, WandbLoggable] = {}
        images_and_palettes: List[ImageAndPalette] = []
        
        for prompt in prompts:
            image_name = f"edge_case/{prompt.text.replace(' ', '_')} : {str(prompt.color_palette)}"
            image_and_palette = self.compute_deterministic_prompt_evaluation(prompt, num_images_per_prompt)
            images[image_name] = image_and_palette["image"]
            images_and_palettes.append(image_and_palette)

        self.log(data=images)
        return images_and_palettes

    @cached_property
    def eval_indices(self) -> list[tuple[int, ColorPalette, str]]:
        l = self.dataset_length
        size = self.config.test_color_palette.num_palette_sample
        indices = list(np.random.choice(l, size=size, replace=False)) # type: ignore
        indices : List[int] = list(map(int, indices)) # type: ignore
        palettes = [self.dataset.get_color_palette(i) for i in indices]
        captions = [self.dataset.get_caption(i) for i in indices]
        return list(zip(indices, palettes, captions))

    def batch_image_palette_metrics(self, images_and_palettes: List[ImageAndPalette], prefix: str = "palette-img"):
        batch_image_palette_metrics(self.log, images_and_palettes, prefix)

    def compute_db_samples_evaluation(self, num_images_per_prompt: int, img_size: int = 512) -> List[ImageAndPalette]:
        images: dict[str, WandbLoggable] = {}
        images_and_palettes: List[ImageAndPalette] = []
        palette_img_size = img_size // self.config.color_palette.max_colors

        for (db_index, palette, caption) in self.eval_indices:
            prompt = ColorPalettePromptConfig(text=caption, color_palette=palette)
            image_and_palette = self.compute_prompt_evaluation(prompt, 1, img_size=img_size)

            image = self.dataset.get_image(db_index)
            resized_image = image.resize((img_size, img_size))
            join_canvas_image: Image.Image = Image.new(mode="RGB", size=(img_size, img_size * 2 + palette_img_size))
            join_canvas_image.paste(image_and_palette["image"], box=(0, 0))
            join_canvas_image.paste(resized_image, box=(0, img_size + palette_img_size))
            image_name = f"db_samples/{db_index}_{caption}"

            images[image_name] = join_canvas_image
            images_and_palettes.append(image_and_palette)

        self.log(data=images)
        return images_and_palettes

    def compute_evaluation(self) -> None:
        prompts = self.config.test_color_palette.prompts
        num_images_per_prompt = self.config.test_color_palette.num_images_per_prompt
        images_and_palettes: List[ImageAndPalette] = []
        if len(prompts) > 0:
            images_and_palettes = self.compute_edge_case_evaluation(prompts, num_images_per_prompt)
            self.batch_image_palette_metrics(images_and_palettes, prefix="palette-image-edge")

        num_palette_sample = self.config.test_color_palette.num_palette_sample
        if num_palette_sample > 0:
            images_and_palettes = self.compute_db_samples_evaluation(num_images_per_prompt)
            self.batch_image_palette_metrics(images_and_palettes, prefix="palette-image-samples")


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