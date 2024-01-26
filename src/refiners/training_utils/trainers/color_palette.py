from functools import cached_property
from typing import Any, List, TypedDict
from refiners.fluxion.utils import load_from_safetensors

from loguru import logger
from PIL import Image
from pydantic import BaseModel
from torch import Tensor, randn, tensor, cat
import numpy as np
from refiners.fluxion.utils import image_to_tensor

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.color_palette import ColorPaletteEncoder, SD1ColorPaletteAdapter
from refiners.fluxion.utils import save_to_safetensors
from refiners.foundationals.latent_diffusion import (
    DPMSolver,
    StableDiffusion_1,
    SD1UNet
)
from refiners.training_utils.callback import Callback
from refiners.training_utils.datasets.color_palette import ColorPaletteDataset
from refiners.training_utils.metrics.color_palette import ImageAndPalette, batch_image_palette_metrics
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

class ColorPaletteLatentDiffusionTrainer(
    LatentDiffusionBaseTrainer[ColorPaletteLatentDiffusionConfig, TextEmbeddingColorPaletteLatentsBatch]
):
    @cached_property
    def color_palette_encoder(self) -> ColorPaletteEncoder:
        assert (
            self.config.models["color_palette_encoder"] is not None
        ), "The config must contain a color_palette_encoder entry."

        encoder = ColorPaletteEncoder(
            max_colors=self.config.color_palette.max_colors,
            embedding_dim=self.config.color_palette.embedding_dim,
            num_layers=self.config.color_palette.num_layers,
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

    def load_dataset(self) -> ColorPaletteDataset:
        return ColorPaletteDataset(
            config=self.config.dataset
		)
    
    @cached_property
    def dataset(self) -> ColorPaletteDataset:  # type: ignore
        return self.load_dataset() 
    
    def load_models(self) -> dict[str, fl.Module]:
        return {
            "unet": self.unet,
            "text_encoder": self.text_encoder,
            "lda": self.lda,
            "color_palette_encoder": self.color_palette_encoder,
        }

    def compute_loss(self, batch: TextEmbeddingColorPaletteLatentsBatch) -> Tensor:
        
        texts = [item.text for item in batch]
        text_embeddings = self.text_encoder(texts)
        
        image_tensor = cat([image_to_tensor(item.image, device=self.lda.device, dtype=self.lda.dtype) for item in batch])
        
        latents = self.lda.encode(image_tensor)
        color_palette_embeddings = self.color_palette_encoder(tensor([item.color_palette for item in batch], dtype=self.color_palette_encoder.dtype, device=self.color_palette_encoder.device))

        timestep = self.sample_timestep()
        noise = self.sample_noise(size=latents.shape, dtype=latents.dtype)
        noisy_latents = self.ddpm_scheduler.add_noise(x=latents, noise=noise, step=self.current_step)
        self.unet.set_timestep(timestep=timestep)

        self.unet.set_clip_text_embedding(clip_text_embedding=text_embeddings)
        self.color_palette_adapter.set_color_palette_embedding(color_palette_embeddings)

        prediction = self.unet(noisy_latents)
        loss = self.mse_loss(prediction, noise)
        return loss

    @cached_property
    def sd(self) -> StableDiffusion_1:
        scheduler = DPMSolver(
            device=self.device, num_inference_steps=self.config.test_color_palette.num_inference_steps, dtype=self.dtype
        )

        self.sharding_manager.add_device_hooks(scheduler, scheduler.device)

        return StableDiffusion_1(unet=self.unet, lda=self.lda, clip_text_encoder=self.text_encoder, scheduler=scheduler)
    
    @scoped_seed(42)
    def compute_deterministic_prompt_evaluation(
        self, prompt: ColorPalettePromptConfig, num_images_per_prompt: int, img_size: int = 512
    ) -> ImageAndPalette:
        return self.compute_prompt_evaluation(prompt, num_images_per_prompt, img_size=img_size)
        
    def compute_prompt_evaluation(
        self, prompt: ColorPalettePromptConfig, num_images_per_prompt: int, img_size: int = 512
    ) -> ImageAndPalette:
        sd = self.sd
        palette_img_size = img_size // self.config.color_palette.max_colors
        canvas_image: Image.Image = Image.new(
            mode="RGB", size=(img_size * num_images_per_prompt, img_size + palette_img_size)
        )
        for i in range(num_images_per_prompt):
            logger.info(
                f"Generating image {i+1}/{num_images_per_prompt} for prompt: {prompt.text} and palette {prompt.color_palette}"
            )
            x = randn(1, 4, 64, 64, dtype=self.dtype, device=self.device)

            cfg_clip_text_embedding = sd.compute_clip_text_embedding(text=prompt.text).to(device=self.device)
            cfg_color_palette_embedding = self.color_palette_encoder.compute_color_palette_embedding(
                tensor([prompt.color_palette])
            )

            self.color_palette_adapter.set_color_palette_embedding(cfg_color_palette_embedding)

            for step in sd.steps:
                x = sd(
                    x,
                    step=step,
                    clip_text_embedding=cfg_clip_text_embedding,
                )
            canvas_image.paste(sd.lda.decode_latents(x=x), box=(img_size * i, 0))
            for index, palette in enumerate(prompt.color_palette):
                color_box = Image.fromarray(np.full((palette_img_size, palette_img_size, 3), palette, dtype=np.uint8))  # type: ignore
                canvas_image.paste(color_box, box=(img_size * i + palette_img_size * index, img_size))

        return ImageAndPalette(image=canvas_image, palette=prompt.color_palette)

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

        tensors = {f"color_palette_adapter.{i:03d}": w for i, w in enumerate(adapter.weights)}
        encoder = trainer.color_palette_encoder

        state_dict = encoder.state_dict()
        for i in state_dict:
            tensors.update({f"color_palette_encoder.{i}": state_dict[i]})
        
        path = trainer.ensure_checkpoints_save_folder / f"step{trainer.clock.step}.safetensors"
        logger.info(
            f"Saving {len(tensors)} tensors to {path}"
        )
        save_to_safetensors(
            path=path, tensors=tensors
        )