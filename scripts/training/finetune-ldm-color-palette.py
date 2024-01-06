import random
from typing import Any
from functools import cached_property

from loguru import logger
from pydantic import BaseModel
from torch import Tensor,tensor, float32, cat
from torch.utils.data import Dataset
from refiners.fluxion.adapters.color_palette import SD1ColorPaletteAdapter, ColorPaletteEncoder
import refiners.fluxion.layers as fl
from refiners.fluxion.utils import save_to_safetensors
from refiners.training_utils.callback import Callback
from refiners.training_utils.huggingface_datasets import HuggingfaceDatasetConfig
from refiners.training_utils.latent_diffusion import (
    FinetuneLatentDiffusionConfig,
    LatentDiffusionConfig,
    LatentDiffusionTrainer,
    TextEmbeddingLatentsBatch,
    TextEmbeddingLatentsDataset,
)
from refiners.foundationals.latent_diffusion import (
    DPMSolver,
    StableDiffusion_1,
)
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import os
import hashlib
from dataclasses import dataclass

class ColorPaletteConfig(BaseModel):
    max_colors: int
    model_dim: int
    trigger_phrase: str = ""
    use_only_trigger_probability: float = 0.0

class ColorPaletteDatasetConfig(HuggingfaceDatasetConfig):
    local_folder: str = "data/color-palette"

@dataclass
class TextEmbeddingColorPaletteLatentsBatch:
    text_embeddings: Tensor
    latents: Tensor
    color_palette_embeddings: Tensor

class ColorPaletteDataset(TextEmbeddingLatentsDataset):

    def __init__(
        self,
        trainer: "ColorPaletteLatentDiffusionTrainer",
    ) -> None:
        super().__init__(trainer=trainer)
        self.trigger_phrase = trainer.config.color_palette.trigger_phrase
        self.use_only_trigger_probability = trainer.config.color_palette.use_only_trigger_probability
        logger.info(f"Trigger phrase: {self.trigger_phrase}")
        self.color_palette_encoder = trainer.color_palette_encoder

        self.local_folder = trainer.config.dataset.local_folder
        
        # Download images
        # Question : there might be a more efficient way to do this
        # I didn't find the way to do this easily with hugging face
        # dataset library
        for item in tqdm(self.dataset, desc="Downloading images"):
            self.download_image(item)
    
    def get_image_path_from_url(self, url: str) -> str:
        hash_md5 = hashlib.md5()
        hash_md5.update(url.encode())
        filename = hash_md5.hexdigest()
        return self.local_folder + f"/{filename}"
    
    def download_image(self, item: dict[str, Any]) -> None:
        url = item["url"]
        image_path = self.get_image_path_from_url(url)
        if not os.path.exists(image_path):
            # download image from url
            logger.info(f"Downloading image {image_path} from {url}")
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                # Save the image bytes to the image_path
                with open(image_path, "wb") as file:
                    file.write(response.content)
            else:
                print(f"Failed to download image from {url}")
                return None
            
    def get_caption(self, index: int) -> str:
        return self.dataset[index]["ai_description"]

    def get_image(self, index: int) -> str:
        url = self.dataset[index]["url"]
        image_path = self.get_image_path_from_url(url)
        
        if not os.path.exists(image_path):
            raise Exception(f"Image {image_path} does not exist")
        return Image.open(image_path)
    def process_caption(self, caption: str) -> str:
        caption = super().process_caption(caption=caption)
        if self.trigger_phrase:
            caption = (
                f"{self.trigger_phrase} {caption}"
                if random.random() < self.use_only_trigger_probability
                else self.trigger_phrase
            )
        return caption
    def get_color_palette(self, index: int) -> Tensor:
        # TO IMPLEMENT : use other palettes
        return tensor([self.dataset[index]["palette_8"]])
    def __getitem__(self, index: int) -> TextEmbeddingColorPaletteLatentsBatch:
        caption = self.get_caption(index=index)
        color_palette = self.get_color_palette(index=index)
        image = self.get_image(index=index)
        resized_image = self.resize_image(
            image=image,
            min_size=self.config.dataset.resize_image_min_size,
            max_size=self.config.dataset.resize_image_max_size,
        )
        processed_image = self.process_image(resized_image)
        latents = self.lda.encode_image(image=processed_image).to(device=self.device)
        processed_caption = self.process_caption(caption=caption)
        
        clip_text_embedding = self.text_encoder(processed_caption).to(device=self.device)
        color_palette_embedding = self.color_palette_encoder(color_palette).to(device=self.device)
        return TextEmbeddingColorPaletteLatentsBatch(
            text_embeddings=clip_text_embedding,
            latents=latents,
            color_palette_embeddings=color_palette_embedding
        )
    def collate_fn(self, batch: list[TextEmbeddingColorPaletteLatentsBatch]) -> TextEmbeddingColorPaletteLatentsBatch:
        text_embeddings = cat(tensors=[item.text_embeddings for item in batch])
        latents = cat(tensors=[item.latents for item in batch])
        color_palette_embeddings = cat(tensors=[item.color_palette_embeddings for item in batch])
        return TextEmbeddingColorPaletteLatentsBatch(text_embeddings=text_embeddings, latents=latents, color_palette_embeddings=color_palette_embeddings)

class ColorPaletteLatentDiffusionConfig(FinetuneLatentDiffusionConfig):
    dataset: ColorPaletteDatasetConfig
    latent_diffusion: LatentDiffusionConfig
    color_palette: ColorPaletteConfig

    def model_post_init(self, __context: Any) -> None:
        """Pydantic v2 does post init differently, so we need to override this method too."""
        logger.info("Freezing models to train only the color palette.")
        self.models["text_encoder"].train = False
        self.models["lda"].train = False
        self.models["color_palette_encoder"].train = True
        
        # Question : Here I should not freeze the CrossAttentionBlock2d 
        # But what is the unfreeze only this block ?
        self.models["unet"].train = False
        
class ColorPaletteLatentDiffusionTrainer(LatentDiffusionTrainer[ColorPaletteLatentDiffusionConfig]):
    @cached_property
    def color_palette_encoder(self) -> ColorPaletteEncoder:
        assert self.config.models["color_palette_encoder"] is not None, "The config must contain a color_palette_encoder entry."
        
        # TO FIX : connect this to unet cross attention embedding dim
        EMBEDDING_DIM = 768
        
        return ColorPaletteEncoder(
            max_colors=self.config.color_palette.max_colors,
            embedding_dim=EMBEDDING_DIM,
            model_dim=self.config.color_palette.model_dim,
            device=self.device
        )

    def __init__(
        self,
        config: ColorPaletteLatentDiffusionConfig,
        callbacks: "list[Callback[Any]] | None" = None,
    ) -> None:
        super().__init__(config=config, callbacks=callbacks)
        self.callbacks.extend((LoadColorPalette(), SaveColorPalette()))

    def load_dataset(self) -> Dataset[TextEmbeddingColorPaletteLatentsBatch]:
        return ColorPaletteDataset(trainer=self)
    
    def load_models(self) -> dict[str, fl.Module]:
        return {
            "unet": self.unet, 
            "text_encoder": self.text_encoder, 
            "lda": self.lda,
            "color_palette_encoder": self.color_palette_encoder
        }

    def set_adapter(self, adapter) -> None:
        self.adapter = adapter
    
    def compute_loss(self, batch: TextEmbeddingColorPaletteLatentsBatch) -> Tensor:
        text_embeddings, latents, color_palette_embeddings = batch.text_embeddings, batch.latents, batch.color_palette_embeddings
        timestep = self.sample_timestep()
        noise = self.sample_noise(size=latents.shape, dtype=latents.dtype)
        noisy_latents = self.ddpm_scheduler.add_noise(x=latents, noise=noise, step=self.current_step)
        
        self.unet.set_timestep(timestep=timestep)

        clip_text_embedding = cat([
            text_embeddings, 
            color_palette_embeddings
        ], dim=1)
        
        self.unet.set_clip_text_embedding(clip_text_embedding=clip_text_embedding)
        prediction = self.unet(noisy_latents)
        loss = mse_loss(input=prediction, target=noise)
        return loss
    # def compute_evaluation(self) -> None:
    #     sd = StableDiffusion_1(
    #         unet=self.unet,
    #         lda=self.lda,
    #         clip_text_encoder=self.text_encoder,
    #         scheduler=DPMSolver(
    #             num_inference_steps=self.config.test_diffusion.num_inference_steps
    #         ),
    #         device=self.device,
    #     )
    #     prompts = self.config.test_diffusion.prompts
    #     num_images_per_prompt = self.config.test_diffusion.num_images_per_prompt
    #     if self.config.test_diffusion.use_short_prompts:
    #         prompts = [prompt.split(sep=",")[0] for prompt in prompts]
    #     images: dict[str, WandbLoggable] = {}
    #     for prompt in prompts:
    #         canvas_image: Image.Image = Image.new(mode="RGB", size=(512, 512 * num_images_per_prompt))
    #         for i in range(num_images_per_prompt):
    #             logger.info(f"Generating image {i+1}/{num_images_per_prompt} for prompt: {prompt}")
    #             x = randn(1, 4, 64, 64, device=self.device)
                
    #             clip_text_embedding = sd.compute_clip_text_embedding(text=prompt).to(device=self.device)
                
    #             for step in sd.steps:
    #                 x = sd(
    #                     x,
    #                     step=step,
    #                     clip_text_embedding=clip_text_embedding,
    #                 )
    #             canvas_image.paste(sd.lda.decode_latents(x=x), box=(0, 512 * i))
    #         images[prompt] = canvas_image
    #     self.log(data=images)
class LoadColorPalette(Callback[ColorPaletteLatentDiffusionTrainer]):
    def on_train_begin(self, trainer: ColorPaletteLatentDiffusionTrainer) -> None:
        color_palette_config = trainer.config.color_palette
        
        adapter = SD1ColorPaletteAdapter(
            target=trainer.unet,
            color_palette_encoder=trainer.color_palette_encoder
        )
        
        trainer.set_adapter(adapter)
        
        adapter.inject()


class SaveColorPalette(Callback[ColorPaletteLatentDiffusionTrainer]):
    def on_checkpoint_save(self, trainer: ColorPaletteLatentDiffusionTrainer) -> None:
        tensors: dict[str, Tensor] = {}
        metadata: dict[str, str] = {}
        
        model = trainer.unet
        adapter = model.parent
        
        tensors = {f"unet.{i:03d}": w for i, w in enumerate(adapter.weights)}
        metadata = {f"unet_targets": ",".join(adapter.sub_targets)}
        
        save_to_safetensors(
            path=trainer.ensure_checkpoints_save_folder / f"step{trainer.clock.step}.safetensors",
            tensors=tensors,
            metadata=metadata,
        )


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1]
    config = ColorPaletteLatentDiffusionConfig.load_from_toml(
        toml_path=config_path,
    )
    trainer = ColorPaletteLatentDiffusionTrainer(config=config)
    trainer.train()
