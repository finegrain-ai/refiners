from dataclasses import dataclass
from functools import cached_property
from random import randint
from typing import Any, List, TypedDict, Tuple
from pydantic import BaseModel


from loguru import logger
from PIL import Image
from pydantic import BaseModel
from torch import Tensor, cat, randn, tensor
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import ndcg_score

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.color_palette import ColorPaletteEncoder, SD1ColorPaletteAdapter
from refiners.fluxion.utils import save_to_safetensors
from refiners.foundationals.latent_diffusion import (
    DPMSolver,
    StableDiffusion_1,
)
from refiners.training_utils.callback import Callback
from refiners.training_utils.huggingface_datasets import HuggingfaceDatasetConfig
from refiners.training_utils.latent_diffusion import (
    CaptionImage,
    FinetuneLatentDiffusionBaseConfig,
    LatentDiffusionBaseTrainer,
    TestDiffusionBaseConfig,
    TextEmbeddingLatentsBaseDataset,
    TextEmbeddingLatentsBatch,
)
from refiners.training_utils.wandb import WandbLoggable
import numpy as np

class ColorPaletteConfig(BaseModel):
    feedforward_dim: int
    trigger_phrase: str = ""
    use_only_trigger_probability: float = 0.0
    max_colors: int
    

Color = Tuple[int, int, int]
ColorPalette = List[Color]

class ColorPalettePromptConfig(BaseModel):
    text: str
    color_palette: ColorPalette

class ColorPaletteDatasetConfig(HuggingfaceDatasetConfig):
    local_folder: str = "data/color-palette"

class TestColorPaletteConfig(TestDiffusionBaseConfig):
    prompts: list[ColorPalettePromptConfig]
    num_palette_sample: int = 0
    condition_scale: float = 7.5

@dataclass
class TextEmbeddingColorPaletteLatentsBatch(TextEmbeddingLatentsBatch):
    text_embeddings: Tensor
    latents: Tensor
    color_palette_embeddings: Tensor

class CaptionPaletteImage(CaptionImage):
    palette_1: ColorPalette
    palette_2: ColorPalette
    palette_3: ColorPalette
    palette_4: ColorPalette
    palette_5: ColorPalette
    palette_6: ColorPalette
    palette_7: ColorPalette
    palette_8: ColorPalette

class ImageAndPalette(TypedDict):
    image: Image.Image
    palette: ColorPalette

class ColorPaletteDataset(TextEmbeddingLatentsBaseDataset[TextEmbeddingColorPaletteLatentsBatch]):
    def __init__(
        self,
        trainer: "ColorPaletteLatentDiffusionTrainer",
    ) -> None:
        super().__init__(trainer=trainer)
        self.trigger_phrase = trainer.config.color_palette.trigger_phrase
        self.use_only_trigger_probability = trainer.config.color_palette.use_only_trigger_probability
        logger.info(f"Trigger phrase: {self.trigger_phrase}")
        self.color_palette_encoder = trainer.color_palette_encoder


    def get_color_palette(self, index: int) -> ColorPalette:
        # Randomly pick a palette between 1 and 8
        palette_index = randint(1, 8)
        return self.dataset[index][f"palette_{palette_index}"]
        
        # Pick color palette 8
        #return self.dataset[index][f"palette_8"]

    def __getitem__(self, index: int) -> TextEmbeddingColorPaletteLatentsBatch:
        caption = self.get_caption(index=index, caption_key=self.config.dataset.caption_key)
        color_palette = tensor([self.get_color_palette(index=index)], dtype=self.trainer.dtype)
        image = self.get_image(index=index)
        resized_image = self.resize_image(
            image=image,
            min_size=self.config.dataset.resize_image_min_size,
            max_size=self.config.dataset.resize_image_max_size,
        )
        processed_image = self.process_image(resized_image)
        latents = self.lda.encode_image(image=processed_image)
        processed_caption = self.process_caption(caption=caption)

        clip_text_embedding = self.text_encoder(processed_caption)
        color_palette_embedding = self.color_palette_encoder(color_palette)
        return TextEmbeddingColorPaletteLatentsBatch(
            text_embeddings=clip_text_embedding, latents=latents, color_palette_embeddings=color_palette_embedding
        )

    def collate_fn(self, batch: list[TextEmbeddingColorPaletteLatentsBatch]) -> TextEmbeddingColorPaletteLatentsBatch:
        text_embeddings = cat(tensors=[item.text_embeddings for item in batch])
        latents = cat(tensors=[item.latents for item in batch])
        color_palette_embeddings = cat(tensors=[item.color_palette_embeddings for item in batch])
        return TextEmbeddingColorPaletteLatentsBatch(
            text_embeddings=text_embeddings, latents=latents, color_palette_embeddings=color_palette_embeddings
        )


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

        # TO FIX : connect this to unet cross attention embedding dim
        EMBEDDING_DIM = 768

        encoder = ColorPaletteEncoder(
            max_colors=self.config.color_palette.max_colors,
            embedding_dim=EMBEDDING_DIM,
            feedforward_dim=self.config.color_palette.feedforward_dim,
            device=self.device,
        )
        return encoder

    @cached_property
    def color_palette_adapter(self) -> SD1ColorPaletteAdapter[Any]:
        adapter = SD1ColorPaletteAdapter(target=self.unet, color_palette_encoder=self.color_palette_encoder)
        return adapter

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
            "color_palette_encoder": self.color_palette_encoder,
        }

    def compute_loss(self, batch: TextEmbeddingColorPaletteLatentsBatch) -> Tensor:
        text_embeddings, latents, color_palette_embeddings = (
            batch.text_embeddings,
            batch.latents,
            batch.color_palette_embeddings,
        )
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
            device=self.device,
            num_inference_steps=self.config.test_color_palette.num_inference_steps,
            dtype=self.dtype
        )

        self.sharding_manager.add_device_hooks(scheduler, scheduler.device)

        return StableDiffusion_1(unet=self.unet, lda=self.lda, clip_text_encoder=self.text_encoder, scheduler=scheduler)
    
    def compute_prompt_evaluation(self, prompt: ColorPalettePromptConfig, num_images_per_prompt: int, img_size: int = 512) -> ImageAndPalette:
        sd = self.sd
        palette_img_size = img_size//self.config.color_palette.max_colors
        canvas_image: Image.Image = Image.new(mode="RGB", size=(img_size * num_images_per_prompt, img_size+palette_img_size))
        for i in range(num_images_per_prompt):
            logger.info(
                f"Generating image {i+1}/{num_images_per_prompt} for prompt: {prompt.text} and palette {prompt.color_palette}"
            )
            x = randn(1, 4, 64, 64, dtype=self.dtype, device=self.device)

            cfg_clip_text_embedding = sd.compute_clip_text_embedding(text=prompt.text).to(device=self.device)
            cfg_color_palette_embedding = self.color_palette_encoder.compute_color_palette_embedding(
                [prompt.color_palette]
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
                color_box = Image.fromarray(np.full((palette_img_size, palette_img_size, 3), palette, dtype=np.uint8)) # type: ignore
                canvas_image.paste(color_box,box=(img_size * i+palette_img_size*index, img_size))
            
        return ImageAndPalette(image=canvas_image, palette=prompt.color_palette)
    
    def compute_edge_case_evaluation(self, prompts: List[ColorPalettePromptConfig], num_images_per_prompt: int) -> List[ImageAndPalette]:
        images: dict[str, WandbLoggable] = {}
        images_and_palettes: List[ImageAndPalette] = []
        for prompt in prompts:
            image_name = f"edge_case/{prompt.text.replace(' ', '_')} : {str(prompt.color_palette)}"
            image_and_palette = self.compute_prompt_evaluation(prompt, num_images_per_prompt)
            images[image_name] = image_and_palette['image']
            images_and_palettes.append(image_and_palette)
        
        self.log(data=images)
        return images_and_palettes
        
    
    @cached_property
    def eval_indices(self) -> list[int]:
        l = self.dataset_length
        size = self.config.test_color_palette.num_palette_sample
        indices = list(np.random.choice(l, size=size, replace=False))
        return list(map(int, indices))
    
    def image_palette_metrics(self, image: Image.Image, palette: ColorPalette, img_size : Tuple[int, int]=(256,256), sampling_size :int = 1000):
        resized_img = image.resize(img_size)
        all_points : ColorPalette = np.array(resized_img.getdata(), dtype=np.float64)
        choices = np.random.choice(len(all_points), sampling_size)
        points = all_points[choices]
        
        num = len(palette)
        
        centroids = np.stack(palette) 
        
        nn = NearestNeighbors(n_neighbors=num)
        nn.fit(centroids) 
        
        indices = nn.kneighbors(points, return_distance=False)
        indices = indices[:, 0]
        
        counts = np.bincount(indices)
        counts = np.pad(counts, (0, num - len(counts)), 'constant')
        y_true_ranking = list(range(num, 0, -1))
        
        distances_list = []
        
        def distance(a, b):
            return np.linalg.norm(a - b) 
        
        for i in range(len(centroids)):
            cluster_points = points[np.where(indices == i)]
            distances = [distance(p, centroids[i]) for p in cluster_points]
            distances_list.extend(distances)
                
        return ([y_true_ranking], [counts], distances_list)
    
    def batch_image_palette_metrics(self, images_and_palettes: List[ImageAndPalette], prefix: str = "palette-img"):

        per_num : dict[int, Any] = {}
        for image_and_palette in images_and_palettes:
            palette = image_and_palette['palette']
            image = image_and_palette['image']
            num = len(palette)
            
            (y_true_ranking, counts, distances_list) = self.image_palette_metrics(image, palette)
            if not num in per_num:
                per_num[num] = {
                    "y_true_ranking": y_true_ranking,
                    "counts": counts,
                    "distances": distances_list,
                }
            else: 
                per_num[num]["y_true_ranking"] += y_true_ranking
                per_num[num]["counts"] += counts
                per_num[num]["distances"] += distances_list

        for num in per_num:
            if num > 1:
                self.log({
                    f"{prefix}/ndcg_{num}": ndcg_score(per_num[num]["y_true_ranking"], per_num[num]["counts"]),
                    f"{prefix}/std_dev_{num}": np.std(per_num[num]["distances"])
                })
            else:
                self.log({
                    f"{prefix}/std_dev_{num}": np.std(per_num[num]["distances"])
                })
    def compute_db_samples_evaluation(self, num_images_per_prompt: int, img_size: int = 512) -> None:
        sd = self.sd
        images: dict[str, WandbLoggable] = {}
        images_and_palettes : List[ImageAndPalette] = []
        palette_img_size = img_size//self.config.color_palette.max_colors

        for eval_index, db_index in enumerate(self.eval_indices):
            # we only eval on palette_8 here
            palette = self.dataset.dataset[db_index][f"palette_8"]
            caption = self.dataset.get_caption(db_index, self.config.dataset.caption_key)
            
            prompt = ColorPalettePromptConfig(text=caption, color_palette=palette)
            image_and_palette = self.compute_prompt_evaluation(prompt, 1, img_size=img_size)
            
            image = self.dataset.get_image(db_index)
            resized_image = image.resize((img_size, img_size))
            join_canvas_image: Image.Image = Image.new(mode="RGB", size=(img_size, img_size*2+palette_img_size))
            join_canvas_image.paste(image_and_palette['image'], box=(0, 0))
            join_canvas_image.paste(resized_image, box=(0, img_size+palette_img_size))
            image_name = f"db_samples/{db_index}_{caption}"

            images[image_name] = join_canvas_image
            images_and_palettes.append(image_and_palette)
            
        self.log(data=images)
        return images_and_palettes
    
    def compute_evaluation(self) -> None:
        prompts = self.config.test_color_palette.prompts
        num_images_per_prompt = self.config.test_color_palette.num_images_per_prompt
        images_and_palettes : List[ImageAndPalette] = []
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
        adapter.zero_init()      
        adapter.inject()
        


class SaveColorPalette(Callback[ColorPaletteLatentDiffusionTrainer]):
    def on_checkpoint_save(self, trainer: ColorPaletteLatentDiffusionTrainer) -> None:
        tensors: dict[str, Tensor] = {}
        metadata: dict[str, str] = {}

        model = trainer.unet
        if model.parent is None:
            raise ValueError("The model must have a parent.")
        adapter = model.parent

        tensors = {f"unet.{i:03d}": w for i, w in enumerate(adapter.weights)}
        metadata = {f"unet_targets": ",".join(adapter.sub_targets)}

        save_to_safetensors(
            path=trainer.ensure_checkpoints_save_folder / f"step{trainer.clock.step}.safetensors",
            tensors=tensors,
            metadata=metadata,
        )
