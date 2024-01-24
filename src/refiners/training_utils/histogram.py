from dataclasses import dataclass
from functools import cached_property
from pydantic import BaseModel
from typing import Any, List, Tuple, TypedDict

from loguru import logger
from PIL import Image
from pydantic import BaseModel
from torch import Tensor, randn, tensor, stack

import refiners.fluxion.layers as fl
from refiners.fluxion.utils import save_to_safetensors
from refiners.foundationals.latent_diffusion import (
    DPMSolver,
    StableDiffusion_1,
)
from torch.utils.data import Dataset

from refiners.training_utils.callback import Callback
from refiners.training_utils.latent_diffusion import (
    FinetuneLatentDiffusionBaseConfig,
    LatentDiffusionBaseTrainer,
    TextEmbeddingLatentsBaseDataset,
    TestDiffusionBaseConfig,
    TextEmbeddingLatentsBatch,
)
from refiners.training_utils.trainer import Trainer
from refiners.training_utils.config import BaseConfig
from refiners.training_utils.color_palette import ColorPalette
from refiners.fluxion.adapters.histogram import HistogramEncoder, HistogramExtractor, HistogramDistance, SD1HistogramAdapter

from refiners.training_utils.wandb import WandbLoggable
import numpy as np

class HistogramConfig(BaseModel):
    feedforward_dim: int = 256
    color_bits: int = 5
    num_attention_heads: int = 2
    num_layers: int = 2
    trigger_phrase: str = ""
    use_only_trigger_probability: float = 0.0

Color = Tuple[int, int, int]
Histogram = Tensor

class HistogramPrompt(TypedDict):
    text: str
    histogram: Histogram

class HistogramDbIndexPromptConfig(BaseModel):
    text: str
    histogram_db_index: int

class TestHistogramConfig(TestDiffusionBaseConfig):
    prompts: list[HistogramDbIndexPromptConfig]
    num_samples: int = 0

class ImageAndHistogram(TypedDict):
    image: Image.Image
    histogram: Histogram
    palette: ColorPalette

@dataclass
class TextEmbeddingHistogramLatentsBatch(TextEmbeddingLatentsBatch):
    text_embeddings: Tensor
    latents: Tensor
    histogram_embeddings: Tensor

class HistogramLatentDiffusionConfig(FinetuneLatentDiffusionBaseConfig):
    histogram: HistogramConfig
    test_histogram: TestHistogramConfig

class GradientNormLogging(Callback["Trainer[BaseConfig, Any]"]):
    def on_backward_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        named_gradient_norm = trainer.named_gradient_norm
        for (layer_name, norm) in named_gradient_norm:
            trainer.log(data={f"layer_grad_norm/{layer_name}": norm})


class HistogramLatentsDataset(TextEmbeddingLatentsBaseDataset[TextEmbeddingHistogramLatentsBatch]):
    def __getitem__(self, index: int) -> TextEmbeddingLatentsBatch:
        item = self.dataset[index]
        latents = item['latents']
        image = item['image']
        clip_text_embedding = item['clip_text_embedding']
        histogram = self.histogram_extractor(image)
        histogram_embedding = self.histogram_encoder(histogram)
        
        if random.random() > self.config.latent_diffusion.unconditional_sampling_probability:
            clip_text_embedding = empty_clip_text_embedding
            # TODO: empty histogram embedding
        
        return TextEmbeddingHistogramLatentsBatch(
            text_embeddings=clip_text_embedding, 
            latents=latents,
            histogram_embeddings=histogram_embeddings
        )
    
    @cached_property
    def empty_clip_text_embedding(self) -> Tensor:
        return self.text_encoder.encode_text("")

    def collate_fn(self, batch: list[TextEmbeddingLatentsBatch]) -> TextEmbeddingLatentsBatch:
        text_embeddings = cat(tensors=[item.text_embeddings for item in batch])
        latents = cat(tensors=[item.latents for item in batch])
        histogram_embeddings = cat(tensors=[item.histogram_embeddings for item in batch])
        
        return TextEmbeddingHistogramLatentsBatch(
            text_embeddings=text_embeddings, 
            latents=latents,
            histogram_embeddings=histogram_embeddings
        )


class HistogramLatentDiffusionTrainer(
    LatentDiffusionBaseTrainer[HistogramLatentDiffusionConfig, TextEmbeddingHistogramLatentsBatch]
):
    def load_dataset(self) -> Dataset[TextEmbeddingHistogramLatentsBatch]:
        return HistogramLatentsDataset(trainer=self)

    @cached_property
    def histogram_encoder(self) -> HistogramEncoder:
        assert (
            self.config.models["histogram"] is not None
        ), "The config must contain a histogram entry."

        # TO FIX : connect this to unet cross attention embedding dim
        EMBEDDING_DIM = 768

        encoder = HistogramEncoder(
            color_bits=self.config.histogram.color_bits,
            embedding_dim=EMBEDDING_DIM,
            num_layers=self.config.histogram.num_layers,
            num_attention_heads=self.config.histogram.num_attention_heads,
            feedforward_dim=self.config.histogram.feedforward_dim,
            device=self.device,
        )
        return encoder

    def encoder_grad_norm(self) -> float:
        return self.histogram_encoder.grad_norm()
    
    def cross_attention_grad_norm(self) -> float:
        return self.unet.cross_attention_grad_norm()
    
    @cached_property
    def histogram_adapter(self) -> SD1HistogramAdapter[Any]:
        adapter = SD1HistogramAdapter(target=self.unet, histogram_encoder=self.histogram_encoder)
        return adapter
    
    def __init__(
        self,
        config: HistogramLatentDiffusionConfig,
        callbacks: "list[Callback[Any]] | None" = None,
    ) -> None:
        super().__init__(config=config, callbacks=callbacks)
        self.callbacks.extend((LoadHistogram(), SaveHistogram(), GradientNormLogging()))

    def load_models(self) -> dict[str, fl.Module]:
        return {
            "unet": self.unet,
            "text_encoder": self.text_encoder,
            "lda": self.lda,
            "histogram_encoder": self.histogram_encoder,
        }

    def compute_loss(self, batch: TextEmbeddingHistogramLatentsBatch) -> Tensor:
        
        text_embeddings, latents, histogram_embeddings = (
            batch.text_embeddings,
            batch.latents,
            batch.histogram_embeddings,
        )
        
        timestep = self.sample_timestep()
        noise = self.sample_noise(size=latents.shape, dtype=latents.dtype)
        noisy_latents = self.ddpm_scheduler.add_noise(x=latents, noise=noise, step=self.current_step)
        self.unet.set_timestep(timestep=timestep)

        self.unet.set_clip_text_embedding(clip_text_embedding=text_embeddings)
        self.histogram_adapter.set_histogram_embedding(histogram_embeddings)

        prediction = self.unet(noisy_latents)
        loss = self.mse_loss(prediction, noise)
        return loss

    @cached_property
    def sd(self) -> StableDiffusion_1:
        scheduler = DPMSolver(
            device=self.device,
            num_inference_steps=self.config.test_histogram.num_inference_steps,
            dtype=self.dtype
        )
        
        self.sharding_manager.add_device_hooks(scheduler, scheduler.device)
        return StableDiffusion_1(unet=self.unet, lda=self.lda, clip_text_encoder=self.text_encoder, scheduler=scheduler)
    
    def compute_prompt_evaluation(self, prompt: HistogramPrompt, num_images_per_prompt: int, img_size: int = 512) -> ImageAndHistogram:
        sd = self.sd
        canvas_image: Image.Image = Image.new(mode="RGB", size=(img_size * num_images_per_prompt, img_size))
        for i in range(num_images_per_prompt):
            logger.info(
                f"Generating image {i+1}/{num_images_per_prompt} for prompt: {prompt['text']}"
            )
            x = randn(1, 4, 64, 64, dtype=self.dtype, device=self.device)

            cfg_clip_text_embedding = sd.compute_clip_text_embedding(text=prompt['text']).to(device=self.device)
            cfg_histogram_embedding = self.histogram_encoder.compute_histogram_embedding(
                tensor([prompt['histogram']])
            )

            self.histogram_adapter.set_histogram_embedding(cfg_histogram_embedding)

            for step in sd.steps:
                x = sd(
                    x,
                    step=step,
                    clip_text_embedding=cfg_clip_text_embedding,
                )
            canvas_image.paste(sd.lda.decode_latents(x=x), box=(img_size * i, 0))

        return ImageAndHistogram(image=canvas_image, histogram=prompt.histogram, palette=prompt.palette)
    
    def compute_edge_case_evaluation(self, 
            prompts: List[HistogramDbIndexPromptConfig], 
            num_images_per_prompt: int, 
            img_size: int = 512
        ) -> List[ImageAndHistogram]:
        images: dict[str, WandbLoggable] = {}
        images_and_histograms: List[ImageAndHistogram] = []
        for prompt in prompts:
            (db_image) = self.dataset[prompt.histogram_db_index]
            histogram = self.histogram_extractor(db_image)
            prompt_histo = HistogramPrompt(text=prompt.text, histogram=histogram, palette=palette)
            image_name = f"edge_case/{prompt.text.replace(' ', '_')} : with colors from {prompt.histogram_db_index}"
            image_and_histogram = self.compute_prompt_evaluation(prompt_histo, num_images_per_prompt)
            top_image = image_and_histogram['image']
            resized_image = image.resize((img_size, img_size))
            
            join_canvas_image: Image.Image = Image.new(mode="RGB", size=(img_size, img_size*2))
            join_canvas_image.paste(top_image, box=(0, 0))
            join_canvas_image.paste(resized_image, box=(0, img_size))
            images[image_name] = join_canvas_image
            images_and_histograms.append(image_and_histogram)
        
        self.log(data=images)
        return images_and_histograms
        
    
    @cached_property
    def eval_indices(self) -> list[tuple[int, Histogram, str]]:
        l = self.dataset_length
        dataset = self.dataset
        size = self.config.test_histogram.num_samples
        indices = list(np.random.choice(l, size=size, replace=False))
        indices = list(map(int, indices))
        histograms = [self.histogram_extractor(dataset.get_image(i)) for i in indices]
        captions = [self.dataset.get_caption(i, self.config.dataset.caption_key) for i in indices]
        return list(zip(indices, histograms, captions))
    
    @cached_property
    def histogram_extractor(self) -> HistogramExtractor:
        return HistogramExtractor(color_bits=self.config.histogram.color_bits)
    
    @cached_property
    def histogram_distance(self) -> HistogramDistance:
        return HistogramDistance(color_bits=self.config.histogram.color_bits)
    
    def batch_image_histogram_metrics(self, images_and_histograms: List[ImageAndHistogram], prefix: str = "histogram-img") -> None:
        expected_histograms : List[Histogram] = [image_and_histogram['histogram'] for image_and_histogram in images_and_histograms]
        actual_histograms : List[Histogram] = [self.histogram_extractor(image_and_histogram['image']) for image_and_histogram in images_and_histograms]
        
        self.log({
            f"{prefix}/mse": self.histogram_distance(stack(actual_histograms), stack(expected_histograms))
        })
            
    def compute_db_samples_evaluation(self, num_images_per_prompt: int, img_size: int = 512) -> List[ImageAndHistogram]:
        images: dict[str, WandbLoggable] = {}
        images_and_histograms : List[ImageAndHistogram] = []
        for (db_index, histogram, caption) in self.eval_indices:            
            prompt = HistogramPrompt(text=caption, histogram=histogram)
            image_and_histogram = self.compute_prompt_evaluation(prompt, 1, img_size=img_size)
            
            image = self.dataset.get_image(db_index)
            resized_image = image.resize((img_size, img_size))
            join_canvas_image: Image.Image = Image.new(mode="RGB", size=(img_size, img_size*2))
            join_canvas_image.paste(image_and_histogram['image'], box=(0, 0))
            join_canvas_image.paste(resized_image, box=(0, img_size))
            image_name = f"db_samples/{db_index}_{caption}"

            images[image_name] = join_canvas_image
            images_and_histograms.append(image_and_histogram)
            
        self.log(data=images)
        return images_and_histograms
    
    def compute_evaluation(self) -> None:
        prompts = self.config.test_histogram.prompts
        num_images_per_prompt = self.config.test_histogram.num_images_per_prompt
        images_and_histograms : List[ImageAndHistogram] = []
        if len(prompts) > 0:
            images_and_histograms = self.compute_edge_case_evaluation(prompts, num_images_per_prompt)
            self.batch_image_histogram_metrics(images_and_histograms, prefix="histogram-image-edge")

        num_samples = self.config.test_histogram.num_samples
        if num_samples > 0:
            images_and_histograms = self.compute_db_samples_evaluation(num_images_per_prompt)
            self.batch_image_histogram_metrics(images_and_histograms, prefix="histogram-histogram-samples")
        
class LoadHistogram(Callback[HistogramLatentDiffusionTrainer]):
    def on_train_begin(self, trainer: HistogramLatentDiffusionTrainer) -> None:
        adapter = trainer.histogram_adapter
        adapter.zero_init()      
        adapter.inject()
        
class SaveHistogram(Callback[HistogramLatentDiffusionTrainer]):
    def on_checkpoint_save(self, trainer: HistogramLatentDiffusionTrainer) -> None:
        tensors: dict[str, Tensor] = {}
        #metadata: dict[str, str] = {}

        model = trainer.unet
        if model.parent is None:
            raise ValueError("The model must have a parent.")
        adapter = model.parent

        tensors = {f"unet.{i:03d}": w for i, w in enumerate(adapter.weights)}
        encoder = trainer.histogram_encoder
            
        state_dict = encoder.state_dict()
        for i in state_dict:
            tensors.update({f"histogram_encoder.{i}": state_dict[i]})

        save_to_safetensors(
            path=trainer.ensure_checkpoints_save_folder / f"step{trainer.clock.step}.safetensors",
            tensors=tensors
        )
