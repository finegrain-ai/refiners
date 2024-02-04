from functools import cached_property
from typing import Any, List, Tuple, TypedDict, Sequence

from loguru import logger
from PIL import Image, ImageDraw
from refiners.training_utils.trainers.histogram_auto_encoder import HistogramAutoEncoderConfig
from refiners.training_utils.config import TrainingConfig
from refiners.training_utils.huggingface_datasets import HuggingfaceDatasetConfig
from refiners.training_utils.wandb import WandbLoggable
from torch import Tensor, randn, stack, cat, uint8, empty
from refiners.fluxion.adapters.histogram_auto_encoder import HistogramAutoEncoder
from torch.utils.data import DataLoader

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.histogram import (
    HistogramDistance,
    HistogramExtractor,
    SD1HistogramAdapter,
    ColorLoss,
    histogram_to_histo_channels
)
from refiners.fluxion.utils import images_to_tensor, save_to_safetensors
from refiners.foundationals.latent_diffusion import (
    DPMSolver,
    StableDiffusion_1,
)
from refiners.training_utils.callback import Callback, GradientNormLayerLogging
from refiners.training_utils.datasets.color_palette import ColorPalette, ColorPaletteDataset, SamplingByPalette, TextEmbeddingColorPaletteLatentsBatch
from refiners.training_utils.metrics.color_palette import batch_palette_metrics
from refiners.training_utils.trainers.latent_diffusion import (
    FinetuneLatentDiffusionBaseConfig,
    LatentDiffusionBaseTrainer,
    TestDiffusionBaseConfig,
)
from refiners.training_utils.datasets.color_palette import ColorPaletteDataset

Color = Tuple[int, int, int]
Histogram = Tensor


class TestCoverHistogramConfig(TestDiffusionBaseConfig):
    histogram_db_indexes: List[int]


class ImageAndHistogram(TypedDict):
    image: Image.Image
    histogram: Histogram
    palette: ColorPalette

class BatchHistogramPrompt(Dataset):
    def __init__(
        self,
        source_histogram_embeddings: Tensor,
        source_histograms: Tensor,
        source_prompts: List[str],
        palettes: List[ColorPalette],
        text_embeddings: Tensor,
        db_indexes: List[int],
        source_images: List[Image.Image]
    ) -> None:
        self.source_histogram_embeddings = source_histogram_embeddings
        self.source_histograms = source_histograms
        self.source_prompts = source_prompts
        self.palettes = palettes
        self.text_embeddings = text_embeddings
        self.db_indexes = db_indexes
        self.source_images = source_images

    @classmethod
    def collate_fn(cls, batch: Sequence["BatchHistogramPrompt"]) -> "BatchHistogramPrompt":
        source_histograms = stack([item.source_histograms for item in batch])
        source_histogram_embeddings = stack([item.source_histogram_embeddings for item in batch])
        palettes = [palette for item in batch for palette in item.palettes]
        source_prompts = [prmpt for item in batch for prmpt in item.source_prompts]
        source_images = [image for item in batch for image in item.source_images]
        return BatchHistogramPrompt(
            db_indexes=[index for item in batch for index in item.db_indexes],
            source_histograms=source_histograms,
            source_histogram_embeddings=source_histogram_embeddings,
            source_prompts=source_prompts,
            palettes=palettes,
            text_embeddings=stack([item.text_embeddings for item in batch]),
            source_images=source_images
        )


class BatchHistogramResults(BatchHistogramPrompt):
    images: Tensor
    result_histograms: Tensor

    def __init__(
        self,
        images: Tensor,
        result_histograms: Tensor,
        source_histogram_embeddings: Tensor,
        source_histograms: Tensor,
        source_prompts: List[str],
        palettes: List[ColorPalette],
        text_embeddings: Tensor,
        source_images: List[Image.Image],
        db_indexes: List[int]
    ) -> None:
        super().__init__(
            source_histogram_embeddings=source_histogram_embeddings,
            source_histograms=source_histograms,
            source_prompts=source_prompts,
            palettes=palettes,
            text_embeddings=text_embeddings,
            db_indexes=db_indexes,
            source_images=source_images
        )

        self.images = images
        self.result_histograms = result_histograms

    def get_prompt(self, prompt: str) -> "BatchHistogramResults":
        indices = [i for i, p in enumerate(self.source_prompts) if p == prompt]

        return BatchHistogramResults(
            images=self.images[indices],
            result_histograms=self.result_histograms[indices],
            source_histogram_embeddings=self.source_histogram_embeddings[indices],
            source_histograms=self.source_histograms[indices],
            source_prompts=[prompt for _ in indices],
            palettes=[self.palettes[i] for i in indices],
            text_embeddings=self.text_embeddings[indices],
            db_indexes=[self.db_indexes[i] for i in indices],
            source_images=[self.source_images[i] for i in indices]
        )

    @classmethod
    def empty(cls) -> "BatchHistogramResults":
        return BatchHistogramResults(
            images=empty((0, 3, 512, 512)),
            result_histograms=empty((0, 64, 64, 64)),
            source_histogram_embeddings=empty((0, 8, 2, 2, 2)),
            source_histograms=empty((0, 64, 64, 64)),
            source_prompts=[],
            palettes=[],
            text_embeddings=empty((0, 1024, 768)),
            db_indexes=[],
            source_images=[]
        )

    def to_hist_prompt(self) -> BatchHistogramPrompt:
        return BatchHistogramPrompt(
            source_histogram_embeddings=self.source_histogram_embeddings,
            source_histograms=self.source_histograms,
            source_prompts=self.source_prompts,
            palettes=self.palettes,
            text_embeddings=self.text_embeddings,
            db_indexes=self.db_indexes,
            source_images=self.source_images
        )

    @classmethod
    def collate_fn(cls, batch: Sequence["BatchHistogramResults"]) -> "BatchHistogramResults":
        histo_prompts = [item.to_hist_prompt() for item in batch]
        histo_prompt = super().collate_fn(histo_prompts)

        images = stack([item.images for item in batch])
        result_histograms = stack([item.result_histograms for item in batch])
        return BatchHistogramResults(
            images=images,
            result_histograms=result_histograms,
            source_histograms=histo_prompt.source_histograms,
            source_histogram_embeddings=histo_prompt.source_histogram_embeddings,
            source_prompts=histo_prompt.source_prompts,
            palettes=histo_prompt.palettes,
            text_embeddings=histo_prompt.text_embeddings,
            db_indexes=histo_prompt.db_indexes,
            source_images=histo_prompt.source_images
        )



class ColorTrainingConfig(TrainingConfig):
    color_loss_weight: float = 1.0

class HistogramLatentDiffusionConfig(FinetuneLatentDiffusionBaseConfig):
    histogram_auto_encoder: HistogramAutoEncoderConfig
    test_cover_histogram: TestCoverHistogramConfig
    training: ColorTrainingConfig
    validation_dataset: HuggingfaceDatasetConfig




class HistogramLatentDiffusionTrainer(
    LatentDiffusionBaseTrainer[HistogramLatentDiffusionConfig, TextEmbeddingColorPaletteLatentsBatch]
):
    
    def load_dataset(self) -> ColorPaletteDataset:
        return ColorPaletteDataset(
            config=self.config.dataset,
            # use only palette 8 here
            sampling_by_palette = SamplingByPalette(
                sampling={
                    "palette_8": 1.0
                }
            )
		)

    @cached_property
    def dataset(self) -> ColorPaletteDataset:  # type: ignore
        return self.load_dataset() 

    @cached_property
    def histogram_auto_encoder(self) -> HistogramAutoEncoder:
        assert self.config.models["histogram_auto_encoder"] is not None, "The config must contain a histogram entry."
        
        autoencoder = HistogramAutoEncoder(
            latent_dim=self.config.histogram_auto_encoder.latent_dim, 
            resnet_sizes=self.config.histogram_auto_encoder.resnet_sizes,
            color_bits=self.config.histogram_auto_encoder.color_bits,
            n_down_samples=self.config.histogram_auto_encoder.n_down_samples,
            device=self.device
        )
        logger.info(f"Building autoencoder with compression rate {autoencoder.compression_rate}")
        return autoencoder

    @cached_property
    def histogram_adapter(self) -> SD1HistogramAdapter[Any]:
        embedding_dim = self.histogram_auto_encoder.embedding_dim
            
        adapter = SD1HistogramAdapter(target=self.unet, embedding_dim=embedding_dim)
        return adapter

    @cached_property
    def color_loss(self) -> ColorLoss:
        return ColorLoss()
    
    def __init__(
        self,
        config: HistogramLatentDiffusionConfig,
        callbacks: "list[Callback[Any]] | None" = None,
    ) -> None:
        super().__init__(config=config, callbacks=callbacks)
        self.callbacks.extend((LoadHistogram(), SaveHistogram(), GradientNormLayerLogging()))

    def load_models(self) -> dict[str, fl.Module]:
        return {
            "unet": self.unet,
            "text_encoder": self.text_encoder,
            "lda": self.lda,
            "histogram_auto_encoder": self.histogram_auto_encoder
        }

    def compute_loss(self, batch: TextEmbeddingColorPaletteLatentsBatch) -> Tensor:
        
        texts = [item.text for item in batch]
        text_embeddings = self.text_encoder(texts)

        images = [item.image for item in batch]
        latents = self.lda.images_to_latents(images)
        images_tensor = images_to_tensor(images, device=self.device, dtype=self.dtype)
        
        histograms = self.histogram_extractor.images_to_histograms([item.image for item in batch], device = self.device, dtype = self.dtype)
        histogram_embeddings = self.histogram_auto_encoder.encode(histograms)
        histogram_embeddings = histogram_embeddings.reshape(histogram_embeddings.shape[0], 1, -1)

        timestep = self.sample_timestep()
        noise = self.sample_noise(size=latents.shape, dtype=latents.dtype)
        noisy_latents = self.ddpm_solver.add_noise(x=latents, noise=noise, step=self.current_step)
        self.unet.set_timestep(timestep=timestep)

        self.unet.set_clip_text_embedding(clip_text_embedding=text_embeddings)
        self.histogram_adapter.set_histogram_embedding(histogram_embeddings)

        prediction = self.unet(noisy_latents)

        loss_1 = self.mse_loss(prediction, noise)
        
        predicted_latents = self.ddpm_solver.remove_noise(
            x=noisy_latents.to(device=self.ddpm_solver.device), 
            noise=prediction.to(device=self.ddpm_solver.device), 
            step=self.current_step
        )

        predicted_decoded = self.lda.decode(x=predicted_latents).to(device=self.device)
        predicted_images_tensor = (predicted_decoded + 1) / 2
        loss_2 = self.color_loss(
            predicted_images_tensor,
            images_tensor
        )
        
        self.log({
            f"losses/color_loss": loss_2.item(),
            f"losses/image": loss_1.item()
        })

        return loss_1 + loss_2 * self.config.training.color_loss_weight

    @cached_property
    def sd(self) -> StableDiffusion_1:
        solver = DPMSolver(
            device=self.device, num_inference_steps=self.config.test_cover_histogram.num_inference_steps, dtype=self.dtype
        )

        self.sharding_manager.add_device_hooks(solver, solver.device)
        return StableDiffusion_1(
            unet=self.unet, lda=self.lda, clip_text_encoder=self.text_encoder, solver=solver)
    
    @cached_property
    def eval_prompts(self) -> list[tuple[str, Tensor]]:
        return [(prompt, self.text_encoder(prompt)) for prompt in self.config.test_cover_histogram.prompts]
    
    @cached_property
    def eval_dataloader(self) -> DataLoader[BatchHistogramPrompt]:
                
        evaluations : List[BatchHistogramPrompt] = []
        
        for (prompt, prompt_embedding) in self.eval_prompts:
            for db_index, histogram, histogram_embedding, palette, image in self.eval_indices:
                batch_histogram_prompt = BatchHistogramPrompt(
                    source_histogram_embeddings= histogram_embedding,
                    source_histograms= histogram,
                    source_prompts= [prompt],
                    db_indexes= [db_index],
                    palettes= [palette],
                    text_embeddings= prompt_embedding,
                    source_images= [image]
                )
                evaluations.append(batch_histogram_prompt)
        
        return DataLoader(
            dataset=evaluations, 
            batch_size=self.config.training.batch_size, 
            shuffle=False,
            collate_fn=BatchHistogramPrompt.collate_fn, 
            num_workers=self.config.training.num_workers
        )
    
    @cached_property
    def unconditionnal_text_embedding(self) -> Tensor:
        return self.text_encoder("")
    
    def compute_batch_evaluation(self, batch: BatchHistogramPrompt, same_seed: bool = True) -> BatchHistogramResults:
        batch_size = len(batch.source_prompts)
        
        logger.info(f"Generating {batch_size} images for prompts: {batch.source_prompts}")
        
        if same_seed:
            x = randn(1, 4, 64, 64, dtype=self.dtype, device=self.device)
            x = x.repeat(batch_size, 1, 1, 1)
        else: 
            x = randn(batch_size, 4, 64, 64, dtype=self.dtype, device=self.device)
        
        unconditionnal_text_emb = self.unconditionnal_text_embedding.repeat(batch_size, 1, 1)
        cfg_clip_text_embedding = cat([batch.text_embeddings, unconditionnal_text_emb], dim=0)
        
        unconditionnal_histo_embedding = self.histogram_auto_encoder.unconditionnal_embedding_like(batch.source_histogram_embeddings)
        cfg_histogram_embedding = cat([batch.source_histogram_embeddings, unconditionnal_histo_embedding], dim=0)
        
        self.histogram_adapter.set_histogram_embedding(cfg_histogram_embedding)
        for step in self.sd.steps:
            x = self.sd(
                x,
                step=step,
                clip_text_embedding=cfg_clip_text_embedding,
            )
        
        image_tensors = (self.sd.lda.decode(x=x) + 1)/2
        
        return BatchHistogramResults(
            source_histogram_embeddings = batch.source_histogram_embeddings,
            source_histograms = batch.source_histograms,
            source_prompts = batch.source_prompts,
            palettes = batch.palettes,
            text_embeddings = batch.text_embeddings,
            images = image_tensors,
            result_histograms = self.histogram_extractor(image_tensors),
            db_indexes= batch.db_indexes,
            source_images= batch.source_images
        )
    
    def draw_curves(self, res_histo: list[float], src_histo: list[float], color: str, width: int, height: int) -> Image.Image:
        histo_img = Image.new(mode="RGB", size=(width, height))
        
        draw = ImageDraw.Draw(histo_img)
        
        if len(res_histo) != len(src_histo):
            raise ValueError("The histograms must have the same length.")
        
        ratio = width/len(res_histo)
        
        draw.line([
            (i*ratio, res_histo[i]*height) for i in range(len(res_histo))
        ], fill=color, width=4)
        
        draw.line([
            (i*ratio, src_histo[i]*height) for i in range(len(src_histo))
        ], fill=color, width=1)
        
        return histo_img
    
    def draw_cover_image(self, batch: BatchHistogramResults) -> Image.Image:
        (batch_size, channels, height, width) = batch.images.shape
        vertical_images = batch.images.view(1, channels, height*batch_size, width)
        results_histograms = batch.result_histograms
        source_histograms = batch.source_histograms
        source_images = batch.source_images

        join_canvas_image: Image.Image = Image.new(
            mode="RGB", size=(width + width//2, height * batch_size)
        )
        res_image = (vertical_images * 255).to(dtype=uint8).permute(1, 2, 0).cpu().numpy()
        join_canvas_image.paste(res_image, box=(width//2, 0))
        
        res_histo_channels = histogram_to_histo_channels(results_histograms)
        src_histo_channels = histogram_to_histo_channels(source_histograms)
        
        colors = ["red", "green", "blue"]
        
        for i in range(batch_size):
            join_canvas_image.paste(source_images[i].resize((width//2, height//2)), box=(0, height *i))

            for (color_id, color_name) in enumerate(colors):
                image_curve = self.draw_curves(
                    res_histo_channels[color_id][i].cpu().tolist(), # type: ignore
                    src_histo_channels[color_id][i].cpu().tolist(), # type: ignore
                    color_name,
                    width//2,
                    height//6
                )
                join_canvas_image.paste(image_curve, box=(0, height *i + height//2 + color_id*height//6))
                
        return join_canvas_image
    
    def compute_evaluation(
        self
    ) -> None:
        
        per_prompts : dict[str, BatchHistogramResults] = {}
        images : dict[str, WandbLoggable] = {}
        
        all_results : BatchHistogramResults = BatchHistogramResults.empty()
        
        
        for batch in self.eval_dataloader:
            results = self.compute_batch_evaluation(batch)
        
            for prompt in list(set(results.source_prompts)):
                batch = results.get_prompt(prompt)
                if prompt not in per_prompts:
                    per_prompts[prompt] = batch
                else:
                    per_prompts[prompt] = BatchHistogramResults.collate_fn([
                        per_prompts[prompt],
                        batch
                    ])
        
        for prompt in per_prompts:
            image = self.draw_cover_image(per_prompts[prompt])
            image_name = f"eval_images/{prompt}"
            images[image_name] = image
        self.log(data=images)
        all_results = BatchHistogramResults.collate_fn(list(per_prompts.values()))
        self.batch_image_histogram_metrics(all_results, prefix="eval")

    @cached_property
    def eval_indices(self) -> list[tuple[int, Histogram, Tensor, ColorPalette, Image.Image]]:
        dataset = self.dataset
        indices = self.config.test_cover_histogram.histogram_db_indexes
        items = [dataset[i][0] for i in indices]
        palette = [item.color_palette for item in items]
        images = [item.image for item in items]
        histograms = self.histogram_extractor.images_to_histograms(images, device = self.device, dtype = self.dtype)
        histogram_embeddings = self.histogram_auto_encoder(histograms)
        return list(zip(indices, histograms, histogram_embeddings, palette, images))

    @cached_property
    def histogram_extractor(self) -> HistogramExtractor:
        return HistogramExtractor(color_bits=self.config.histogram_auto_encoder.color_bits)

    @cached_property
    def histogram_distance(self) -> HistogramDistance:
        return HistogramDistance(color_bits=self.config.histogram_auto_encoder.color_bits)

    def batch_image_histogram_metrics(
        self, images_and_histograms: BatchHistogramResults, prefix: str = "histogram-img"
    ) -> None:
        
        self.log({f"{prefix}/mse": self.histogram_distance(
            images_and_histograms.source_histograms, 
            images_and_histograms.result_histograms
        )})
        
        self.log({f"{prefix}/rgb_distance": self.color_loss.image_vs_histo(
                images_and_histograms.images,
                images_and_histograms.source_histograms,
                self.config.histogram_auto_encoder.color_bits
        ).item()})

        batch_palette_metrics(self.log, images_and_histograms, prefix)

    # def compute_evaluation(self) -> None:
    #     prompts = self.config.test_cover_histogram.prompts
    #     num_images_per_prompt = self.config.test_cover_histogram.num_images_per_prompt
    #     images_and_histograms: List[ImageAndHistogram] = []
        
    #     if len(prompts) > 0:
    #         images_and_histograms = self.compute_db_samples_evaluation(
    #             prompts, 
    #             num_images_per_prompt
    #         )
    #         self.batch_image_histogram_metrics(images_and_histograms, prefix="histogram-image-edge")


class LoadHistogram(Callback[HistogramLatentDiffusionTrainer]):
    def on_train_begin(self, trainer: HistogramLatentDiffusionTrainer) -> None:
        adapter = trainer.histogram_adapter
        adapter.zero_init()
        adapter.inject()


class SaveHistogram(Callback[HistogramLatentDiffusionTrainer]):
    def on_checkpoint_save(self, trainer: HistogramLatentDiffusionTrainer) -> None:
        tensors: dict[str, Tensor] = {}
        # metadata: dict[str, str] = {}

        model = trainer.unet
        if model.parent is None:
            raise ValueError("The model must have a parent.")
        adapter = model.parent

        tensors = {f"unet.{i:03d}": w for i, w in enumerate(adapter.weights)}

        save_to_safetensors(
            path=trainer.ensure_checkpoints_save_folder / f"step{trainer.clock.step}.safetensors", tensors=tensors
        )
