from functools import cached_property
from tkinter import W
from typing import Any
from loguru import logger
from refiners.training_utils.wandb import WandbLoggable

from torch import Tensor, isnan
from refiners.fluxion.adapters.histogram_auto_encoder import HistogramAutoEncoder

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.histogram import (
    HistogramDistance,
    HistogramExtractor,
    histogram_to_histo_channels
)
from torch.utils.data import DataLoader
from refiners.fluxion.utils import save_to_safetensors

from refiners.training_utils.callback import Callback, GradientNormLayerLogging
from refiners.training_utils.config import BaseConfig
from refiners.training_utils.datasets.color_palette import ColorDatasetConfig, ColorPaletteDataset, TextEmbeddingColorPaletteLatentsBatch
from refiners.training_utils.huggingface_datasets import HuggingfaceDatasetConfig
from refiners.training_utils.trainers.trainer import Trainer
from pydantic import BaseModel
from refiners.fluxion.adapters.color_palette import ColorPaletteExtractor, ColorPalette
from PIL import Image, ImageDraw

class HistogramAutoEncoderConfig(BaseModel):
    latent_dim: int
    resnet_sizes: list[int]
    n_down_samples: int
    color_bits: int
    num_groups: int = 4
    loss: str = "kl_div"

class TrainHistogramAutoEncoderConfig(BaseConfig):
    dataset: ColorDatasetConfig
    histogram_auto_encoder: HistogramAutoEncoderConfig
    eval_dataset: ColorDatasetConfig

class HistogramAutoEncoderTrainer(
    Trainer[TrainHistogramAutoEncoderConfig, TextEmbeddingColorPaletteLatentsBatch]
):
    def load_dataset(self) -> ColorPaletteDataset:
        return ColorPaletteDataset(
            config=self.config.dataset
		)
    
    @cached_property
    def dataset(self) -> ColorPaletteDataset:  # type: ignore
        return self.load_dataset() 
    
    @cached_property
    def eval_dataset(self) -> ColorPaletteDataset:  # type: ignore
        return self.load_eval_dataset() 
    
    def load_eval_dataset(self) -> ColorPaletteDataset:
        return ColorPaletteDataset(
            config=self.config.eval_dataset
		)
    
    def draw_palette(self, palette: ColorPalette, width: int, height: int) -> Image.Image:
        palette_img = Image.new(mode="RGB", size=(width, height))
        
        # sort the palette by weight
        current_x = 0
        for (color, weight) in palette:
            box_width = int(weight*width)            
            color_box = Image.fromarray(np.full((height, box_width, 3), color, dtype=np.uint8)) # type: ignore
            palette_img.paste(color_box, box=(current_x, 0))
            current_x+=box_width
            
        return palette_img
    
    @cached_property
    def color_palette_extractor(self) -> ColorPaletteExtractor:
        return ColorPaletteExtractor(
            size=8,
            weighted_palette=True
        )

    @cached_property
    def histogram_auto_encoder(self) -> HistogramAutoEncoder:
        assert self.config.models["histogram_auto_encoder"] is not None, "The config must contain a histogram entry."
        autoencoder = HistogramAutoEncoder(
            num_groups=self.config.histogram_auto_encoder.num_groups,
            latent_dim=self.config.histogram_auto_encoder.latent_dim, 
            resnet_sizes=self.config.histogram_auto_encoder.resnet_sizes,
            n_down_samples=self.config.histogram_auto_encoder.n_down_samples,
            device=self.device,
            color_bits=self.config.histogram_auto_encoder.color_bits
        )
        logger.info(f"Building autoencoder with compression rate {autoencoder.compression_rate}")
        return autoencoder

    def __init__(
        self,
        config: TrainHistogramAutoEncoderConfig,
        callbacks: "list[Callback[Any]] | None" = None,
    ) -> None:
        super().__init__(config=config, callbacks=callbacks)
        self.callbacks.extend((GradientNormLayerLogging(),SaveHistogramAutoEncoder()))

    def load_models(self) -> dict[str, fl.Module]:
        return {
            "histogram_auto_encoder": self.histogram_auto_encoder
        }

    def compute_loss(self, batch: TextEmbeddingColorPaletteLatentsBatch) -> Tensor:
        
        expected = self.histogram_extractor.images_to_histograms([item.image for item in batch], device = self.device, dtype = self.dtype)

        actual_logits = self.histogram_auto_encoder(expected)
        
        if isnan(actual).any():
            raise ValueError("The autoencoder produced NaNs.")
        
        if self.config.histogram_auto_encoder.loss == "mse":
            actual = actual_logits.reshape(expected.shape[0], -1).softmax(dim=1).reshape(expected.shape)
            loss = self.histogram_distance.mse(actual, expected)
        elif self.config.histogram_auto_encoder.loss == "kl_div":
            loss = self.histogram_distance.kl_div(actual_logits, expected)
        else:
            raise ValueError(f"Unknown loss {self.config.histogram_auto_encoder.loss}")
        
        return loss

    @cached_property
    def histogram_extractor(self) -> HistogramExtractor:
        return HistogramExtractor(color_bits=self.config.histogram_auto_encoder.color_bits)

    @cached_property
    def histogram_distance(self) -> HistogramDistance:
        return HistogramDistance(color_bits=self.config.histogram_auto_encoder.color_bits)
    
    @cached_property
    def eval_dataloader(self) -> DataLoader[TextEmbeddingColorPaletteLatentsBatch]:
        
        collate_fn = getattr(self.eval_dataset, "collate_fn", None)
        return DataLoader(
            dataset=self.eval_dataset, 
            batch_size=self.config.training.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn, 
            num_workers=self.config.training.num_workers
        )
    
    def draw_curves(self, res_histo: list[float], src_histo: list[float], color: str, width: int, height: int) -> Image.Image:
        histo_img = Image.new(mode="RGB", size=(width, height))
        
        draw = ImageDraw.Draw(histo_img)
        
        if len(res_histo) != len(src_histo):
            raise ValueError("The histograms must have the same length.")
        
        ratio = width/len(res_histo)
        semi_height = height//2
        
        scale_ratio = 5
                
        draw.line([
            (i*ratio, (1-res_histo[i]*scale_ratio)*semi_height + semi_height) for i in range(len(res_histo))
        ], fill=color, width=4)
        
        draw.line([
            (i*ratio, (1-src_histo[i]*scale_ratio)*semi_height) for i in range(len(src_histo))
        ], fill=color, width=1)
        
        return histo_img

    def compute_evaluation_metrics(self, batch: TextEmbeddingColorPaletteLatentsBatch) -> Tensor:
                
        expected = self.histogram_extractor.images_to_histograms([item.image for item in batch], device = self.device, dtype = self.dtype)

        actual_logits = self.histogram_auto_encoder(expected)
        
        actual = actual_logits.reshape(expected.shape[0], -1).softmax(dim=1).reshape(expected.shape)
        
        metrics = self.histogram_distance.metrics(actual, expected)
        log_dict : dict[str, WandbLoggable] = {}
        for (key, value) in metrics.items():
            log_dict[f"eval/{key}"] = value
        self.log(log_dict)
        
        
        images : dict[str, WandbLoggable] = {}
        
        res_histo_channels = histogram_to_histo_channels(actual)
        src_histo_channels = histogram_to_histo_channels(expected)
        
        batch_size = expected.shape[0]
        
        histo_h = 30
        width = 256
        
        colors = ["red", "green", "blue"]

        joint_canvas = Image.new(mode="RGB", size=(256, 3*histo_h * batch_size))
        draw = ImageDraw.Draw(joint_canvas)
        for i in range(batch_size):
            draw.line([(0, histo_h * 3 * i), (width, histo_h * 3 * i)], fill="white", width=5)
            for (color_id, color_name) in enumerate(colors):
                image_curve = self.draw_curves(
                    res_histo_channels[color_id][i].cpu().tolist(), # type: ignore
                    src_histo_channels[color_id][i].cpu().tolist(), # type: ignore
                    color_name,
                    width,
                    histo_h
                )
                
                joint_canvas.paste(image_curve, box=(0, histo_h * (3 * i + color_id)))
        
        images[f"color_curves/eval"] = joint_canvas
        self.log(data=images)


        # images = [item.image for item in batch]
        # [red_gt, green_gt, blue_gt] = images_to_histo_channels(images)
        # histograms = self.histogram_extractor.images_to_histograms(images, device = self.device, dtype = self.dtype)
        # histograms_pred = self.histogram_auto_encoder(histograms)
        # [red_pred, green_pred, blue_pred] = histogram_to_histo_channels(histograms_pred)
        
        # color_bits = self.config.histogram_auto_encoder.color_bits
        # x_width = 2**(8 - color_bits)
        # x_values = [i*x_width for i in range(2**color_bits)]

        # for i, _ in enumerate(images):
        #     data = [list(a) for a in zip(
        #         x_values, 
        #         red_gt[i].tolist(), 
        #         green_gt[i].tolist(), 
        #         blue_gt[i].tolist(),
        #         red_pred[i].tolist(), 
        #         green_pred[i].tolist(), 
        #         blue_pred[i].tolist()              
        #     )]
        #     table = wandb.Table(data=data, columns = ["color", "red_gt", "green_gt", "blue_gt", "red_pred", "green_pred", "blue_pred"])
        #     self.log({f"eval/image_{i}": table})

    def compute_evaluation(self) -> None:
        
        for batch in self.eval_dataloader:
            self.compute_evaluation_metrics(batch)

class SaveHistogramAutoEncoder(Callback[HistogramAutoEncoderTrainer]):
    def on_checkpoint_save(self, trainer: HistogramAutoEncoderTrainer) -> None:

        histogram_auto_encoder = trainer.histogram_auto_encoder
        tensors = histogram_auto_encoder.state_dict()
        
        path = f"{trainer.ensure_checkpoints_save_folder}/step{trainer.clock.step}.safetensors"
        
        logger.info(
            f"Saving {len(tensors)} tensors to {path}"
        )
        save_to_safetensors(
            path=path, tensors=tensors
        )