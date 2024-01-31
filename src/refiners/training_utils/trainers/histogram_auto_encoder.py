from ast import List
from functools import cached_property
from typing import Any
from loguru import logger

from torch import Tensor
from refiners.fluxion.adapters.histogram_auto_encoder import HistogramAutoEncoder
from torch.nn.functional import mse_loss

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.histogram import (
    HistogramDistance,
    HistogramExtractor
)
from torch.utils.data import DataLoader
from refiners.fluxion.utils import save_to_safetensors

from refiners.training_utils.callback import Callback, GradientNormLayerLogging
from refiners.training_utils.config import BaseConfig
from refiners.training_utils.datasets.color_palette import ColorPaletteDataset, TextEmbeddingColorPaletteLatentsBatch
from refiners.training_utils.huggingface_datasets import HuggingfaceDatasetConfig
from refiners.training_utils.trainers.trainer import Trainer
from pydantic import BaseModel
import wandb

class HistogramAutoEncoderConfig(BaseModel):
    latent_dim: int
    resnet_sizes: list[int]
    n_down_samples: int
    color_bits: int

class TrainHistogramAutoEncoderConfig(BaseConfig):
    dataset: HuggingfaceDatasetConfig
    histogram_auto_encoder: HistogramAutoEncoderConfig
    eval_dataset: HuggingfaceDatasetConfig

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
    
    @cached_property
    def histogram_auto_encoder(self) -> HistogramAutoEncoder:
        assert self.config.models["histogram_auto_encoder"] is not None, "The config must contain a histogram entry."
        autoencoder = HistogramAutoEncoder(
            latent_dim=self.config.histogram_auto_encoder.latent_dim, 
            resnet_sizes=self.config.histogram_auto_encoder.resnet_sizes,
            n_down_samples=self.config.histogram_auto_encoder.n_down_samples,
            device=self.device
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
        
        actual = self.histogram_extractor.images_to_histograms([item.image for item in batch], device = self.device, dtype = self.dtype)

        expected = self.histogram_auto_encoder(actual)

        loss = mse_loss(actual, expected)

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
    
    def compute_evaluation_metrics(self, batch: TextEmbeddingColorPaletteLatentsBatch) -> Tensor:
        
        eval_loss = self.compute_loss(batch)
        self.log({f"eval/loss": eval_loss})

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
        histogram_auto_encoder.state_dict()
        
        path = f"{trainer.ensure_checkpoints_save_folder}/step{trainer.clock.step}.safetensors"
        
        logger.info(
            f"Saving {len(tensors)} tensors to {path}"
        )
        save_to_safetensors(
            path=path, tensors=tensors
        )