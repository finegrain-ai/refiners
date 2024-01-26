import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from pydantic import BaseModel
from PIL import Image
from refiners.training_utils.datasets.latent_diffusion import TextEmbeddingLatentsBaseDataset
from refiners.training_utils.huggingface_datasets import HuggingfaceDatasetConfig
from loguru import logger

Color = Tuple[int, int, int]

ColorPalette = List[Color]

@dataclass
class ColorPaletteDatasetItem:
    color_palette: ColorPalette
    text: str
    image: Image.Image
    conditional_flag: bool

TextEmbeddingColorPaletteLatentsBatch = List[ColorPaletteDatasetItem]

class SamplingByPalette(BaseModel):
    palette_1: float = 1.0
    palette_2: float = 2.0
    palette_3: float = 3.0
    palette_4: float = 4.0
    palette_5: float = 5.0
    palette_6: float = 6.0
    palette_7: float = 7.0
    palette_8: float = 8.0


class ColorPaletteDataset(TextEmbeddingLatentsBaseDataset[TextEmbeddingColorPaletteLatentsBatch]):
    def __init__(
        self,
        config: HuggingfaceDatasetConfig,
        sampling_by_palette: SamplingByPalette = SamplingByPalette(),
        unconditional_sampling_probability: float = 0.2,
    ) -> None:
        self.sampling_by_palette = sampling_by_palette
        super().__init__(
            config=config,
            unconditional_sampling_probability=unconditional_sampling_probability,
        )

    def __getitem__(self, index: int) -> TextEmbeddingColorPaletteLatentsBatch:
        logger.info(f"Getting latents {index}")
        
        image = self.get_processed_image(index)
        (caption, conditional_flag) = self.process_caption(self.get_caption(index))        
        
        return [
            ColorPaletteDatasetItem(
                color_palette=self.get_color_palette(index),
                text=caption,
                image=image,
                conditional_flag=conditional_flag
            )
        ]

    def get_color_palette(self, index: int) -> ColorPalette:
        choices = range(1, 9)
        weights = np.array([getattr(self.sampling_by_palette, f"palette_{i}") for i in choices])
        sum = weights.sum()
        probabilities = weights / sum
        palette_index = int(random.choices(choices, probabilities, k=1)[0])
        item = self.hf_dataset[index]
        palette: ColorPalette = item[f"palettes"][str(palette_index)]
        return palette

    def collate_fn(self, batch: list[TextEmbeddingColorPaletteLatentsBatch]) -> TextEmbeddingColorPaletteLatentsBatch:
        return [item for sublist in batch for item in sublist]
