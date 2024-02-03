import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from pydantic import BaseModel
from torch import Tensor, tensor, empty
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
    
@dataclass
class DatasetItem:
    palettes: dict[str, ColorPalette]
    image: Image.Image


TextEmbeddingColorPaletteLatentsBatch = List[ColorPaletteDatasetItem]

DEFAULT_SAMPLING= {
    "palette_1": 1.0,
    "palette_2": 2.0,
    "palette_3": 3.0,
    "palette_4": 4.0,
    "palette_5": 5.0,
    "palette_6": 6.0,
    "palette_7": 7.0,
    "palette_8": 8.0
}

class SamplingByPalette(BaseModel):
    def __init__(self, sampling = DEFAULT_SAMPLING) -> None:
        for key in sampling:
            setattr(self, key, sampling[key])
        super().__init__()


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
        
        item : DatasetItem = self.hf_dataset[index]

        resized_image = self.resize_image(
            image=item["image"],
            min_size=self.config.resize_image_min_size,
            max_size=self.config.resize_image_max_size,
        )

        image = self.process_image(resized_image)
        
        caption_key = self.config.caption_key
        caption = item[caption_key]
        (caption_processed, conditional_flag) = self.process_caption(caption)   
        
        return [
            ColorPaletteDatasetItem(
                color_palette=self.process_color_palette(item),
                text=caption_processed,
                image=image,
                conditional_flag=conditional_flag
            )
        ]
        
    def process_color_palette(self, item: DatasetItem) -> ColorPalette:
        choices = range(1, 9)
        weights_list : List[float] = []
        for i in choices:
            if hasattr(self.sampling_by_palette, f"palette_{i}"):
                weight = getattr(self.sampling_by_palette, f"palette_{i}")
                weights_list.append(weight)
        
        weights = np.array(weights_list)
        sum = weights.sum()
        probabilities = weights / sum
        palette_index = int(random.choices(choices, probabilities, k=1)[0])
        palette: ColorPalette = item[f"palettes"][str(palette_index)]
        
        return palette
    def get_color_palette(self, index: int) -> ColorPalette:
        item = self.hf_dataset[index]
        return self.process_color_palette(item)

    def collate_fn(self, batch: list[TextEmbeddingColorPaletteLatentsBatch]) -> TextEmbeddingColorPaletteLatentsBatch:
        return [item for sublist in batch for item in sublist]
