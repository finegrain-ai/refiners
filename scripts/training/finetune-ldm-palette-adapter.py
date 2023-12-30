import random
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

from loguru import logger
from PIL import Image
from pydantic import BaseModel
from torch import Tensor, cat
from torch.nn import Module
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip

from refiners.training_utils.callback import Callback
from refiners.training_utils.huggingface_datasets import HuggingfaceDataset, HuggingfaceDatasetConfig, load_hf_dataset
from refiners.training_utils.latent_diffusion import (
    FinetuneLatentDiffusionConfig,
    LatentDiffusionConfig,
    LatentDiffusionTrainer,
    resize_image,
)
from refiners.training_utils.trainer import Trainer


class PaletteImage(TypedDict):
    """Structure of the data in the HuggingfaceDataset."""

    caption: str
    palette_colors: Tensor
    image: Image.Image


@dataclass
class PaletteBatch:
    """Structure of the batch for the palette adapter."""

    text_embeddings: Tensor
    palette_colors: Tensor
    latents: Tensor


class PaletteDataset(Dataset[PaletteBatch]):
    """Dataset for the palette adapter.

    Transforms PaletteImages from the HuggingfaceDataset into PaletteBatches.
    This would be called a PaletteDatamodule if we were using PyTorch Lightning.
    """

    def __init__(self, trainer: "Trainer[Any, PaletteBatch]") -> None:
        super().__init__()
        self.trainer = trainer
        self.dataset = self.load_huggingface_dataset()
        self.image_processor = self.build_image_processor()

    def load_huggingface_dataset(self) -> HuggingfaceDataset[PaletteImage]:
        dataset_config = self.trainer.config.dataset
        logger.info(f"Loading dataset from {dataset_config.hf_repo} revision {dataset_config.revision}")
        return load_hf_dataset(
            path=dataset_config.hf_repo,
            revision=dataset_config.revision,
            split=dataset_config.split,
        )

    def build_image_processor(self) -> Callable[[Image.Image], Image.Image]:
        transforms: list[Module] = []
        if self.trainer.config.dataset.random_crop:
            transforms.append(RandomCrop(size=512))
        if self.trainer.config.dataset.horizontal_flip:
            transforms.append(RandomHorizontalFlip(p=0.5))
        if not transforms:
            return lambda image: image
        return Compose(transforms)

    def resize_image(
        self,
        image: Image.Image,
        min_size: int = 512,
        max_size: int = 576,
    ) -> Image.Image:
        return resize_image(
            image=image,
            min_size=min_size,
            max_size=max_size,
        )

    def process_caption(self, caption: str) -> str:
        if random.random() > self.trainer.config.latent_diffusion.unconditional_sampling_probability:
            return caption
        else:
            return ""

    def get_caption(self, index: int) -> str:
        return self.dataset[index]["caption"]

    def get_image(self, index: int) -> Image.Image:
        return self.dataset[index]["image"]

    def get_palette_colors(self, index: int) -> Tensor:
        return self.dataset[index]["palette_colors"]

    def __getitem__(self, index: int) -> PaletteBatch:
        """Creates a PaletteBatch from the HuggingfaceDataset."""
        # retreive data from the dataset
        caption = self.get_caption(index=index)
        image = self.get_image(index=index)
        palette_colors = self.get_palette_colors(index=index)

        # process the image into latents
        resized_image = self.resize_image(
            image=image,
            min_size=self.trainer.config.dataset.resize_image_min_size,
            max_size=self.trainer.config.dataset.resize_image_max_size,
        )
        processed_image = self.image_processor(resized_image)
        latents = self.trainer.lda.encode_image(image=processed_image).to(device=self.trainer.device)

        # process the caption into text embeddings
        processed_caption = self.process_caption(caption=caption)
        clip_text_embedding = self.trainer.text_encoder(processed_caption).to(device=self.trainer.device)

        # the palette_colors stays as is, as we need to train the color_encoder with it

        return PaletteBatch(
            palette_colors=palette_colors,
            text_embeddings=clip_text_embedding,
            latents=latents,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def collate_fn(self, batch: list[PaletteBatch]) -> PaletteBatch:
        """Constructs a PaletteBatch from a list of PaletteBatch."""
        palette_colors = cat(tensors=[item.palette_colors for item in batch])
        text_embeddings = cat(tensors=[item.text_embeddings for item in batch])
        latents = cat(tensors=[item.latents for item in batch])

        return PaletteBatch(
            palette_colors=palette_colors,
            text_embeddings=text_embeddings,
            latents=latents,
        )


class ColorEncoderConfig(BaseModel):
    """Configuration for the color encoder (of the palette adapter)."""

    dim_sinusoids: int = 64
    dim_embeddings: int = 256


class AdapterConfig(BaseModel):
    """Configuration for the palette adapter."""

    color_encoder_config: ColorEncoderConfig


class AdapterLatentDiffusionConfig(FinetuneLatentDiffusionConfig):
    """Finetunning configuration.

    contains the configuration of the ldm, adapter and dataset configs.
    """

    dataset: HuggingfaceDatasetConfig
    ldm: LatentDiffusionConfig
    adapter: AdapterConfig

    def model_post_init(self, _context: Any) -> None:
        """Pydantic v2 does post init differently, so we need to override this method too."""
        logger.info("Freezing models to train only the adapter.")
        self.models["unet"].train = False
        self.models["text_encoder"].train = False
        self.models["lda"].train = False


class AdapterLatentDiffusionTrainer(Trainer[AdapterLatentDiffusionConfig, PaletteBatch]):
    # TODO

    def __init__(
        self,
        config: AdapterLatentDiffusionConfig,
        callbacks: "list[Callback[Any]] | None" = None,
    ) -> None:
        super().__init__(config=config, callbacks=callbacks)
        self.callbacks.extend((LoadAdapter(), SaveAdapter()))

    def load_dataset(self) -> PaletteDataset:
        return PaletteDataset(trainer=self)


class LoadAdapter(Callback[AdapterLatentDiffusionTrainer]):
    """Callback to load the adapter at the beginning of the training."""

    pass


class SaveAdapter(Callback[AdapterLatentDiffusionTrainer]):
    """Callback to save the adapter when a checkpoint is saved."""

    pass


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1]
    config = AdapterLatentDiffusionConfig.load_from_toml(toml_path=config_path)
    trainer = AdapterLatentDiffusionTrainer(config=config)
    trainer.train()
