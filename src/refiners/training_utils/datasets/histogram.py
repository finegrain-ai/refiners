from dataclasses import dataclass

from torch import Tensor, cat

from refiners.fluxion.adapters.histogram import HistogramEncoder, HistogramExtractor
from refiners.fluxion.utils import image_to_tensor
from refiners.foundationals.clip.text_encoder import CLIPTextEncoder
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder
from refiners.training_utils.datasets.color_palette import ColorPalette
from refiners.training_utils.datasets.latent_diffusion import TextEmbeddingLatentsBaseDataset, TextEmbeddingLatentsBatch
from refiners.training_utils.huggingface_datasets import HuggingfaceDatasetConfig


@dataclass
class TextEmbeddingHistogramLatentsBatch(TextEmbeddingLatentsBatch):
    histogram_embeddings: Tensor


class HistogramLatentsDataset(TextEmbeddingLatentsBaseDataset[TextEmbeddingHistogramLatentsBatch]):
    def __init__(
        self,
        config: HuggingfaceDatasetConfig,
        histogram_encoder: HistogramEncoder,
        histogram_extractor: HistogramExtractor,
        unconditional_sampling_probability: float = 0.2,
    ) -> None:
        self.histogram_encoder = histogram_encoder
        self.histogram_extractor = histogram_extractor
        super().__init__(
            config=config,
            unconditional_sampling_probability=unconditional_sampling_probability,
        )

    def __getitem__(self, index: int) -> TextEmbeddingHistogramLatentsBatch:
        (latents, image) = self.get_processed_latents(index)
        caption = self.get_caption(index=index)
        (processed_caption, conditionnal_flag) = self.process_caption(caption=caption)

        if conditionnal_flag:
            histogram_embedding = self.histogram_encoder(self.histogram_extractor(image_to_tensor(image)))
        else:
            # TODO we should do something better here
            # like empty histogram of something
            histogram_embedding = self.histogram_encoder(self.histogram_extractor(image_to_tensor(image)))

        clip_text_embedding = self.text_encoder(processed_caption)
        return TextEmbeddingHistogramLatentsBatch(
            text_embeddings=clip_text_embedding, latents=latents, histogram_embeddings=histogram_embedding
        )

    def get_palette(self, index: int, palette_index: int = 8) -> ColorPalette:
        item = self.hf_dataset[index]
        return item[f"palette_{palette_index}"]

    def collate_fn(self, batch: list[TextEmbeddingHistogramLatentsBatch]) -> TextEmbeddingHistogramLatentsBatch:
        text_embeddings = cat(tensors=[item.text_embeddings for item in batch])
        latents = cat(tensors=[item.latents for item in batch])
        histogram_embeddings = cat(tensors=[item.histogram_embeddings for item in batch])
        return TextEmbeddingHistogramLatentsBatch(
            text_embeddings=text_embeddings, latents=latents, histogram_embeddings=histogram_embeddings
        )
