from refiners.training_utils.datasets.latent_diffusion import TextEmbeddingLatentsBaseDataset, TextEmbeddingLatentsBatch
from dataclasses import dataclass
from refiners.fluxion.adapters.color_palette import ColorPaletteEncoder
from torch import Tensor, cat
from pydantic import BaseModel

from refiners.training_utils.huggingface_datasets import HuggingfaceDatasetConfig
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder
from refiners.foundationals.clip.text_encoder import CLIPTextEncoder

@dataclass
class TextEmbeddingColorPaletteLatentsBatch(TextEmbeddingLatentsBatch):
    color_palette_embeddings: Tensor
    
class SamplingByPalette(BaseModel):
    palette_1: float = 1.0
    palette_2: float = 2.0
    palette_3: float = 3.0
    palette_4: float = 4.0
    palette_5: float = 5.0
    palette_6: float = 6.0
    palette_7: float = 7.0
    palette_8: float = 8.0
    
class ColorPaletteDataset(TextEmbeddingLatentsBaseDataset[TextEmbeddingLatentsBatch]):
    def __init__(self,
         config: HuggingfaceDatasetConfig, 
        lda: SD1Autoencoder, 
        text_encoder: CLIPTextEncoder,
        color_palette_encoder: ColorPaletteEncoder,
        sampling_by_palette : SamplingByPalette = SamplingByPalette(),
        unconditional_sampling_probability: float = 0.2
    ) -> None:
        self.color_palette_encoder = color_palette_encoder
        super().__init__(
            config=config, lda=lda, text_encoder=text_encoder, unconditional_sampling_probability=unconditional_sampling_probability
        )
        
    def __getitem__(self, index: int) -> TextEmbeddingLatentsBatch:
        latents = self.get_processed_latents(index)
        (clip_text_embedding, color_palette_embedding) = self.process_text_embedding_and_palette(index)
        self.color_palette_encoder(processed_palette)
        
        return TextEmbeddingColorPaletteLatentsBatch(
            text_embeddings=clip_text_embedding, 
            latents=latents, 
            color_palette_embeddings=color_palette_embedding
        )
    
    def process_text_embedding_and_palette(self, index: int) -> tuple[Tensor, Tensor]:
        caption = self.get_caption(index=index)
        
        rand_num = random.random()
		processed_caption = self.process_caption(caption=caption, rand_num=rand_num)
		
        return self.text_encoder(processed_caption)
        
        clip_text_embedding = self.text_encoder(item["caption"])
        color_palette_embedding = self.get_processed_palette(index)
        return (clip_text_embedding, color_palette_embedding)
    
    def get_processed_palette(self, index: int) -> Tensor:
        choices = range(1, 9)
        weights = np.array([
            getattr(self.config.color_palette.sampling_by_palette, f"palette_{i}") for i in choices
        ])
        sum = weights.sum()
        probabilities = weights / sum
        palette_index = int(random.choices(choices, probabilities, k=1)[0])
        item = self.dataset[index]
        palette = self.dataset[index][f"palette_{palette_index}"]
        return self.color_palette_encoder(processed_palette)

    def collate_fn(self, batch: list[TextEmbeddingColorPaletteLatentsBatch]) -> TextEmbeddingColorPaletteLatentsBatch:
        text_embeddings = cat(tensors=[item.text_embeddings for item in batch])
        latents = cat(tensors=[item.latents for item in batch])
        color_palette_embeddings = cat(tensors=[item.color_palette_embeddings for item in batch])
        return TextEmbeddingColorPaletteLatentsBatch(
            text_embeddings=text_embeddings, latents=latents, color_palette_embeddings=color_palette_embeddings
        )