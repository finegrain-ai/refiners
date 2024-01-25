from refiners.training_utils.datasets.latent_diffusion import TextEmbeddingLatentsBaseDataset, TextEmbeddingLatentsBatch
from dataclasses import dataclass

@dataclass
class TextEmbeddingColorPaletteLatentsBatch(TextEmbeddingLatentsBatch):
    text_embeddings: Tensor
    latents: Tensor
    color_palette_embeddings: Tensor

class ColorPaletteDataset(TextEmbeddingLatentsBaseDataset[TextEmbeddingLatentsBatch]):
	def __init__(
        *args,
        **kwargs,
    ) -> None:
        self.color_palette_encoder = kwargs.pop("color_palette_encoder")
        super().__init__(*args, **kwargs)
        
	def __getitem__(self, index: int) -> TextEmbeddingLatentsBatch:
     	clip_text_embedding = self.get_processed_text_embedding(index)
		latents = self.get_processed_latents(index)
		self.color_palette_encoder(processed_palette)
        
        return TextEmbeddingColorPaletteLatentsBatch(
            text_embeddings=clip_text_embedding, 
            latents=latents, 
            color_palette_embeddings=color_palette_embedding
        )
    
    def empty_text_embedding(self):
        return self.text_encoder("")
    
    def process_text_embedding_and_palette(self, clip_text_embedding: Tensor, color_palette: Tensor) -> tuple[Tensor, Tensor]:
        if random.random() < self.config.latent_diffusion.unconditional_sampling_probability:
            empty = color_palette[:,0:0,:]
            return (self.empty_text_embedding, empty)
        if random.random() < self.config.color_palette.without_caption_probability:
            return (self.empty_text_embedding, color_palette)
        return (clip_text_embedding, color_palette)

    def collate_fn(self, batch: list[TextEmbeddingColorPaletteLatentsBatch]) -> TextEmbeddingColorPaletteLatentsBatch:
        text_embeddings = cat(tensors=[item.text_embeddings for item in batch])
        latents = cat(tensors=[item.latents for item in batch])
        color_palette_embeddings = cat(tensors=[item.color_palette_embeddings for item in batch])
        return TextEmbeddingColorPaletteLatentsBatch(
            text_embeddings=text_embeddings, latents=latents, color_palette_embeddings=color_palette_embeddings
        )


class ColorPaletteDataset(TextEmbeddingLatentsBaseDataset[TextEmbeddingColorPaletteLatentsBatch]):
    def __init__(
        *args,
        **kwargs,
    ) -> None:
        self.color_palette_encoder = kwargs.pop("color_palette_encoder")
        super().__init__(*args, **kwargs)
    
    def get_color_palette(self, index: int) -> ColorPalette:
        # Randomly pick a palette between 1 and 8
        choices = range(1, 9)
        weights = np.array([
            getattr(self.config.color_palette.sampling_by_palette, f"palette_{i}") for i in choices
        ])
        sum = weights.sum()
        probabilities = weights / sum
        palette_index = int(random.choices(choices, probabilities, k=1)[0])
        item = self.dataset[index]
        return self.dataset[index][f"palette_{palette_index}"]
    
    def process_text_embedding_and_palette(self, clip_text_embedding: Tensor, color_palette: Tensor) -> tuple[Tensor, Tensor]:
        if random.random() < self.config.latent_diffusion.unconditional_sampling_probability:
            empty = color_palette[:,0:0,:]
            return (self.empty_text_embedding, empty)
        if random.random() < self.config.color_palette.without_caption_probability:
            return (self.empty_text_embedding, color_palette)
        return (clip_text_embedding, color_palette)
    
    @cached_property
    def empty_text_embedding(self):
        return self.text_encoder("")
    
    def __getitem__(self, index: int) -> TextEmbeddingColorPaletteLatentsBatch:
        color_palette = tensor([self.get_color_palette(index=index)], dtype=self.trainer.dtype)
        # image = self.get_image(index=index)
        # resized_image = self.resize_image(
        #     image=image,
        #     min_size=self.config.dataset.resize_image_min_size,
        #     max_size=self.config.dataset.resize_image_max_size,
        # )
        # processed_image = self.process_image(resized_image)
        item = self.dataset[index]
        latents = tensor(item["latents"])
        clip_text_embedding = tensor(item["clip_text_embedding"])

        (processed_caption, processed_palette) = self.process_text_embedding_and_palette(clip_text_embedding=clip_text_embedding, color_palette=color_palette)

        color_palette_embedding = self.color_palette_encoder(processed_palette)
        
        return TextEmbeddingColorPaletteLatentsBatch(
            text_embeddings=clip_text_embedding, 
            latents=latents, 
            color_palette_embeddings=color_palette_embedding
        )

    def collate_fn(self, batch: list[TextEmbeddingColorPaletteLatentsBatch]) -> TextEmbeddingColorPaletteLatentsBatch:
        text_embeddings = cat(tensors=[item.text_embeddings for item in batch])
        latents = cat(tensors=[item.latents for item in batch])
        color_palette_embeddings = cat(tensors=[item.color_palette_embeddings for item in batch])
        return TextEmbeddingColorPaletteLatentsBatch(
            text_embeddings=text_embeddings, latents=latents, color_palette_embeddings=color_palette_embeddings
        )

