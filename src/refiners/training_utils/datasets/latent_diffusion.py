
from refiners.training_utils.huggingface_datasets import HuggingfaceDataset, HuggingfaceDatasetConfig, load_hf_dataset
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder
from refiners.foundationals.clip.text_encoder import CLIPTextEncoder
from datasets import DownloadManager  # type: ignore
from refiners.training_utils.datasets.utils import resize_image

from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip  # type: ignore

from torch import cat, Tensor
from torch.nn import Module as TorchModule
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Any, Callable, TypeVar
from PIL import Image
from loguru import logger
import random
from abc import abstractmethod

@dataclass
class TextEmbeddingLatentsBatch:
	text_embeddings: Tensor
	latents: Tensor

BatchType = TypeVar("BatchType", bound=Any)


class TextEmbeddingLatentsBaseDataset(Dataset[BatchType]):
	def __init__(self, 
			config: HuggingfaceDatasetConfig, 
			lda: SD1Autoencoder, 
			text_encoder: CLIPTextEncoder,
			unconditional_sampling_probability: float = 0.2
		) -> None:
		self.config = config
		self.lda = lda
		self.text_encoder = text_encoder
		self.hf_dataset = self.load_huggingface_dataset()
		self.process_image = self.build_image_processor()
		self.download_manager = DownloadManager()
		self.unconditional_sampling_probability = unconditional_sampling_probability

		logger.info(f"Loaded {len(self.hf_dataset)} samples from dataset")

	def build_image_processor(self) -> Callable[[Image.Image], Image.Image]:
		# TODO: make this configurable and add other transforms
		transforms: list[TorchModule] = []
		if self.config.random_crop:
			transforms.append(RandomCrop(size=512))
		if self.config.horizontal_flip:
			transforms.append(RandomHorizontalFlip(p=0.5))
		if not transforms:
			return lambda image: image
		return Compose(transforms)

	def load_huggingface_dataset(self) -> HuggingfaceDataset[Any]:
		dataset_config = self.config
		logger.info(f"Loading dataset from {dataset_config.hf_repo} revision {dataset_config.revision}")
		dataset = load_hf_dataset(
			path=dataset_config.hf_repo, revision=dataset_config.revision, split=dataset_config.split
		)
		return dataset

	def resize_image(self, image: Image.Image, min_size: int = 512, max_size: int = 576) -> Image.Image:
		return resize_image(image=image, min_size=min_size, max_size=max_size)

	def process_caption(self, caption: str, rand_num: float | None = None) -> str:
		if rand_num is None:
			rand_num = random.random()
		return caption if rand_num > self.unconditional_sampling_probability else ""

	def get_caption(self, index: int) -> str:
		caption_key=self.config.caption_key or "caption"
		
		caption = self.hf_dataset[index][caption_key]
		if not isinstance(caption, str):
			raise RuntimeError(
				f"Dataset item at index [{index}] and caption_key [{caption_key}] does not contain a string caption"
			)
		return caption

	def get_image(self, index: int) -> Image.Image:
		if "image" in self.hf_dataset[index]:
			return self.hf_dataset[index]["image"]
		elif "url" in self.hf_dataset[index]:
			url: str = self.hf_dataset[index]["url"]
			filename: str = self.download_manager.download(url)  # type: ignore
			return Image.open(filename)
		else:
			raise RuntimeError(f"Dataset item at index [{index}] does not contain 'image' or 'url'")
	
	@abstractmethod
	def __getitem__(self, index: int) -> BatchType:
		...
	
	@abstractmethod
	def collate_fn(self, batch: list[TextEmbeddingLatentsBatch]) -> BatchType:
		...
	
	def get_processed_text_embedding(self, index: int) -> Tensor:
		caption = self.get_caption(index=index)
		processed_caption = self.process_caption(caption=caption)
		return self.text_encoder(processed_caption)

	def get_processed_latents(self, index: int) -> Tensor:
		image = self.get_image(index=index)
		resized_image = self.resize_image(
			image=image,
			min_size=self.config.resize_image_min_size,
			max_size=self.config.resize_image_max_size,
		)
		processed_image = self.process_image(resized_image)
		encoded_image = self.lda.encode_image(image=processed_image)
		return encoded_image

	def __len__(self) -> int:
		return len(self.hf_dataset)

class TextEmbeddingLatentsDataset(TextEmbeddingLatentsBaseDataset[TextEmbeddingLatentsBatch]):
	def __getitem__(self, index: int) -> TextEmbeddingLatentsBatch:
		clip_text_embedding = self.get_processed_text_embedding(index)
		latents = self.get_processed_latents(index)
		return TextEmbeddingLatentsBatch(text_embeddings=clip_text_embedding, latents=latents)

	def collate_fn(self, batch: list[TextEmbeddingLatentsBatch]) -> TextEmbeddingLatentsBatch:
		text_embeddings = cat(tensors=[item.text_embeddings for item in batch])
		latents = cat(tensors=[item.latents for item in batch])
		return TextEmbeddingLatentsBatch(text_embeddings=text_embeddings, latents=latents)
