import random
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, List

from datasets import DownloadManager  # type: ignore
from loguru import logger
from PIL import Image
from torch.nn import Module as TorchModule
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip  # type: ignore

from refiners.training_utils.datasets.utils import resize_image
from refiners.training_utils.huggingface_datasets import HuggingfaceDataset, HuggingfaceDatasetConfig, load_hf_dataset

@dataclass
class TextImageDatasetItem:
    text: str
    image: Image.Image

TextEmbeddingLatentsBatch = List[TextImageDatasetItem]

BatchType = TypeVar("BatchType", bound=List[Any])


class TextEmbeddingLatentsBaseDataset(Dataset[BatchType]):
    def __init__(
        self,
        config: HuggingfaceDatasetConfig,
        unconditional_sampling_probability: float = 0.2,
    ) -> None:
        self.config = config
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

    def process_caption(self, caption: str) -> tuple[str, bool]:
        conditional_flag = random.random() > self.unconditional_sampling_probability
        if conditional_flag:
            return (caption, conditional_flag)
        else:
            return ("", conditional_flag)

    def get_caption(self, index: int) -> str:
        caption_key = self.config.caption_key or "caption"

        caption = self.hf_dataset[index][caption_key]
        if not isinstance(caption, str):
            raise RuntimeError(
                f"Dataset item at index [{index}] and caption_key [{caption_key}] does not contain a string caption"
            )
        return caption

    def get_image(self, index: int) -> Image.Image:
        item = self.hf_dataset[index]
        if "image" in item:
            logger.info(f"get_image image {index}")
            return item["image"]
        elif "url" in item[index]:
            url: str = item[index]["url"]
            filename: str = self.download_manager.download(url)  # type: ignore
            return Image.open(filename)
        else:
            raise RuntimeError(f"Dataset item at index [{index}] does not contain 'image' or 'url'")

    @abstractmethod
    def get_hf_item(self, index: int) -> Any:
        return self.hf_dataset[index]

    @abstractmethod
    def __getitem__(self, index: int) -> BatchType:
        ...

    @abstractmethod
    def collate_fn(self, batch: list[BatchType]) -> BatchType:
        ...

    # def get_processed_text_embedding(self, index: int) -> Tensor:
    #     caption = self.get_caption(index=index)
    #     (processed_caption, _) = self.process_caption(caption=caption)
    #     return self.text_encoder(processed_caption)

    def get_processed_image(self, index: int) -> Image.Image:
        image = self.get_image(index=index)
        logger.info(f"resize_image image {index}")

        resized_image = self.resize_image(
            image=image,
            min_size=self.config.resize_image_min_size,
            max_size=self.config.resize_image_max_size,
        )
        logger.info(f"resized_image image {index}")
        
        return self.process_image(resized_image)

    def __len__(self) -> int:
        return len(self.hf_dataset)


class TextEmbeddingLatentsDataset(TextEmbeddingLatentsBaseDataset[TextEmbeddingLatentsBatch]):
    def __getitem__(self, index: int) -> TextEmbeddingLatentsBatch:        
        image = self.get_processed_image(index)
        (caption, _) = self.process_caption(self.get_caption(index))        
        
        return [
            TextImageDatasetItem(
                text=caption,
                image=image
            )
        ]

    def collate_fn(self, batch: list[TextEmbeddingLatentsBatch]) -> TextEmbeddingLatentsBatch:
        return [item for sublist in batch for item in sublist]
