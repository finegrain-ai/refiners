import random
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, TypedDict, TypeVar

from loguru import logger
from PIL import Image
from pydantic import BaseModel
from torch import Generator, Tensor, cat, device as Device, dtype as DType, randn
from torch.nn import Module
from torch.nn.functional import mse_loss
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip  # type: ignore

import refiners.fluxion.layers as fl
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
from refiners.foundationals.latent_diffusion import (
    DPMSolver,
    SD1UNet,
    StableDiffusion_1,
)
from refiners.foundationals.latent_diffusion.solvers import DDPM, Solver
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder
from refiners.training_utils.callback import Callback
from refiners.training_utils.config import BaseConfig
from refiners.training_utils.huggingface_datasets import HuggingfaceDataset, HuggingfaceDatasetConfig, load_hf_dataset
from refiners.training_utils.trainer import Trainer
from refiners.training_utils.wandb import WandbLoggable


class LatentDiffusionConfig(BaseModel):
    unconditional_sampling_probability: float = 0.2
    offset_noise: float = 0.1
    min_step: int = 0
    max_step: int = 999


class TestDiffusionConfig(BaseModel):
    seed: int = 0
    num_inference_steps: int = 30
    use_short_prompts: bool = False
    prompts: list[str] = []
    num_images_per_prompt: int = 1


class FinetuneLatentDiffusionConfig(BaseConfig):
    dataset: HuggingfaceDatasetConfig
    latent_diffusion: LatentDiffusionConfig
    test_diffusion: TestDiffusionConfig


@dataclass
class TextEmbeddingLatentsBatch:
    text_embeddings: Tensor
    latents: Tensor


class CaptionImage(TypedDict):
    caption: str
    image: Image.Image


ConfigType = TypeVar("ConfigType", bound=FinetuneLatentDiffusionConfig)


class TextEmbeddingLatentsDataset(Dataset[TextEmbeddingLatentsBatch]):
    def __init__(self, trainer: "LatentDiffusionTrainer[Any]") -> None:
        self.trainer = trainer
        self.config = trainer.config
        self.device = self.trainer.device
        self.lda = self.trainer.lda
        self.text_encoder = self.trainer.text_encoder
        self.dataset = self.load_huggingface_dataset()
        self.process_image = self.build_image_processor()
        logger.info(f"Loaded {len(self.dataset)} samples from dataset")

    def build_image_processor(self) -> Callable[[Image.Image], Image.Image]:
        # TODO: make this configurable and add other transforms
        transforms: list[Module] = []
        if self.config.dataset.random_crop:
            transforms.append(RandomCrop(size=512))
        if self.config.dataset.horizontal_flip:
            transforms.append(RandomHorizontalFlip(p=0.5))
        if not transforms:
            return lambda image: image
        return Compose(transforms)

    def load_huggingface_dataset(self) -> HuggingfaceDataset[CaptionImage]:
        dataset_config = self.config.dataset
        logger.info(f"Loading dataset from {dataset_config.hf_repo} revision {dataset_config.revision}")
        return load_hf_dataset(
            path=dataset_config.hf_repo, revision=dataset_config.revision, split=dataset_config.split
        )

    def resize_image(self, image: Image.Image, min_size: int = 512, max_size: int = 576) -> Image.Image:
        return resize_image(image=image, min_size=min_size, max_size=max_size)

    def process_caption(self, caption: str) -> str:
        return caption if random.random() > self.config.latent_diffusion.unconditional_sampling_probability else ""

    def get_caption(self, index: int) -> str:
        return self.dataset[index]["caption"]

    def get_image(self, index: int) -> Image.Image:
        return self.dataset[index]["image"]

    def __getitem__(self, index: int) -> TextEmbeddingLatentsBatch:
        caption = self.get_caption(index=index)
        image = self.get_image(index=index)
        resized_image = self.resize_image(
            image=image,
            min_size=self.config.dataset.resize_image_min_size,
            max_size=self.config.dataset.resize_image_max_size,
        )
        processed_image = self.process_image(resized_image)
        latents = self.lda.image_to_latents(image=processed_image).to(device=self.device)
        processed_caption = self.process_caption(caption=caption)
        clip_text_embedding = self.text_encoder(processed_caption).to(device=self.device)
        return TextEmbeddingLatentsBatch(text_embeddings=clip_text_embedding, latents=latents)

    def collate_fn(self, batch: list[TextEmbeddingLatentsBatch]) -> TextEmbeddingLatentsBatch:
        text_embeddings = cat(tensors=[item.text_embeddings for item in batch])
        latents = cat(tensors=[item.latents for item in batch])
        return TextEmbeddingLatentsBatch(text_embeddings=text_embeddings, latents=latents)

    def __len__(self) -> int:
        return len(self.dataset)


class LatentDiffusionTrainer(Trainer[ConfigType, TextEmbeddingLatentsBatch]):
    @cached_property
    def unet(self) -> SD1UNet:
        assert self.config.models["unet"] is not None, "The config must contain a unet entry."
        return SD1UNet(in_channels=4, device=self.device).to(device=self.device)

    @cached_property
    def text_encoder(self) -> CLIPTextEncoderL:
        assert self.config.models["text_encoder"] is not None, "The config must contain a text_encoder entry."
        return CLIPTextEncoderL(device=self.device).to(device=self.device)

    @cached_property
    def lda(self) -> SD1Autoencoder:
        assert self.config.models["lda"] is not None, "The config must contain a lda entry."
        return SD1Autoencoder(device=self.device).to(device=self.device)

    def load_models(self) -> dict[str, fl.Module]:
        return {"unet": self.unet, "text_encoder": self.text_encoder, "lda": self.lda}

    def load_dataset(self) -> Dataset[TextEmbeddingLatentsBatch]:
        return TextEmbeddingLatentsDataset(trainer=self)

    @cached_property
    def ddpm_solver(self) -> Solver:
        return DDPM(
            num_inference_steps=1000,
            device=self.device,
        ).to(device=self.device)

    def sample_timestep(self) -> Tensor:
        random_step = random.randint(a=self.config.latent_diffusion.min_step, b=self.config.latent_diffusion.max_step)
        self.current_step = random_step
        return self.ddpm_solver.timesteps[random_step].unsqueeze(dim=0)

    def sample_noise(self, size: tuple[int, ...], dtype: DType | None = None) -> Tensor:
        return sample_noise(
            size=size, offset_noise=self.config.latent_diffusion.offset_noise, device=self.device, dtype=dtype
        )

    def compute_loss(self, batch: TextEmbeddingLatentsBatch) -> Tensor:
        clip_text_embedding, latents = batch.text_embeddings, batch.latents
        timestep = self.sample_timestep()
        noise = self.sample_noise(size=latents.shape, dtype=latents.dtype)
        noisy_latents = self.ddpm_solver.add_noise(x=latents, noise=noise, step=self.current_step)
        self.unet.set_timestep(timestep=timestep)
        self.unet.set_clip_text_embedding(clip_text_embedding=clip_text_embedding)
        prediction = self.unet(noisy_latents)
        loss = mse_loss(input=prediction, target=noise)
        return loss

    def compute_evaluation(self) -> None:
        sd = StableDiffusion_1(
            unet=self.unet,
            lda=self.lda,
            clip_text_encoder=self.text_encoder,
            solver=DPMSolver(num_inference_steps=self.config.test_diffusion.num_inference_steps),
            device=self.device,
        )
        prompts = self.config.test_diffusion.prompts
        num_images_per_prompt = self.config.test_diffusion.num_images_per_prompt
        if self.config.test_diffusion.use_short_prompts:
            prompts = [prompt.split(sep=",")[0] for prompt in prompts]
        images: dict[str, WandbLoggable] = {}
        for prompt in prompts:
            canvas_image: Image.Image = Image.new(mode="RGB", size=(512, 512 * num_images_per_prompt))
            for i in range(num_images_per_prompt):
                logger.info(f"Generating image {i+1}/{num_images_per_prompt} for prompt: {prompt}")
                x = randn(1, 4, 64, 64, device=self.device)
                clip_text_embedding = sd.compute_clip_text_embedding(text=prompt).to(device=self.device)
                for step in sd.steps:
                    x = sd(
                        x,
                        step=step,
                        clip_text_embedding=clip_text_embedding,
                    )
                canvas_image.paste(sd.lda.latents_to_image(x=x), box=(0, 512 * i))
            images[prompt] = canvas_image
        self.log(data=images)


def sample_noise(
    size: tuple[int, ...],
    offset_noise: float = 0.1,
    device: Device | str = "cpu",
    dtype: DType | None = None,
    generator: Generator | None = None,
) -> Tensor:
    """Sample noise from a normal distribution.

    If `offset_noise` is more than 0, the noise will be offset by a small amount. It allows the model to generate
    images with a wider range of contrast https://www.crosslabs.org/blog/diffusion-with-offset-noise.
    """
    device = Device(device)
    noise = randn(*size, generator=generator, device=device, dtype=dtype)
    return noise + offset_noise * randn(*size[:2], 1, 1, generator=generator, device=device, dtype=dtype)


def resize_image(image: Image.Image, min_size: int = 512, max_size: int = 576) -> Image.Image:
    image_min_size = min(image.size)
    if image_min_size > max_size:
        if image_min_size == image.size[0]:
            image = image.resize(size=(max_size, int(max_size * image.size[1] / image.size[0])))
        else:
            image = image.resize(size=(int(max_size * image.size[0] / image.size[1]), max_size))
    if image_min_size < min_size:
        if image_min_size == image.size[0]:
            image = image.resize(size=(min_size, int(min_size * image.size[1] / image.size[0])))
        else:
            image = image.resize(size=(int(min_size * image.size[0] / image.size[1]), min_size))
    return image


class MonitorTimestepLoss(Callback[LatentDiffusionTrainer[Any]]):
    def on_train_begin(self, trainer: LatentDiffusionTrainer[Any]) -> None:
        self.timestep_bins: dict[int, list[float]] = {i: [] for i in range(10)}

    def on_compute_loss_end(self, trainer: LatentDiffusionTrainer[Any]) -> None:
        loss_value = trainer.loss.detach().cpu().item()
        current_step = trainer.current_step
        bin_index = min(current_step // 100, 9)
        self.timestep_bins[bin_index].append(loss_value)

    def on_epoch_end(self, trainer: LatentDiffusionTrainer[Any]) -> None:
        log_data: dict[str, WandbLoggable] = {}
        for bin_index, losses in self.timestep_bins.items():
            if losses:
                avg_loss = sum(losses) / len(losses)
                log_data[f"average_loss_timestep_bin_{bin_index * 100}"] = avg_loss
                self.timestep_bins[bin_index] = []

        trainer.log(data=log_data)
