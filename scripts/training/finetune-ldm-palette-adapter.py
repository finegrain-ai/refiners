import random
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import datasets
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from torch import Tensor, cat, device as Device, dtype as DType, randn
from torch.distributions import Beta
from torch.nn import Module
from torch.nn.functional import mse_loss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip

import refiners.fluxion.layers as fl
from refiners.fluxion.utils import save_to_safetensors
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
from refiners.foundationals.latent_diffusion.palette_adapter import ColorEncoder, PaletteAdapter
from refiners.foundationals.latent_diffusion.schedulers.ddpm import DDPM
from refiners.foundationals.latent_diffusion.schedulers.dpm_solver import DPMSolver
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder, SD1UNet, StableDiffusion_1
from refiners.training_utils.callback import Callback
from refiners.training_utils.config import BaseConfig
from refiners.training_utils.latent_diffusion import (
    LatentDiffusionConfig,
    TestDiffusionConfig,
    resize_image,
    sample_noise,
)
from refiners.training_utils.trainer import Trainer
from refiners.training_utils.wandb import WandbLoggable

# some images of the unsplash lite dataset are bigger than the default limit
Image.MAX_IMAGE_PIXELS = 200_000_000


class ColorEncoderConfig(BaseModel):
    """Configuration for the color encoder (of the palette adapter)."""

    dim_sinusoids: int = 64
    dim_embeddings: int = 256


class AdapterConfig(BaseModel):
    """Configuration for the palette adapter."""

    color_encoder: ColorEncoderConfig
    scale: float = 1.0


class DatasetConfig(BaseModel):
    """Configuration for the dataset."""

    hf_repo: str
    revision: str = "main"
    split: str = "train"
    horizontal_flip_probability: float = 0.5
    resize_image_min_size: int = 512
    resize_image_max_size: int = 576
    random_crop_size: int = 512


class TestPaletteDiffusionConfig(TestDiffusionConfig):
    """Configuration to test the diffusion model, during the `evaluation` loop of the trainer."""

    palettes: list[list[tuple[int, int, int]]] = []


class AdapterLatentDiffusionConfig(BaseConfig):
    """Finetunning configuration.

    Contains the configs of the dataset, the latent diffusion model and the adapter.
    """

    dataset: DatasetConfig
    ldm: LatentDiffusionConfig
    adapter: AdapterConfig
    test_ldm: TestPaletteDiffusionConfig


@dataclass
class PaletteBatch:
    """Structure of the data in the PaletteDataset."""

    latent: Tensor
    text_embedding: Tensor
    palette_embedding: Tensor


class PaletteDataset(Dataset[PaletteBatch]):
    """Dataset for the palette adapter.

    Transforms the data from the HuggingFace dataset into `PaletteBatch`.
    The `collate_fn` is used by the trainer to batch the data.
    """

    def __init__(self, trainer: "AdapterLatentDiffusionTrainer") -> None:
        super().__init__()
        self.trainer = trainer
        self.dataset = self.load_huggingface_dataset()

    @staticmethod
    def download_images(
        urls: list[Any],
        dl_manager: datasets.DownloadManager,
    ) -> dict[str, list[str]]:
        """Download the images from the urls."""
        return {
            "image": dl_manager.download(urls),  # type: ignore
        }

    @staticmethod
    def resize_images(
        images: list[Image.Image],
        min_size: int = 512,
        max_size: int = 576,
    ) -> dict[str, list[Image.Image]]:
        """Resize the images such that their shortest side is between `min_size` and `max_size`."""
        return {
            "image": [
                resize_image(
                    image=image,
                    min_size=min_size,
                    max_size=max_size,
                )
                for image in images
            ],
        }

    @staticmethod
    def encode_captions(
        captions: list[str],
        text_encoder: CLIPTextEncoderL,
    ) -> dict[str, list[Tensor]]:
        """Encode the captions with the text encoder."""
        return {
            "text_embedding": [text_encoder(caption) for caption in captions],
        }

    def load_huggingface_dataset(self) -> datasets.Dataset:
        """Load the dataset from Hugging Face and apply some pre-processing."""
        dataset_config = self.trainer.config.dataset
        logger.info(
            f"Loading dataset from {dataset_config.hf_repo}, "
            f"revision {dataset_config.revision}, "
            f"split {dataset_config.split}"
        )
        dataset = datasets.load_dataset(  # type: ignore
            path=dataset_config.hf_repo,
            revision=dataset_config.revision,
            split=dataset_config.split,
        )
        dataset = dataset.select(list(range(100)))  # type: ignore # FIXME: temporary

        # download images from urls
        dl_manager = datasets.DownloadManager()  # TODO: add a DownloadConfig
        dataset = dataset.map(  # type: ignore
            function=self.download_images,
            input_columns=["url"],
            remove_columns=["url"],
            batched=True,
            num_proc=8,  # FIXME: harcoded value
            fn_kwargs={
                "dl_manager": dl_manager,
            },
            desc="Downloading images",  # type: ignore
        )

        # cast the "image" column to Image feature type
        dataset = dataset.cast_column(  # type: ignore
            column="image",
            feature=datasets.Image(),
        )

        # limit max image size
        dataset = dataset.map(  # type: ignore
            function=self.resize_images,
            input_columns=["image"],
            batched=True,
            batch_size=10,  # FIXME: harcoded value
            num_proc=8,  # FIXME: harcoded value
            fn_kwargs={
                "min_size": dataset_config.resize_image_min_size,
                "max_size": dataset_config.resize_image_max_size,
            },
            desc="Capping image sizes",  # type: ignore
        )

        # encode the captions into text embedding
        self.trainer.prepare_model("text_encoder")
        dataset = dataset.rename_column("ai_description", "caption")  # type: ignore
        dataset = dataset.map(  # type: ignore
            function=self.encode_captions,
            input_columns=["caption"],
            remove_columns=["caption"],
            batched=True,
            batch_size=50,  # FIXME: harcoded value
            fn_kwargs={
                "text_encoder": self.trainer.text_encoder  # weights must be loaded to get same hash everytime
            },
            desc="Encoding captions into embeddings",  # type: ignore
        )

        # convert entries to torch tensors, except the image
        dataset.set_format(  # type: ignore
            type="torch",
            output_all_columns=True,
            columns=[
                "text_embedding",
                "palettes",
            ],
        )

        return dataset  # type: ignore

    @cached_property
    def empty_text_embedding(self) -> Tensor:
        """Return an empty text embedding."""
        return self.trainer.text_encoder("")

    def transform(self, data: dict[str, Any]) -> PaletteBatch:
        """Apply transforms to data."""
        # apply augmentation to the image
        image_transforms: list[Module] = []
        if self.trainer.config.dataset.random_crop_size:
            image_transforms.append(
                RandomCrop(size=self.trainer.config.dataset.random_crop_size),
            )
        if self.trainer.config.dataset.horizontal_flip_probability:
            image_transforms.append(
                RandomHorizontalFlip(p=self.trainer.config.dataset.horizontal_flip_probability),
            )
        image_compose = Compose(image_transforms)
        image = image_compose(data["image"])  # type: ignore

        # encode the image into latent
        latent = self.trainer.lda.encode_image(image=image)  # type: ignore

        # randomly drop the text (cfg)
        if random.random() < self.trainer.config.ldm.unconditional_sampling_probability:
            text_embedding = self.empty_text_embedding
        else:
            text_embedding = data["text_embedding"]

        # randomly select a palette size
        beta = Beta(2, 2)  # FIXME: harcoded value (2, 2)
        n = int(beta.sample() * 8) + 1  # FIXME: harcoded value (8)
        palette = data["palettes"][str(n)]  # (n, 3) palette

        # randomly drop the palette (cfg)
        if random.random() < self.trainer.config.ldm.unconditional_sampling_probability:
            palette = Tensor(size=(0, palette.size(1)))

        # encoder palette colors to embeddings
        palette_embedding = self.trainer.color_encoder(palette)

        return PaletteBatch(
            latent=latent,
            text_embedding=text_embedding,
            palette_embedding=palette_embedding,
        )

    def __getitem__(self, index: int) -> PaletteBatch:
        # retreive data from the huggingface dataset
        data = self.dataset[index]  # type: ignore
        # augment/transform into PaletteBatch
        data = self.transform(data)  # type: ignore
        return data

    def collate_fn(self, batch: list[PaletteBatch]) -> PaletteBatch:
        latents = cat(tensors=[item.latent for item in batch])
        text_embeddings = cat(tensors=[item.text_embedding for item in batch])
        palette_embeddings = pad_sequence([item.palette_embedding for item in batch], batch_first=True)
        return PaletteBatch(
            latent=latents,
            text_embedding=text_embeddings,
            palette_embedding=palette_embeddings,
        )

    def __len__(self) -> int:
        return len(self.dataset)


class AdapterLatentDiffusionTrainer(Trainer[AdapterLatentDiffusionConfig, PaletteBatch]):
    @cached_property
    def device(self) -> Device:  # TODO: remove, temporary
        selected_device = Device("cpu")
        logger.info(f"Using device: {selected_device}")
        return selected_device

    @cached_property
    def lda(self) -> SD1Autoencoder:
        assert self.config.models["lda"] is not None, "The config must contain a lda entry."
        return SD1Autoencoder(
            device=self.device,
        ).to(device=self.device)

    @cached_property
    def unet(self) -> SD1UNet:
        assert self.config.models["unet"] is not None, "The config must contain a unet entry."
        return SD1UNet(
            in_channels=4,  # FIXME: harcoded value
            device=self.device,
        ).to(device=self.device)

    @cached_property
    def text_encoder(self) -> CLIPTextEncoderL:
        assert self.config.models["text_encoder"] is not None, "The config must contain a text_encoder entry."
        return CLIPTextEncoderL(
            device=self.device,
        ).to(device=self.device)

    @cached_property
    def color_encoder(self) -> ColorEncoder:
        assert self.config.models["color_encoder"] is not None, "The config must contain a color_encoder entry."
        return ColorEncoder(
            dim_sinusoids=self.config.adapter.color_encoder.dim_sinusoids,
            dim_embeddings=self.config.adapter.color_encoder.dim_embeddings,
            device=self.device,
        ).to(device=self.device)

    @cached_property
    def adapter(self) -> PaletteAdapter[SD1UNet]:
        assert self.config.models["adapter"] is not None, "The config must contain a adapter entry."
        return PaletteAdapter(
            target=self.unet,
            color_encoder=self.color_encoder,
            scale=self.config.adapter.scale,
        )

    @cached_property
    def ddpm_scheduler(self) -> DDPM:
        return DDPM(
            num_inference_steps=1000,  # FIXME: harcoded value
            device=self.device,
        ).to(device=self.device)

    def load_models(self) -> dict[str, fl.Module]:
        return {
            "lda": self.lda,
            "unet": self.unet,
            "text_encoder": self.text_encoder,
            "color_encoder": self.color_encoder,
            "adapter": self.adapter,
        }

    def load_dataset(self) -> PaletteDataset:
        return PaletteDataset(trainer=self)

    def sample_timestep(self) -> Tensor:
        random_step = random.randint(
            a=self.config.ldm.min_step,
            b=self.config.ldm.max_step,
        )
        self.current_step = random_step
        return self.ddpm_scheduler.timesteps[random_step].unsqueeze(dim=0)

    def sample_noise(self, size: tuple[int, ...], dtype: DType | None = None) -> Tensor:
        return sample_noise(
            size=size,
            offset_noise=self.config.ldm.offset_noise,
            device=self.device,
            dtype=dtype,
        )

    def compute_loss(self, batch: PaletteBatch) -> Tensor:
        # retreive data from batch
        latents = batch.latent
        text_embeddings = batch.text_embedding
        palette_embeddings = batch.palette_embedding

        # set palette embeddings context
        self.adapter.set_palette_embeddings(embeddings=palette_embeddings)

        # set text embeddings context
        self.unet.set_clip_text_embedding(clip_text_embedding=text_embeddings)

        # sample timestep and set unet timestep context
        timestep = self.sample_timestep()
        self.unet.set_timestep(timestep=timestep)

        # sample noise and noisify the latents
        noise = self.sample_noise(size=latents.shape, dtype=latents.dtype)
        noisy_latents = self.ddpm_scheduler.add_noise(x=latents, noise=noise, step=self.current_step)

        # get prediction from unet
        prediction = self.unet(noisy_latents)

        # compute mse loss
        loss = mse_loss(input=prediction, target=noise)

        return loss

    def compute_evaluation(self) -> None:
        # initialize an SD1.5 pipeline using the trainer's models
        sd = StableDiffusion_1(
            unet=self.unet,
            lda=self.lda,
            clip_text_encoder=self.text_encoder,
            scheduler=DPMSolver(num_inference_steps=self.config.test_ldm.num_inference_steps),
            device=self.device,
        )

        # retreive data from config
        prompts = self.config.test_ldm.prompts
        palettes = [Tensor(palette) for palette in self.config.test_ldm.palettes]
        num_images_per_prompt = self.config.test_ldm.num_images_per_prompt
        if self.config.test_ldm.use_short_prompts:
            prompts = [prompt.split(sep=",")[0] for prompt in prompts]

        # for each prompt generate `num_images_per_prompt` images
        # TODO: remove this for loop, batch things up
        images: dict[str, WandbLoggable] = {}
        for prompt, palette in zip(prompts, palettes):
            canvas_image = Image.new(mode="RGB", size=(512, 512 * num_images_per_prompt))
            for i in range(num_images_per_prompt):
                logger.info(f"Generating image {i+1}/{num_images_per_prompt} for prompt: {prompt}")
                x = randn(1, 4, 64, 64, device=self.device)
                clip_text_embedding = sd.compute_clip_text_embedding(text=prompt).to(device=self.device)
                palette_embedding = self.adapter.compute_palette_embeddings(palettes=[palette])
                self.adapter.set_palette_embeddings(embeddings=palette_embedding)
                for step in sd.steps:
                    print(step)
                    x = sd(
                        x=x,
                        step=step,
                        clip_text_embedding=clip_text_embedding,
                    )
                canvas_image.paste(sd.lda.decode_latents(x=x), box=(0, 512 * i))
            images[prompt] = canvas_image

        # log images to wandb
        self.log(data=images)

    def __init__(
        self,
        config: AdapterLatentDiffusionConfig,
        callbacks: "list[Callback[Any]] | None" = None,
    ) -> None:
        super().__init__(config=config, callbacks=callbacks)
        self.callbacks.extend((LoadAdapter(), SaveAdapter()))


class LoadAdapter(Callback[AdapterLatentDiffusionTrainer]):
    """Callback to load the adapter at the beginning of the training."""

    def on_train_begin(self, trainer: AdapterLatentDiffusionTrainer) -> None:
        trainer.adapter.inject()


class SaveAdapter(Callback[AdapterLatentDiffusionTrainer]):
    """Callback to save the adapter when a checkpoint is saved."""

    def on_checkpoint_save(self, trainer: AdapterLatentDiffusionTrainer) -> None:
        color_encoder = trainer.color_encoder
        cross_attention_adapters = trainer.adapter.cross_attention_adapters

        tensors: dict[str, Tensor] = {}
        tensors |= {f"ColorEncoder.{key}": value for key, value in color_encoder.state_dict().items()}
        for i, cross_attention_adapter in enumerate(cross_attention_adapters):
            tensors |= {
                f"CrossAttentionAdapter_{i+1}.{key}": value
                for key, value in cross_attention_adapter.state_dict().items()
            }

        save_to_safetensors(
            path=trainer.ensure_checkpoints_save_folder / f"step{trainer.clock.step}.safetensors",
            tensors=tensors,
        )


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1]
    config = AdapterLatentDiffusionConfig.load_from_toml(toml_path=config_path)
    trainer = AdapterLatentDiffusionTrainer(config=config)
    trainer.train()
