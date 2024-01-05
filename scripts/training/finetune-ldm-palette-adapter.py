import random
from functools import cached_property, partial
from typing import Any, TypedDict

import datasets
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from torch import Generator, Tensor, device as Device, dtype as DType, randn
from torch.nn import Module
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip

import refiners.fluxion.layers as fl
from refiners.fluxion.utils import save_to_safetensors
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
from refiners.foundationals.latent_diffusion.palette_adapter import ColorEncoder, PaletteAdapter
from refiners.foundationals.latent_diffusion.schedulers.ddpm import DDPM
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
from refiners.training_utils.callback import Callback
from refiners.training_utils.config import BaseConfig
from refiners.training_utils.latent_diffusion import (
    LatentDiffusionConfig,
)
from refiners.training_utils.trainer import Trainer


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


class ColorEncoderConfig(BaseModel):
    """Configuration for the color encoder (of the palette adapter)."""

    dim_sinusoids: int = 64
    dim_embeddings: int = 256


class AdapterConfig(BaseModel):
    """Configuration for the palette adapter."""

    color_encoder: ColorEncoderConfig


class DatasetConfig(BaseModel):
    """Configuration for the dataset."""

    hf_repo: str
    revision: str = "main"
    split: str = "train"
    horizontal_flip_probability: float = 0.5
    random_crop_size: int = 512
    resize_image_max_size: int = 576


class AdapterLatentDiffusionConfig(BaseConfig):
    """Finetunning configuration.

    Contains the configs of the dataset, the latent diffusion model and the adapter.
    """

    dataset: DatasetConfig
    ldm: LatentDiffusionConfig
    adapter: AdapterConfig

    def model_post_init(self, _context: Any) -> None:
        """Pydantic v2 does post init differently, so we need to override this method too."""
        logger.info("Freezing models to train only the adapter.")
        self.models["lda"].train = False
        self.models["unet"].train = False
        self.models["text_encoder"].train = False
        self.models["color_encoder"].train = True


class PaletteBatch(TypedDict):
    """Structure of the data in the PaletteDataset."""

    latent: Tensor
    text_embedding: Tensor

    palette_1: Tensor
    palette_2: Tensor
    palette_3: Tensor
    palette_4: Tensor
    palette_5: Tensor
    palette_6: Tensor
    palette_7: Tensor
    palette_8: Tensor


class PaletteDataset(Dataset[PaletteBatch]):
    """Dataset for the palette adapter.

    Transforms the data from the Hugging Face dataset into PaletteBatches.
    """

    def __init__(self, trainer: "AdapterLatentDiffusionTrainer") -> None:
        super().__init__()
        self.trainer = trainer
        self.dataset = self.load_huggingface_dataset()

    @staticmethod
    def download_image(
        data: dict[str, Any],  # not "batched"
        dl_manager: datasets.DownloadManager,
    ) -> dict[str, str]:
        """Download the image from the url."""
        url = data["url"]
        filename: str = dl_manager.download(url)  # type: ignore (TODO: open PR upstream)
        return {"image": filename}

    @staticmethod
    def resize_image(
        data: dict[str, list[Any]],  # "batched"
        max_size: int = 576,
    ) -> dict[str, list[Image.Image]]:
        """Resize the longest side of the image to `max_size`."""
        return {
            "image": [image.convert("RGB").thumbnail((max_size, max_size)) for image in data["image"]],
        }

    @staticmethod
    def encode_caption(
        data: dict[str, list[Any]],  # "batched"
        text_encoder: CLIPTextEncoderL,
    ) -> dict[str, list[Tensor]]:
        """Encode the caption with the encoder."""
        return {
            "text_embedding": [text_encoder(caption) for caption in data["caption"]],
        }

    @staticmethod
    def transform(
        data: dict[str, list[Any]],  # "batched"
        image_encoder: Module,
        random_crop_size: int | None = None,
        horizontal_flip_probability: float | None = None,
        text_embedding_drop_probability: float | None = None,
        empty_text_embedding: Tensor | None = None,
    ) -> dict[str, list[Any]]:
        """Apply transforms to data."""
        # create the image transform
        image_transforms: list[Module] = []
        if random_crop_size:
            image_transforms.append(RandomCrop(size=random_crop_size))
        if horizontal_flip_probability:
            image_transforms.append(RandomHorizontalFlip(p=horizontal_flip_probability))
        image_compose = Compose(image_transforms)

        # encode the images into latents
        data["latent"] = [image_encoder(image=image_compose(image)) for image in data["image"]]
        del data["image"]

        # drop text embeddings randomly (classifier-free guidance ?)
        if text_embedding_drop_probability:
            data["text_embedding"] = [
                empty_text_embedding if random.random() < text_embedding_drop_probability else embedding
                for embedding in data["text_embedding"]
            ]

        return data

    def load_huggingface_dataset(self) -> datasets.Dataset:
        """Load the dataset from Hugging Face.

        Apply some pre-processing and set the "on-the-fly" transform.
        """
        dataset_config = self.trainer.config.dataset
        logger.info(
            f"Loading dataset from {dataset_config.hf_repo}, "
            f"split {dataset_config.split}, "
            f"revision {dataset_config.revision}"
        )
        dataset = datasets.load_dataset(  # type: ignore
            path=dataset_config.hf_repo,
            revision=dataset_config.revision,
            split=dataset_config.split,
        ).with_format(type="torch")

        # download images from urls
        dl_manager = datasets.DownloadManager()
        dataset = dataset.map(  # type: ignore
            function=self.download_image,
            remove_columns=["url"],
            fn_kwargs={"dl_manager": dl_manager},
            desc="Downloading images",  # type: ignore
        )

        # cast the "image" column Image
        dataset = dataset.cast_column(  # type: ignore
            column="image",
            feature=datasets.Image(),
        )

        # limit max image size
        dataset = dataset.map(  # type: ignore
            function=self.resize_image,
            batched=True,
            fn_kwargs={"max_size": dataset_config.resize_image_max_size},
            desc="Capping image sizes",  # type: ignore
        )

        # encode the captions into text embeddings
        encoded_empty_caption: Tensor = self.trainer.text_encoder("")
        dataset = dataset.map(  # type: ignore
            function=self.encode_caption,
            batched=True,
            remove_columns=["caption"],
            fn_kwargs={"encoder": self.trainer.text_encoder},
            desc="Encoding captions into embeddings",  # type: ignore
        )

        # set the "on-the-fly" transform
        transform = partial(  # FIXME: not proud of `partial` usage, find something better
            self.transform,
            random_crop_size=dataset_config.random_crop_size,
            horizontal_flip_probability=dataset_config.horizontal_flip_probability,
            image_encoder=self.trainer.lda,
            text_embedding_drop_probability=self.trainer.config.ldm.unconditional_sampling_probability,
            empty_text_embedding=encoded_empty_caption,
        )
        dataset.set_transform(transform=transform)  # type: ignore

        return dataset  # type: ignore

    def __getitem__(self, index: int) -> PaletteBatch:
        return self.dataset[index]  # type: ignore

    def __len__(self) -> int:
        return len(self.dataset)


class AdapterLatentDiffusionTrainer(Trainer[AdapterLatentDiffusionConfig, PaletteBatch]):
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
        adapter = PaletteAdapter(
            target=trainer.unet,
            color_encoder=trainer.color_encoder,
        )
        adapter.inject()


class SaveAdapter(Callback[AdapterLatentDiffusionTrainer]):
    """Callback to save the adapter when a checkpoint is saved."""

    def on_checkpoint_save(self, trainer: AdapterLatentDiffusionTrainer) -> None:
        unet = trainer.unet
        adapter = unet.parent
        color_encoder = adapter.color_encoder
        cross_attention_adapters = adapter.cross_attention_adapters

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
