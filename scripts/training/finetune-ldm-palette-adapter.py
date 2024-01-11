import random
from functools import cached_property
from typing import Any, TypedDict

import datasets
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from torch import Tensor, device as Device, dtype as DType
from torch.distributions import Beta
from torch.nn.functional import mse_loss
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
    TestDiffusionConfig,
    resize_image,
    sample_noise,
)
from refiners.training_utils.trainer import Trainer

# some images of the unsplash lite dataset are bigger than the default limit
Image.MAX_IMAGE_PIXELS = 200_000_000


class ColorEncoderConfig(BaseModel):
    """Configuration for the color encoder (of the palette adapter)."""

    dim_sinusoids: int = 64
    dim_embeddings: int = 256
    max_colors: int = 8


class AdapterConfig(BaseModel):
    """Configuration for the palette adapter."""

    color_encoder: ColorEncoderConfig


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


class PaletteBatch(TypedDict):
    """Structure of the data in the PaletteDataset."""

    latent: Tensor
    text_embedding: Tensor
    palettes: dict[str, Tensor]


class PaletteDataset(Dataset[PaletteBatch]):
    """Dataset for the palette adapter.

    Transforms the data from the Hugging Face dataset into PaletteBatches.
    """

    def __init__(self, trainer: "AdapterLatentDiffusionTrainer") -> None:
        super().__init__()
        self.trainer = trainer
        self.dataset = self.load_huggingface_dataset()

    @staticmethod
    def download_images(
        data: dict[str, list[Any]],  # "batched"
        dl_manager: datasets.DownloadManager,
    ) -> dict[str, list[str]]:
        """Download the image from the url."""
        urls = data["url"]
        filenames: list[str] = dl_manager.download(urls)  # type: ignore
        return {"image": filenames}

    @staticmethod
    def resize_images(
        data: dict[str, list[Any]],  # "batched"
        min_size: int = 512,
        max_size: int = 576,
    ) -> dict[str, list[Image.Image]]:
        """Resize the shortest side of the image between `min_size` and `max_size`."""
        return {
            "image": [
                resize_image(
                    image=image,
                    min_size=min_size,
                    max_size=max_size,
                )
                for image in data["image"]
            ],
        }

    @staticmethod
    def encode_captions(
        data: dict[str, list[Any]],  # "batched"
        text_encoder: CLIPTextEncoderL,
    ) -> dict[str, list[Tensor]]:
        """Encode the caption with the encoder."""
        return {
            "text_embedding": [text_encoder(caption).squeeze(0) for caption in data["caption"]],
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
        dl_manager = datasets.DownloadManager()
        dataset = dataset.map(  # type: ignore
            function=self.download_images,
            batched=True,
            num_proc=8,  # FIXME: harcoded value
            remove_columns=["url"],
            fn_kwargs={"dl_manager": dl_manager},
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
            batched=True,
            batch_size=50,  # FIXME: harcoded value
            remove_columns=["caption"],
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
        return self.trainer.text_encoder("").squeeze(0)

    def transform(self, data: dict[str, Any]) -> PaletteBatch:  # TODO: break into smaller chunks ?
        """Apply transforms to data."""
        # create the image transform
        image_transforms: list[Any] = []
        if self.trainer.config.dataset.random_crop_size:
            image_transforms.append(
                RandomCrop(size=self.trainer.config.dataset.random_crop_size),
            )
        if self.trainer.config.dataset.horizontal_flip_probability:
            image_transforms.append(
                RandomHorizontalFlip(p=self.trainer.config.dataset.horizontal_flip_probability),
            )
        image_compose = Compose(image_transforms)

        # encode the image into latent
        data["latent"] = self.trainer.lda.encode_image(
            image=image_compose(data["image"]),  # type: ignore
        ).squeeze(0)
        del data["image"]

        # randomly drop the text conditionning (cfg)
        if random.random() < self.trainer.config.ldm.unconditional_sampling_probability:
            data["text_embedding"] = self.empty_text_embedding

        return data  # type: ignore

    def __getitem__(self, index: int) -> PaletteBatch:
        # retreive data from dataset
        data = self.dataset[index]  # type: ignore

        # transform data
        data = self.transform(data)  # type: ignore

        return data

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
            max_colors=self.config.adapter.color_encoder.max_colors,
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

    def compute_loss(self, batch: PaletteBatch) -> Tensor:
        # retreive data from batch
        clip_text_embedding = batch["text_embedding"]
        latents = batch["latent"]
        batch_size = latents.shape[0]

        if random.random() < self.config.ldm.unconditional_sampling_probability:  # TODO: differentiate text and colors
            # randomly drop the palettes conditionning (cfg)
            # TODO: try to move this in the preprocessing above ?
            colors = Tensor(size=(batch_size, 0, 3))  # empty palette
        else:
            # select a palette at random, biased towards the middle, by using a Β(2, 2) distribution
            beta = Beta(2, 2)  # FIXME: harcoded value
            max_colors = self.config.adapter.color_encoder.max_colors
            i = int(beta.sample() * max_colors) + 1  # FIXME: harcoded value (max_colors)
            colors = batch["palettes"][str(i)]  # type: ignore

        # set unet color palette context
        self.unet.set_context("palette", {"colors": colors})

        # set unet text clip context
        self.unet.set_clip_text_embedding(clip_text_embedding=clip_text_embedding)

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
        color_encoder = adapter.color_encoder  # type: ignore
        cross_attention_adapters = adapter.cross_attention_adapters  # type: ignore

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
