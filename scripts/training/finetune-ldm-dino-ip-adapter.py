import random
from dataclasses import dataclass
from functools import cached_property
from typing import Any, List

import datasets
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from torch import Tensor, cat, device as Device, dtype as DType, randn, zeros_like, exp, ones_like, stack, randn_like, no_grad
from torch.distributions import Beta
from torch.nn import Module
from torch.nn.functional import mse_loss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip
from refiners.foundationals.dinov2 import DINOv2_small, DINOv2_small_reg, DINOv2_base, DINOv2_base_reg, DINOv2_large, DINOv2_large_reg, ViT
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
from refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock2d

import refiners.fluxion.layers as fl
from refiners.fluxion.utils import save_to_safetensors
from refiners.foundationals.latent_diffusion.stable_diffusion_1.image_prompt import SD1IPAdapter, get_sd1_image_proj
from refiners.foundationals.latent_diffusion.schedulers.ddpm import DDPM
from refiners.foundationals.latent_diffusion.schedulers.dpm_solver import DPMSolver
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder, SD1UNet, StableDiffusion_1
from refiners.training_utils.callback import Callback
from refiners.training_utils.config import BaseConfig
from refiners.foundationals.latent_diffusion.image_prompt import ImageProjection, PerceiverResampler
from refiners.training_utils.latent_diffusion import (
    LatentDiffusionConfig,
    TestDiffusionConfig,
    resize_image,
    sample_noise,
    filter_image,
)
from refiners.training_utils.trainer import Trainer
from refiners.training_utils.wandb import WandbLoggable
import webdataset as wds
from refiners.fluxion.utils import manual_seed, load_from_safetensors
# some images of the unsplash lite dataset are bigger than the default limit
Image.MAX_IMAGE_PIXELS = 200_000_000

class AdapterConfig(BaseModel):
    """Configuration for the IP adapter."""
    seed: int = 9752
    image_encoder_type: str
    checkpoint: str | None = None
    resolution: int = 518
    scale: float = 1.0
    inference_scale: float = 0.75
    use_pooled_text_embedding: bool = False
    use_timestep_embedding: bool = False
    fine_grained: bool = False


class DatasetConfig(BaseModel):
    """Configuration for the dataset."""

    hf_repo: str
    revision: str = "main"
    split: str = "train"
    horizontal_flip_probability: float = 0.5
    resize_image_min_size: int = 512
    resize_image_max_size: int = 576
    filter_min_image_size: bool = False
    random_crop_size: int = 512
    image_drop_rate: float = 0.05
    text_drop_rate: float = 0.05
    text_and_image_drop_rate: float = 0.05
    to_wds: bool = False # TODO: It seems like using webdatasets increase data fetching speed by around 40% https://github.com/huggingface/pytorch-image-models/discussions/1524
    pre_encode: bool = False # TODO


class TestIPDiffusionConfig(TestDiffusionConfig):
    """Configuration to test the diffusion model, during the `evaluation` loop of the trainer."""

    validation_image_paths: List[str]




class AdapterLatentDiffusionConfig(BaseConfig):
    """Finetunning configuration.

    Contains the configs of the dataset, the latent diffusion model and the adapter.
    """

    dataset: DatasetConfig
    ldm: LatentDiffusionConfig
    adapter: AdapterConfig
    test_ldm: TestIPDiffusionConfig


@dataclass
class IPBatch:
    """Structure of the data in the IPDataset."""

    latent: Tensor
    text_embedding: Tensor
    cond_image: Tensor


class IPDataset(Dataset[IPBatch]):
    """Dataset for the IP adapter.

    Transforms the data from the HuggingFace dataset into `IPBatch`.
    The `collate_fn` is used by the trainer to batch the data.
    """

    def __init__(self, trainer: "AdapterLatentDiffusionTrainer") -> None:
        super().__init__()
        self.trainer = trainer
        self.dataset = self.load_huggingface_dataset()
        cond_resolution = trainer.config.adapter.resolution
        self.image_encoder_transform = lambda x: trainer.adapter.preprocess_image(x, (cond_resolution, cond_resolution))

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
    def filter_images(
        images: list[Image.Image],
        min_size: int = 512,
    ) -> dict[str, list[bool]]:
        """Resize the images such that their shortest side is between `min_size` and `max_size`."""
        return {
            "image": [
                filter_image(
                    image=image,
                    min_size=min_size,
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
        # remove min size images
        if dataset_config.filter_min_image_size:
            dataset = dataset.filter( # type: ignore
                function=self.filter_images,
                input_columns=["image"],
                batched=True,
                batch_size=10,  # FIXME: harcoded value
                num_proc=8,  # FIXME: harcoded value
                fn_kwargs={
                    "min_size": dataset_config.resize_image_min_size,
                },
                desc="Capping image sizes",  # type: ignore
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
            ],
        )

        return dataset  # type: ignore

    @cached_property
    def empty_text_embedding(self) -> Tensor:
        """Return an empty text embedding."""
        return self.trainer.text_encoder("")
    @no_grad()
    def transform(self, data: dict[str, Any]) -> IPBatch:
        """Apply transforms to data."""
        dataset_config = self.trainer.config.dataset
        image = data["image"]
        cond_image = self.image_encoder_transform(image).to(self.trainer.device, dtype=self.trainer.dtype)
        cond_image = self.trainer.adapter.compute_conditional_image_embedding(cond_image)[0]
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
        image = image_compose(image)  # type: ignore

        # encode the image into latent
        latent = self.trainer.lda.encode_image(image=image)  # type: ignore

        text_embedding = data["text_embedding"]
        rand_num = random.random()
        if rand_num < dataset_config.image_drop_rate:
            cond_image = zeros_like(cond_image)
        elif rand_num < (dataset_config.image_drop_rate + dataset_config.text_drop_rate):
            text_embedding = self.empty_text_embedding
        elif rand_num < (dataset_config.image_drop_rate + dataset_config.text_drop_rate + dataset_config.text_and_image_drop_rate):
            text_embedding = self.empty_text_embedding
            cond_image = zeros_like(cond_image)

        return IPBatch(
            latent=latent,
            text_embedding=text_embedding,
            cond_image=cond_image,
        )

    def __getitem__(self, index: int) -> IPBatch:
        # retreive data from the huggingface dataset
        data = self.dataset[index]  # type: ignore
        # augment/transform into IPBatch
        data = self.transform(data)  # type: ignore
        return data

    def collate_fn(self, batch: list[IPBatch]) -> IPBatch:
        latents = cat(tensors=[item.latent.to(self.trainer.device, dtype=self.trainer.dtype) for item in batch])
        text_embeddings = cat(tensors=[item.text_embedding.to(self.trainer.device, dtype=self.trainer.dtype) for item in batch])
        cond_images = pad_sequence([item.cond_image.to(self.trainer.device, dtype=self.trainer.dtype) for item in batch], batch_first=True)
        return IPBatch(
            latent=latents,
            text_embedding=text_embeddings,
            cond_image=cond_images,
        )

    def __len__(self) -> int:
        return len(self.dataset)


class AdapterLatentDiffusionTrainer(Trainer[AdapterLatentDiffusionConfig, IPBatch]):
    @cached_property
    def lda(self) -> SD1Autoencoder:
        assert self.config.models["lda"] is not None, "The config must contain a lda entry."
        return SD1Autoencoder(
            device=self.device,
        ).to(self.device, dtype=self.dtype)

    @cached_property
    def unet(self) -> SD1UNet:
        assert self.config.models["unet"] is not None, "The config must contain a unet entry."
        return SD1UNet(
            in_channels=4,  # FIXME: harcoded value
            device=self.device,
        ).to(self.device, dtype=self.dtype)

    @cached_property
    def text_encoder(self) -> CLIPTextEncoderL:
        assert self.config.models["text_encoder"] is not None, "The config must contain a text_encoder entry."
        return CLIPTextEncoderL(
            device=self.device,
        ).to(self.device, dtype=self.dtype)
    @cached_property
    def image_encoder(self) -> ViT:
        assert self.config.models["image_encoder"] is not None, "The config must contain an image_encoder entry."
        image_encoder_cls = DINOv2_base
        if self.config.adapter.image_encoder_type == "dinov2_vitl14_reg4":
            image_encoder_cls = DINOv2_large_reg
        elif self.config.adapter.image_encoder_type == "dinov2_vitl14":
            image_encoder_cls = DINOv2_large
        elif self.config.adapter.image_encoder_type == "dinov2_vitb14_reg4":
            image_encoder_cls = DINOv2_base_reg
        elif self.config.adapter.image_encoder_type == "dinov2_vits14_reg4":
            image_encoder_cls = DINOv2_small_reg
        elif self.config.adapter.image_encoder_type == "dinov2_vits14":
            image_encoder_cls = DINOv2_small
        return image_encoder_cls().to(self.device, dtype=self.dtype)
    @cached_property
    def image_proj(self) -> ImageProjection | PerceiverResampler:
        assert self.config.models["image_proj"] is not None, "The config must contain an image_encoder entry."
        cross_attn_2d = self.unet.ensure_find(CrossAttentionBlock2d)
        image_proj = get_sd1_image_proj(self.image_encoder, self.unet, cross_attn_2d, self.config.adapter.fine_grained)
        return image_proj.to(self.device, dtype=self.dtype)
    @cached_property
    def adapter(self) -> SD1IPAdapter:
        assert self.config.models["adapter"] is not None, "The config must contain an adapter entry."
        # A bit of hacky method to initialize model with weights.Potentially refactor this
        ip_adapter = SD1IPAdapter(
            target=self.unet,
            weights=load_from_safetensors(self.config.adapter.checkpoint) if self.config.adapter.checkpoint is not None else None,
            strict=False,
            fine_grained=self.config.adapter.fine_grained,
            scale=self.config.adapter.scale,
            use_timestep_embedding=self.config.adapter.use_timestep_embedding,
            use_pooled_text_embedding=self.config.adapter.use_pooled_text_embedding,
            image_encoder=self.image_encoder,
            image_proj=self.image_proj
        )
        return ip_adapter.to(self.device, dtype=self.dtype)

    @cached_property
    def ddpm_scheduler(self) -> DDPM:
        return DDPM(
            num_inference_steps=1000,  # FIXME: harcoded value
            device=self.device,
        ).to(self.device, dtype=self.dtype)

    @cached_property
    def signal_to_noise_ratios(self) -> Tensor:
        return  exp(self.ddpm_scheduler.signal_to_noise_ratios)**2
    def load_models(self) -> dict[str, fl.Module]:
        return {
            "lda": self.lda,
            "unet": self.unet,
            "text_encoder": self.text_encoder,
            "image_encoder": self.image_encoder,
            "image_proj": self.image_proj,
            "adapter": self.adapter,
        }

    def load_dataset(self) -> IPDataset:
        return IPDataset(trainer=self)

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

    def compute_loss(self, batch: IPBatch) -> Tensor:
        # retreive data from batch
        latents = batch.latent.to(self.device, dtype=self.dtype)
        text_embeddings = batch.text_embedding.to(self.device, dtype=self.dtype)
        cond_image = batch.cond_image.to(self.device, dtype=self.dtype)

        # set IP embeddings context
        self.adapter.set_image_embedding(cond_image)

        # set text embeddings context
        self.unet.set_clip_text_embedding(clip_text_embedding=text_embeddings)

        # sample timestep and set unet timestep context
        timestep = self.sample_timestep()
        self.unet.set_timestep(timestep=timestep)

        # sample noise and noisify the latents
        noise = self.sample_noise(size=latents.shape, dtype=latents.dtype)
        input_perturbation = self.config.ldm.input_perturbation
        if input_perturbation > 0:
            new_noise = noise + input_perturbation * randn_like(noise)
        if input_perturbation > 0:
            noisy_latents = self.ddpm_scheduler.add_noise(x=latents, noise=new_noise, step=self.current_step)
        else:
            noisy_latents = self.ddpm_scheduler.add_noise(x=latents, noise=noise, step=self.current_step)

        # get prediction from unet
        prediction = self.unet(noisy_latents)

        # compute mse loss
        snr_gamma = self.config.ldm.snr_gamma
        if snr_gamma is None:
            loss = mse_loss(prediction.float(), noise.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            signal_to_noise_ratios = self.signal_to_noise_ratios[timestep]

            mse_loss_weights = (
                stack([signal_to_noise_ratios, snr_gamma * ones_like(timestep)], dim=1).min(dim=1)[0] / signal_to_noise_ratios
            )

            loss = mse_loss(prediction.float(), noise.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

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
        self.adapter.scale = self.config.adapter.inference_scale
        # retreive data from config
        prompts = self.config.test_ldm.prompts
        validation_image_paths = self.config.test_ldm.validation_image_paths
        assert len(prompts) == len(validation_image_paths)
        num_images_per_prompt = self.config.test_ldm.num_images_per_prompt
        if self.config.test_ldm.use_short_prompts:
            prompts = [prompt.split(sep=",")[0] for prompt in prompts]
        cond_images = [Image.open(validation_image_path) for validation_image_path in validation_image_paths]

        # for each prompt generate `num_images_per_prompt` images
        # TODO: remove this for loop, batch things up
        images: dict[str, WandbLoggable] = {}
        for i in range(len(cond_images)):
            images[f"condition images_{i}"] = cond_images[i]
        for prompt, cond_image in zip(prompts, cond_images):
            canvas_image = Image.new(mode="RGB", size=(512, 512 * num_images_per_prompt))
            clip_text_embedding = sd.compute_clip_text_embedding(text=prompt).to(self.device, dtype=self.dtype)
            cond_resolution = self.config.adapter.resolution
            image_embedding = self.adapter.compute_image_embedding(self.adapter.preprocess_image(cond_image, (cond_resolution, cond_resolution)))
            # TODO: pool text according to end of text id for pooled text embeds if given option
            for i in range(num_images_per_prompt):
                manual_seed(self.config.test_ldm.seed)
                logger.info(f"Generating image {i+1}/{num_images_per_prompt} for prompt: {prompt}")
                x = randn(1, 4, 64, 64, device=self.device)
                self.adapter.set_image_embedding(image_embedding)
                for step in sd.steps:
                    x = sd(
                        x=x,
                        step=step,
                        clip_text_embedding=clip_text_embedding,
                    )
                canvas_image.paste(sd.lda.decode_latents(x=x), box=(0, 512 * i))
            images[prompt] = canvas_image
        # log images to wandb
        self.log(data=images)
        self.adapter.scale = self.config.adapter.scale

    def __init__(
        self,
        config: AdapterLatentDiffusionConfig,
        callbacks: "list[Callback[Any]] | None" = None,
    ) -> None:
        # if initializing after, the on_init_end methods do not get called for the extended callbacks
        if callbacks is None:
            callbacks = []
        callbacks.extend((IPSubmodulesFreeze(), LoadAdapter(), SaveAdapter(), ComputeGradNorm()))
        super().__init__(config=config, callbacks=callbacks)

class IPSubmodulesFreeze(Callback[AdapterLatentDiffusionTrainer]):
    """Callback to compute gradient norm"""
    def on_init_end(self, trainer: AdapterLatentDiffusionTrainer) -> None:
        trainer.image_encoder.requires_grad_(False)
        trainer.unet.requires_grad_(False)
        return super().on_init_end(trainer)

class ComputeGradNorm(Callback[AdapterLatentDiffusionTrainer]):
    """Callback to compute gradient norm"""
    def on_backward_end(self, trainer: AdapterLatentDiffusionTrainer) -> None:
        if trainer.clock.is_evaluation_step:
            for name, param in trainer.adapter.named_parameters():
                if param.grad is not None:
                    grads = param.grad.detach().data
                    grad_norm = (grads.norm(p=2) / grads.numel()).item()
                    trainer.log(data={"grad_norm/" + name: grad_norm})
        return super().on_backward_end(trainer)

class LoadAdapter(Callback[AdapterLatentDiffusionTrainer]):
    """Callback to load the adapter at the beginning of the training."""

    def on_train_begin(self, trainer: AdapterLatentDiffusionTrainer) -> None:
        print("before inject", len(trainer.learnable_parameters))
        trainer.adapter.inject()
        print("after inject", len(trainer.learnable_parameters))
        for model_name in trainer.models:
            print(model_name)
            i = 0
            for name, param in trainer.models[model_name].named_parameters():
                if param.requires_grad:
                    print(name)
                    i += 1
            print(i)


class SaveAdapter(Callback[AdapterLatentDiffusionTrainer]):
    """Callback to save the adapter when a checkpoint is saved."""

    def on_checkpoint_save(self, trainer: AdapterLatentDiffusionTrainer) -> None:
        adapter = trainer.adapter
        cross_attention_adapters = trainer.adapter.cross_attention_adapters

        tensors: dict[str, Tensor] = {}
        tensors |= {f"SD1IPAdapter.{key}": value for key, value in adapter.state_dict().items()}
        for i, cross_attention_adapter in enumerate(cross_attention_adapters):
            tensors |= {
                f"CrossAttentionAdapter_{i+1}.{key}": value
                for key, value in cross_attention_adapter.state_dict().items()
            }
        if trainer.config.adapter.use_pooled_text_embedding:
            tensors |= {
                f"pooled_text_embedding_proj.{key}": value
                for key, value in adapter.pooled_text_embedding_proj.state_dict().items()
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