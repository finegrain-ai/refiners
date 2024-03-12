import random
from dataclasses import dataclass
from functools import cached_property
from typing import Any, List, Callable
import os
import datasets
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from jaxtyping import Float
from torch import (
    tensor,
    Tensor,
    cat,
    device as Device,
    dtype as DType,
    randn,
    zeros_like,
    exp,
    ones_like,
    stack,
    randn_like,
    no_grad,
    randint,
    float32,
    zeros,
    ones,
    norm,
    multinomial
)
from torch.utils.data import DataLoader, Dataset, default_collate

from torch.cuda import empty_cache
from torch.distributions import Beta
from torch.nn import Module, Linear, Embedding, LayerNorm
from torch.nn.init import trunc_normal_
from torch.nn.functional import mse_loss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, CenterCrop, Resize
from refiners.foundationals.dinov2 import (
    DINOv2_small,
    DINOv2_small_reg,
    DINOv2_base,
    DINOv2_base_reg,
    DINOv2_large,
    DINOv2_large_reg,
    ViT,
)
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
from refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock2d
from refiners.fluxion.utils import image_to_tensor, normalize

import refiners.fluxion.layers as fl
from refiners.fluxion.utils import save_to_safetensors
from refiners.foundationals.latent_diffusion.stable_diffusion_1.image_prompt import SD1IPAdapter, get_sd1_image_proj
from refiners.foundationals.latent_diffusion.solvers.ddpm import DDPM
from refiners.foundationals.latent_diffusion.solvers.dpm import DPMSolver
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder, SD1UNet, StableDiffusion_1
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.text_encoder import TextEncoderWithPoolingL
from refiners.training_utils.callback import Callback, CallbackConfig
from refiners.training_utils.config import BaseConfig, ModelConfig
from refiners.foundationals.latent_diffusion.image_prompt import ImageProjection, PerceiverResampler
from refiners.training_utils.latent_diffusion import (
    LatentDiffusionConfig,
    TestDiffusionConfig,
    resize_image,
    sample_noise,
    filter_image,
)
from refiners.training_utils.trainer import register_model, Trainer, register_callback
from refiners.training_utils.wandb import WandbLoggable, WandbMixin, WandbConfig
import webdataset as wds
from refiners.fluxion.utils import load_from_safetensors
import gc
import math
import shutil
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)


# some images of the unsplash lite dataset are bigger than the default limit
Image.MAX_IMAGE_PIXELS = 200_000_000


class AdapterConfig(ModelConfig):
    """Configuration for the IP adapter."""

    image_encoder_type: str
    save_folder: str | None = None
    resolution: int = 518
    scale: float = 1.0
    inference_scale: float = 0.75
    use_pooled_text_embedding: bool = False
    use_timestep_embedding: bool = False
    fine_grained: bool = False
    initialize_model: bool = True
    initializer_range: float = 0.02
    use_bias: bool = False
    do_palp: bool = False
    palp_alpha: float = 15
    palp_beta: float = 7.5
    use_rescaler: bool = False
    image_embedding_div_factor: float = 1
    pooled_text_div_factor: float = 1
    palp_rescale: bool = False
    palp_steps: int = 4
    layernorm_dino: bool = False
    non_palp_image_drop_rate: float = 0.05
    non_palp_text_drop_rate: float = 0.05
    non_palp_text_and_image_drop_rate: float = 0.05
    weighted_sum: bool = False
    timestep_bias_strategy: str = "none"
    timestep_bias_portion: float = 0.5
    timestep_bias_begin: int = 0
    timestep_bias_end: int = 1000
    timestep_bias_multiplier: float = 1.0


class DatasetConfig(BaseModel):
    """Configuration for the dataset."""

    hf_repo: str
    revision: str = "main"
    split: str = "train"
    horizontal_flip_probability: float = 0.5
    resize_image_min_size: int = 512
    resize_image_max_size: int = 576
    filter_min_image_size: bool = False
    random_crop_size: int | None = None
    center_crop_size: int | None = 512
    image_drop_rate: float = 0.05
    text_drop_rate: float = 0.05
    text_and_image_drop_rate: float = 0.05
    webdataset: bool = False  # TODO: It seems like using webdatasets increase data fetching speed by around 40% https://github.com/huggingface/pytorch-image-models/discussions/1524
    train_shards_path_or_url: str | None = None
    shuffle_buffer_size: int = 1000
    pre_encode: bool = False  # TODO
    image_column: str = "image"
    caption_column: str = "caption"
    download_images: bool = True
    save_path: str | None = None
    dataset_length: int | None = None
    zero_uncond: bool = False
    num_train_examples: int = 567597


# Adapted from https://github.com/huggingface/open-muse
def _init_learnable_weights(module: Module, initializer_range: float):
    """
    Initialize the weights according to the original implementation.
    https://github.com/google-research/maskgit/blob/main/maskgit/nets/maskgit_transformer.py#L37
    """

    # TODO: make this configurable
    if isinstance(module, Linear):
        if module.weight.requires_grad:
            if initializer_range == 0:
                module.weight.data.zero_()
            else:
                trunc_normal_(module.weight, std=initializer_range)
        if module.bias is not None and module.bias.requires_grad:
            module.bias.data.zero_()
    elif isinstance(module, Embedding):
        if module.weight.requires_grad:
            if initializer_range == 0:
                module.weight.data.zero_()
            else:
                trunc_normal_(module.weight, std=initializer_range)
    elif isinstance(module, (LayerNorm)):
        if hasattr(module, "weight") and module.weight.requires_grad:
            module.weight.data.fill_(1.0)
        if hasattr(module, "bias") and module.bias is not None and module.bias.requires_grad:
            module.bias.data.zero_()

# taken from simpletuner
def generate_timestep_weights(args: AdapterConfig, num_timesteps: int) -> Tensor:
    weights = ones(num_timesteps)

    # Determine the indices to bias
    num_to_bias = int(args.timestep_bias_portion * num_timesteps)

    if args.timestep_bias_strategy == "later":
        bias_indices = slice(-num_to_bias, None)
    elif args.timestep_bias_strategy == "earlier":
        bias_indices = slice(0, num_to_bias)
    elif args.timestep_bias_strategy == "range":
        # Out of the possible 1000 timesteps, we might want to focus on eg. 200-500.
        range_begin = args.timestep_bias_begin
        range_end = args.timestep_bias_end
        if range_begin < 0:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero."
            )
        if range_end > num_timesteps:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps."
            )
        bias_indices = slice(range_begin, range_end)
    else:  # 'none' or any other string
        return weights
    if args.timestep_bias_multiplier <= 0:
        raise ValueError(
            "The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps."
            " If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead."
            " A timestep bias multiplier less than or equal to 0 is not allowed."
        )

    # Apply the bias
    weights[bias_indices] *= args.timestep_bias_multiplier

    # Normalize
    weights /= weights.sum()

    return weights

class TestIPDiffusionConfig(TestDiffusionConfig):
    """Configuration to test the diffusion model, during the `evaluation` loop of the trainer."""

    validation_image_paths: List[str]

class AdapterLatentDiffusionConfig(BaseConfig):
    """Finetunning configuration.

    Contains the configs of the dataset, the latent diffusion model and the adapter.
    """

    dataset: DatasetConfig
    ldm: LatentDiffusionConfig
    test_ldm: TestIPDiffusionConfig
    compute_grad_norms: CallbackConfig
    compute_param_norms: CallbackConfig
    save_adapter: CallbackConfig
    wandb: WandbConfig
    unet: ModelConfig
    lda: ModelConfig
    text_encoder: ModelConfig
    image_encoder: ModelConfig
    # image proj has to be after image encoder or it fails
    image_proj: ModelConfig
    # adapter needs to be initialized later for this to work
    adapter: AdapterConfig


@dataclass
class IPBatch:
    """Structure of the data in the IPDataset."""

    latent: Tensor
    text_embedding: Tensor
    pooled_text_embedding: Tensor | None
    image_embedding: Tensor

class ComputeGradNormCallback(Callback["AdapterLatentDiffusionTrainer"]):
    """Callback to compute gradient norm"""

    def on_backward_end(self, trainer: "AdapterLatentDiffusionTrainer") -> None:
        if trainer.clock.is_evaluation_step:
            for name, param in trainer.adapter.named_parameters():
                if param.grad is not None:
                    grads = param.grad.detach().data
                    grad_norm = (grads.norm(p=2) / grads.numel()).item()
                    trainer.wandb_log(data={"grad_norm/" + name: grad_norm})
            for name, param in trainer.image_proj.named_parameters():
                if param.grad is not None:
                    grads = param.grad.detach().data
                    grad_norm = (grads.norm(p=2) / grads.numel()).item()
                    trainer.wandb_log(data={"grad_norm/" + name: grad_norm})
        return super().on_backward_end(trainer)


class ComputeParamNormCallback(Callback["AdapterLatentDiffusionTrainer"]):
    """Callback to compute gradient norm"""

    def on_backward_end(self, trainer: "AdapterLatentDiffusionTrainer") -> None:
        if trainer.clock.is_evaluation_step:
            for name, param in trainer.adapter.named_parameters():
                if param.grad is not None:
                    data = param.data.detach()
                    data_norm = (data.norm(p=2) / data.numel()).item()
                    trainer.wandb_log(data={"param_norm/" + name: data_norm})
            for name, param in trainer.image_proj.named_parameters():
                if param.grad is not None:
                    data = param.data.detach()
                    data_norm = (data.norm(p=2) / data.numel()).item()
                    trainer.wandb_log(data={"param_norm/" + name: data_norm})
        return super().on_backward_end(trainer)


class SaveAdapterCallback(Callback["AdapterLatentDiffusionTrainer"]):
    """Callback to save the adapter when a checkpoint is saved."""

    def on_backward_end(self, trainer: "AdapterLatentDiffusionTrainer") -> None:
        if trainer.clock.is_evaluation_step:
            os.makedirs(trainer.config.adapter.save_folder, exist_ok=True)
            cross_attention_adapters = trainer.adapter.sub_adapters
            image_proj = trainer.adapter.image_proj

            tensors: dict[str, Tensor] = {}
            tensors |= {f"image_proj.{key}": value for key, value in image_proj.state_dict().items()}
            for i, cross_attention_adapter in enumerate(cross_attention_adapters):
                tensors |= {f"ip_adapter.{i:03d}.{key}": value for key, value in cross_attention_adapter.state_dict().items()}
            save_to_safetensors(
                path= f"{trainer.config.adapter.save_folder}/step{trainer.clock.iteration}.safetensors",
                tensors=tensors,
            )

def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f
def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


class IPDataset(Dataset[IPBatch]):
    """Dataset for the IP adapter.

    Transforms the data from the HuggingFace dataset into `IPBatch`.
    The `collate_fn` is used by the trainer to batch the data.
    """

    @no_grad()
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
    def convert2rgb(
        images: list[Image.Image],
    ) -> dict[str, list[Image.Image]]:
        rgb_images: list[Image.Image] = []
        for image in images:
            if image.mode != "RGB":
                image = image.convert("RGB")
            rgb_images.append(image)
        return {"image": rgb_images}

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
        text_encoder: CLIPTextEncoderL | TextEncoderWithPoolingL,
    ) -> dict[str, list[Tensor]]:
        """Encode the captions with the text encoder."""
        output: dict[str, list[Tensor]] = {
            "text_embedding": [],
        }
        if isinstance(text_encoder, TextEncoderWithPoolingL):
            output["pooled_text_embedding"] = []
        for caption in captions:
            if isinstance(text_encoder, CLIPTextEncoderL):
                text_embedding = text_encoder(caption)
                output["text_embedding"].append(text_embedding.float().cpu())
            else:
                text_embedding, pooled_text_embedding = text_encoder(caption)
                assert isinstance(text_embedding, Tensor)
                assert isinstance(pooled_text_embedding, Tensor)
                output["text_embedding"].append(text_embedding.float().cpu())
                output["pooled_text_embedding"].append(pooled_text_embedding.float().cpu())

        return output

    @staticmethod
    def cond_transform(
        image: Image.Image,
        device: Device,
        dtype: DType,
        size: tuple[int, int] = (518, 518),
        mean: list[float] | None = None,
        std: list[float] | None = None,
    ) -> Tensor:
        # Default mean and std are parameters from https://github.com/openai/CLIP
        return normalize(
            image_to_tensor(image.resize(size), device=device, dtype=dtype),
            mean=[0.48145466, 0.4578275, 0.40821073] if mean is None else mean,
            std=[0.26862954, 0.26130258, 0.27577711] if std is None else std,
        )

    @staticmethod
    def encode_cond_images(
        images: list[Image.Image],
        image_encoder: ViT,
        image_encoder_column: str,
        device: Device,
        dtype: DType,
        cond_resolution: int,
    ) -> dict[str, list[Tensor]]:
        cond_images = [
            IPDataset.cond_transform(image, device, dtype, (cond_resolution, cond_resolution)) for image in images
        ]
        return {image_encoder_column: [image_encoder(cond_image).float().cpu() for cond_image in cond_images]}

    @staticmethod
    def encode_lda_images(
        images: list[Image.Image],
        lda: SD1Autoencoder,
        random_crop_size: int | None = None,
        center_crop_size: int | None = 512,
        horizontal_flip_probability: float = 0.5,
    ) -> dict[str, list[Tensor]]:
        # TODO: Multiple random latents per image
        image_transforms: list[Module] = []
        if random_crop_size:
            image_transforms.append(
                RandomCrop(size=random_crop_size),
            )
        else:
            image_transforms.append(
                CenterCrop(size=center_crop_size),
            )
        if horizontal_flip_probability:
            image_transforms.append(
                RandomHorizontalFlip(p=horizontal_flip_probability),
            )
        image_compose = Compose(image_transforms)
        lda_images: List[Image.Image] = [image_compose(image) for image in images]
        return {"lda_embedding": [lda.image_to_latents(image=image).float().cpu() for image in lda_images]}
    @no_grad()
    def load_huggingface_dataset(self) -> datasets.Dataset:
        """Load the dataset from Hugging Face and apply some pre-processing."""
        dataset_config = self.trainer.config.dataset
        logger.info(
            f"Loading dataset from {dataset_config.hf_repo}, "
            f"revision {dataset_config.revision}, "
            f"split {dataset_config.split}"
        )
        dataset_save_path: str | None = self.trainer.config.dataset.save_path
        update_dataset: bool = False
        if dataset_save_path and os.path.exists(dataset_save_path):
            dataset = datasets.load_from_disk(self.trainer.config.dataset.save_path)
        else:
            dataset = datasets.load_dataset(  # type: ignore
                path=dataset_config.hf_repo,
                revision=dataset_config.revision,
                split=dataset_config.split,
            )
            if dataset_config.dataset_length is not None:
                dataset = dataset.select(list(range(dataset_config.dataset_length)))
            logger.info(f"Dataset has {len(dataset)} elements")

            if dataset_config.download_images:
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
            else:
                dataset = dataset.rename_column(dataset_config.image_column, "image")  # type: ignore

            # cast the "image" column to Image feature type
            dataset = dataset.cast_column(  # type: ignore
                column="image",
                feature=datasets.Image(),
            )
            dataset = dataset.map(
                function=self.convert2rgb,
                input_columns=["image"],
                batched=True,
                batch_size=10,  # FIXME: harcoded value
                num_proc=8,  # FIXME: harcoded value
                desc="Convert to rbg images",  # type: ignore
            )
            # remove min size images
            if dataset_config.filter_min_image_size:
                dataset = dataset.filter(  # type: ignore
                    function=self.filter_images,
                    input_columns=["image"],
                    batched=True,
                    batch_size=10,  # FIXME: harcoded value
                    num_proc=8,  # FIXME: harcoded value
                    fn_kwargs={
                        "min_size": dataset_config.resize_image_min_size,
                    },
                    desc="Filtering image sizes",  # type: ignore
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
        # encode cond images
        if self.trainer.config.dataset.pre_encode:
            if "image_embedding" not in dataset.features:
                update_dataset = True
                dataset = dataset.map(  # type: ignore
                    function=self.encode_cond_images,
                    input_columns=["image"],
                    batched=True,
                    batch_size=50,  # FIXME: harcoded value
                    fn_kwargs={
                        "image_encoder": self.trainer.adapter.image_encoder,  # weights must be loaded to get same hash everytime
                        "image_encoder_column": "image_embedding",
                        "device": self.trainer.device,
                        "dtype": self.trainer.dtype,
                        "cond_resolution": self.cond_resolution,
                    },
                    desc="Encoding conditional images into embeddings",  # type: ignore
                )
            if self.trainer.config.adapter.layernorm_dino and ("image_embedding_layernorm" not in dataset.features):
                update_dataset = True
                dataset = dataset.map(  # type: ignore
                    function=self.encode_cond_images,
                    input_columns=["image"],
                    batched=True,
                    batch_size=50,  # FIXME: harcoded value
                    fn_kwargs={
                        "image_encoder": self.trainer.adapter.image_encoder,  # weights must be loaded to get same hash everytime
                        "image_encoder_column": "image_embedding_layernorm",
                        "device": self.trainer.device,
                        "dtype": self.trainer.dtype,
                        "cond_resolution": self.cond_resolution,
                    },
                    desc="Encoding conditional images into embeddings",  # type: ignore
                )
            if "lda_embedding" not in dataset.features:
                update_dataset = True
                dataset = dataset.map(  # type: ignore
                    function=self.encode_lda_images,
                    input_columns=["image"],
                    batched=True,
                    batch_size=50,  # FIXME: harcoded value
                    fn_kwargs={
                        "lda": self.trainer.lda,
                        "random_crop_size": self.trainer.config.dataset.random_crop_size,
                        "center_crop_size": self.trainer.config.dataset.center_crop_size,
                        "horizontal_flip_probability": self.trainer.config.dataset.horizontal_flip_probability,
                    },
                    desc="Encoding lda images into embeddings",  # type: ignore
                )
        if "text_embedding" not in dataset.features or (("pooled_text_embedding" not in dataset.features) and self.trainer.config.adapter.use_pooled_text_embedding):
            update_dataset = True
            # encode the captions into text embedding
            dataset = dataset.rename_column(dataset_config.caption_column, "caption")  # type: ignore
            dataset = dataset.map(  # type: ignore
                function=self.encode_captions,
                input_columns=["caption"],
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
            columns=["text_embedding", "image_embedding", "lda_embedding"],
        )
        if dataset_save_path and update_dataset:
            dataset.save_to_disk(dataset_save_path+"_update")
            del dataset
            shutil.rmtree(dataset_save_path)
            os.rename(dataset_save_path+"_update", dataset_save_path)
        return dataset  # type: ignore

    def transform(self, data: dict[str, Any]) -> IPBatch:
        """Apply transforms to data."""
        if not self.trainer.config.dataset.pre_encode:
            image = data["image"]
            cond_image = self.cond_transform(
                image, self.trainer.device, self.trainer.dtype, (self.trainer.cond_resolution, self.trainer.cond_resolution)
            )
            image_embedding = self.trainer.adapter.image_encoder(cond_image)
            # apply augmentation to the image
            image_transforms: list[Module] = []
            if self.trainer.config.dataset.random_crop_size:
                image_transforms.append(
                    RandomCrop(size=self.trainer.config.dataset.random_crop_size),
                )
            else:
                image_transforms.append(
                    CenterCrop(size=self.trainer.config.dataset.random_crop_size),
                )
            if self.trainer.config.dataset.horizontal_flip_probability:
                image_transforms.append(
                    RandomHorizontalFlip(p=self.trainer.config.dataset.horizontal_flip_probability),
                )
            image_compose = Compose(image_transforms)
            image = image_compose(image)  # type: ignore

            # encode the image into latent
            latent = self.trainer.lda.encode_image(image=image)  # type: ignore
        else:
            image_embedding = data["image_embedding"]
            latent = data["lda_embedding"]
        text_embedding = data["text_embedding"]
        pooled_text_embedding = data.get("pooled_text_embedding", self.trainer.empty_pooled_text_embedding)
        if not isinstance(pooled_text_embedding, Tensor):
            assert isinstance(pooled_text_embedding, list)
            pooled_text_embedding = Tensor(pooled_text_embedding)

        return IPBatch(
            latent=latent,
            text_embedding=text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            image_embedding=image_embedding,
        )

    def __getitem__(self, index: int) -> IPBatch:
        # retreive data from the huggingface dataset
        data = self.dataset[index]  # type: ignore
        # augment/transform into IPBatch
        data = self.transform(data)  # type: ignore
        return data

    def __len__(self) -> int:
        return len(self.dataset)


class AdapterLatentDiffusionTrainer(Trainer[AdapterLatentDiffusionConfig, IPBatch], WandbMixin):
    def collate_fn(self, batch: list[IPBatch]) -> IPBatch:
        latents = cat(tensors=[item.latent for item in batch])
        text_embeddings = cat(tensors=[item.text_embedding for item in batch])
        pooled_text_embeddings = None
        if self.config.adapter.use_pooled_text_embedding:
            pooled_text_embeddings = cat(tensors=[item.pooled_text_embedding for item in batch])
        image_embeddings = cat([item.image_embedding for item in batch])
        return IPBatch(
            latent=latents,
            text_embedding=text_embeddings,
            pooled_text_embedding=pooled_text_embeddings,
            image_embedding=image_embeddings,
        )
    def collate_fn_from_dict(self, batch: list[dict]) -> IPBatch:
        latents = cat(tensors=[item["latent"][None] for item in batch])
        text_embeddings = cat(tensors=[item["text_embedding"][None] for item in batch])
        pooled_text_embeddings = None
        if self.config.adapter.use_pooled_text_embedding:
            pooled_text_embeddings = cat(tensors=[item["pooled_text_embedding"][None] for item in batch])
        image_embeddings = cat([item["image_embedding"][None] for item in batch])
        return IPBatch(
            latent=latents,
            text_embedding=text_embeddings,
            pooled_text_embedding=pooled_text_embeddings,
            image_embedding=image_embeddings,
        )
    @staticmethod
    def approximate_loss(timestep: int, /) -> float:
        a = 3.1198626909458634e-08
        exponent = 2.3683577564059
        b = -0.3560275587290773
        c = -13.269541143845919
        C = 0.36245161978354973
        return a * timestep**exponent + b * math.exp(-c / (timestep - 1001)) + C
    def drop_latents(self, image_embedding, text_embedding, pooled_text_embedding=None):
        dataset_config = self.config.dataset
        rand_num = random.random()
        if rand_num < dataset_config.image_drop_rate:
            if dataset_config.zero_uncond:
                image_embedding = zeros_like(image_embedding)
            else:
                image_embedding = self.black_image_embedding
        elif rand_num < (dataset_config.image_drop_rate + dataset_config.text_drop_rate):
            text_embedding = self.empty_text_embedding
            if self.config.adapter.use_pooled_text_embedding:
                pooled_text_embedding = self.empty_pooled_text_embedding
        elif rand_num < (
            dataset_config.image_drop_rate + dataset_config.text_drop_rate + dataset_config.text_and_image_drop_rate
        ):
            text_embedding = self.empty_text_embedding
            if self.config.adapter.use_pooled_text_embedding:
                pooled_text_embedding = self.empty_pooled_text_embedding
            if dataset_config.zero_uncond:
                image_embedding = zeros_like(image_embedding)
            else:
                image_embedding = self.black_image_embedding
        return image_embedding, text_embedding, pooled_text_embedding
    @cached_property
    def dataset_length(self) -> int:
        """
        Returns the length of the dataset.

        This is used to compute the number of batches per epoch.
        """
        if self.config.dataset.webdataset:
            return self.config.dataset.num_train_examples
        return len(self.dataset)
    @register_model()
    def lda(self, lda_config: ModelConfig) -> SD1Autoencoder:
        return SD1Autoencoder(
            device=self.device,
        )

    @register_model()
    def unet(self, unet_config: ModelConfig) -> SD1UNet:
        return SD1UNet(
            in_channels=4,  # FIXME: harcoded value
            device=self.device,
        )

    @register_model()
    def text_encoder(self, text_encoder_config: ModelConfig) -> CLIPTextEncoderL | TextEncoderWithPoolingL:
        text_encoder = CLIPTextEncoderL(
            device=self.device,
        )
        if not self.config.adapter.use_pooled_text_embedding:
            return text_encoder
        text_encoder_with_pooling = TextEncoderWithPoolingL(target=text_encoder)
        text_encoder_with_pooling.inject()
        return text_encoder_with_pooling

    @register_model()
    def image_encoder(self, image_encoder_config: ModelConfig) -> ViT:
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
        return image_encoder_cls()

    @register_model()
    def image_proj(self, image_proj_config: ModelConfig) -> ImageProjection | PerceiverResampler:
        cross_attn_2d = self.unet.ensure_find(CrossAttentionBlock2d)
        image_proj = get_sd1_image_proj(
            self.image_encoder, self.unet, cross_attn_2d, self.config.adapter.fine_grained, self.config.adapter.use_bias, device=self.device, dtype=float32
        )
        image_proj.requires_grad_(True)
        for module in image_proj.modules():
            _init_learnable_weights(module, self.config.adapter.initializer_range)
        i=0
        for param in image_proj.parameters():
            if param.requires_grad:
                i += 1
        logger.info(f"Initialized {i} parameters in image_proj")
        empty_cache()
        gc.collect()
        return image_proj

    @register_model()
    def adapter(self, adapter_config: ModelConfig) -> SD1IPAdapter:
        # At the point this gets called the unet, image_encoder, and image_proj will get called thanks to @register_model
        ip_adapter = SD1IPAdapter(
            target=self.unet,
            weights=load_from_safetensors(self.config.adapter.checkpoint)
            if self.config.adapter.checkpoint is not None
            else None,
            strict=False,
            fine_grained=self.config.adapter.fine_grained,
            scale=self.config.adapter.scale,
            use_timestep_embedding=self.config.adapter.use_timestep_embedding,
            use_pooled_text_embedding=self.config.adapter.use_pooled_text_embedding,
            image_encoder=self.image_encoder,
            image_proj=self.image_proj,
            use_bias=self.config.adapter.use_bias,
            layernorm_dino=self.config.adapter.layernorm_dino,
            weighted_sum=self.config.adapter.weighted_sum
        ).inject()
        for adapter in ip_adapter.sub_adapters:
            adapter.image_cross_attention.requires_grad_(True)
            adapter.image_cross_attention.to(self.device, float32)

        for module in ip_adapter.modules():
            _init_learnable_weights(module, self.config.adapter.initializer_range)

        i=0
        for param in ip_adapter.parameters():
            if param.requires_grad:
                i += 1
        logger.info(f"Initialized {i} parameters in ip adapter")
        empty_cache()
        gc.collect()
        return ip_adapter

    @cached_property
    def ddpm_solver(self) -> DDPM:
        return DDPM(
            num_inference_steps=1000,  # FIXME: harcoded value
            device=self.device,
        ).to(device=self.device)

    @cached_property
    def signal_to_noise_ratios(self) -> Tensor:
        return exp(self.ddpm_solver.signal_to_noise_ratios) ** 2

    @cached_property
    def timestep_weights(self) -> Tensor:
        return generate_timestep_weights(self.config.adapter, 1000).to(self.device, dtype=float32)
    def get_constants(self):
        self.cond_resolution: int = self.config.adapter.resolution
        if isinstance(self.text_encoder, TextEncoderWithPoolingL):
            self.empty_text_embedding = self.text_encoder("")[0].float().cpu()
            self.empty_pooled_text_embedding = self.text_encoder("")[1].float().cpu()
        else:
            self.empty_text_embedding = self.text_encoder("").float().cpu()
            self.empty_pooled_text_embedding = self.text_encoder("").float().cpu()[:, 1]
        self.black_image_embedding = self.image_encoder(zeros((1, 3, self.cond_resolution, self.cond_resolution)).to(self.device, dtype=self.dtype)).float().cpu()
    def load_web_dataset(self) -> wds.DataPipeline:
        all_keys = ["text_embedding", "pooled_text_embedding", "latent", "image_embedding"]
        if self.config.adapter.layernorm_dino:
            image_encoder_pth = "dinov2_vitl14_reg4_pretrain.pth"
        else:
            image_encoder_pth = "dinov2_vitl14_reg4_pretrain_no_norm.pth"
        if isinstance(self.text_encoder, CLIPTextEncoderL):
            all_keys.remove("pooled_text_embedding")
        processing_pipeline = [
            wds.decode(wds.handle_extension("pth", wds.autodecode.torch_loads), handler=wds.ignore_and_continue),
            wds.rename(
                text_embedding="CLIPL.pth".lower(),
                pooled_text_embedding="CLIPLPool.pth".lower(),
                latent="sd15_lda.pth",
                image_embedding=image_encoder_pth,
                handler=wds.warn_and_continue,
            ),
            wds.map(filter_keys(set(all_keys))),
        ]
        pipeline = [
            wds.ResampledShards(self.config.dataset.train_shards_path_or_url),
            tarfile_to_samples_nothrow,
            wds.shuffle(self.config.dataset.shuffle_buffer_size),
            *processing_pipeline,
            wds.batched(self.config.training.batch_size, partial=False, collation_fn=self.collate_fn_from_dict),
        ]
        global_batch_size = self.config.training.batch_size
        num_workers = self.config.training.dataset_workers
        num_train_examples = self.config.dataset.num_train_examples
        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))  # per dataloader worker

        # each worker is iterating over this
        return wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
    def load_dataset(self) -> IPDataset:
        self.get_constants()
        if self.config.dataset.webdataset:
            return self.load_web_dataset()
        return IPDataset(trainer=self)
    @cached_property
    def dataloader(self) -> DataLoader[Any]:
        global_batch_size = self.config.training.batch_size
        num_workers = self.config.training.dataset_workers
        num_train_examples = self.config.dataset.num_train_examples
        num_batches = math.ceil(num_train_examples / global_batch_size)
        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size

        if self.config.dataset.webdataset:
            dataloader = wds.WebLoader(
                self.dataset,
                batch_size=None,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
            )
            dataloader.num_batches = num_batches
            dataloader.num_samples = num_samples
            return dataloader
        return DataLoader(
            dataset=self.dataset, batch_size=self.config.training.batch_size, num_workers=self.config.training.dataset_workers, shuffle=True, collate_fn=self.collate_fn
        )
    def sample_timestep(self, batch_size: int, /) -> Tensor:
        """Sample a timestep from a uniform distribution."""
        assert isinstance(self, Trainer), "This mixin can only be used with a Trainer"
        random_steps = 999-multinomial(self.timestep_weights, batch_size, replacement=True).long()
        self.random_steps = random_steps
        return self.ddpm_solver.timesteps[random_steps]

    def add_noise_to_latents(
        self, latents: Tensor, noise: Tensor
    ) -> Tensor:
        """Add noise to latents."""
        return cat(
            [
                self.ddpm_solver.add_noise(
                    latents[i : i + 1], noise[i : i + 1], int(self.random_steps[i].item())
                )
                for i in range(latents.shape[0])
            ],
            dim=0,
        )
    def remove_noise_from_latents(
        self, latents: Tensor, noise: Tensor
    ) -> Tensor:
        """Add noise to latents."""
        return cat(
            [
                self.ddpm_solver.remove_noise(
                    latents[i : i + 1], noise[i : i + 1], int(self.random_steps[i].item())
                )
                for i in range(latents.shape[0])
            ],
            dim=0,
        )
    def sample_noise(self, size: tuple[int, ...], dtype: DType | None = None) -> Tensor:
        return sample_noise(
            size=size,
            offset_noise=self.config.ldm.offset_noise,
            device=self.device,
            dtype=dtype,
        )

    def compute_loss(self, batch: IPBatch) -> Tensor:
        input_dtype = float32 if self.config.training.amp else self.dtype
        # retreive data from batch
        latents = batch.latent.to(self.device, dtype=self.dtype)
        batch_size = latents.shape[0]
        text_embeddings = batch.text_embedding.to(self.device, dtype=self.dtype)
        if self.config.adapter.use_pooled_text_embedding:
            pooled_text_embeddings = batch.pooled_text_embedding.to(self.device, dtype=self.dtype)
        div_factor = self.config.adapter.image_embedding_div_factor
        image_embeddings = batch.image_embedding.to(self.device, dtype=input_dtype)/div_factor
        print(image_embeddings.shape, latents.shape, text_embeddings.shape)
        for i in range(batch_size):
            if self.config.adapter.use_pooled_text_embedding:
                image_embeddings[i], text_embeddings[i], pooled_text_embeddings[i] = self.drop_latents(image_embeddings[i], text_embeddings[i], pooled_text_embeddings[i])
            else:
                image_embeddings[i], text_embeddings[i], _ = self.drop_latents(image_embeddings[i], text_embeddings[i])
        image_embeddings = self.image_proj(image_embeddings)
        # set IP embeddings context
        self.adapter.set_image_embedding(image_embeddings)
        # set pooled text embedding
        if self.config.adapter.use_pooled_text_embedding:
            self.adapter.set_pooled_text_embedding(pooled_text_embeddings/self.config.adapter.pooled_text_div_factor)
        # set text embeddings context
        self.unet.set_clip_text_embedding(clip_text_embedding=text_embeddings)

        # sample timestep and set unet timestep context
        timestep = self.sample_timestep(batch_size)
        self.unet.set_timestep(timestep=timestep)

        # sample noise and noisify the latents
        noise = self.sample_noise(size=latents.shape, dtype=latents.dtype)
        input_perturbation = self.config.ldm.input_perturbation
        if input_perturbation > 0:
            new_noise = noise + input_perturbation * randn_like(noise)
            noisy_latents = self.add_noise_to_latents(latents, new_noise)
        else:
            noisy_latents = self.add_noise_to_latents(latents, noise)
        # get prediction from unet
        prediction = self.unet(noisy_latents)
        # compute mse loss
        snr_gamma = self.config.ldm.snr_gamma
        rescaler = self.config.adapter.use_rescaler
        loss = mse_loss(input=prediction.float(), target=noise.float(), reduction="none")
        if rescaler:
            scales = tensor(
                [self.approximate_loss(999 - int(t.item())) for t in timestep],
                device=self.device,
                dtype=float32,
            ).reshape(-1, 1, 1, 1)
            loss = (loss / scales).mean()
        elif snr_gamma is None:
            loss = loss.mean()
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            signal_to_noise_ratios = self.signal_to_noise_ratios[timestep]

            mse_loss_weights = (
                stack([signal_to_noise_ratios, snr_gamma * ones_like(timestep)], dim=1).min(dim=1)[0]
                / signal_to_noise_ratios
            )
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        return loss

    def compute_evaluation(self) -> None:
        # initialize an SD1.5 pipeline using the trainer's models
        pipeline_dtype = None if self.config.training.amp else self.dtype
        sd = StableDiffusion_1(
            unet=self.unet,
            lda=self.lda,
            solver=DPMSolver(num_inference_steps=self.config.test_ldm.num_inference_steps),
            device=self.device,
            dtype=pipeline_dtype
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
            conditional_embedding = self.text_encoder(prompt)
            negative_embedding = self.text_encoder("")
            if self.config.adapter.use_pooled_text_embedding:
                assert isinstance(self.text_encoder, TextEncoderWithPoolingL)
                assert isinstance(negative_embedding, tuple)
                assert isinstance(negative_embedding[0], Tensor)
                assert isinstance(negative_embedding[1], Tensor)
                assert isinstance(conditional_embedding, tuple)
                assert isinstance(conditional_embedding[0], Tensor)
                assert isinstance(conditional_embedding[1], Tensor)
                clip_text_embedding = cat(tensors=(negative_embedding[0], conditional_embedding[0]), dim=0)
                pooled_clip_text_embedding = cat(tensors=(negative_embedding[1], conditional_embedding[1]), dim=0)
            else:
                assert isinstance(self.text_encoder, CLIPTextEncoderL)
                assert isinstance(negative_embedding, Tensor)
                assert isinstance(conditional_embedding, Tensor)
                clip_text_embedding = cat(tensors=(negative_embedding, conditional_embedding), dim=0)



            cond_resolution = self.config.adapter.resolution
            image_embedding = self.adapter.compute_image_embedding(
                self.adapter.preprocess_image(cond_image, (cond_resolution, cond_resolution)).to(self.device, dtype=self.dtype),
                div_factor=self.config.adapter.image_embedding_div_factor
            )
            # TODO: pool text according to end of text id for pooled text embeds if given option
            for i in range(num_images_per_prompt):
                logger.info(f"Generating image {i+1}/{num_images_per_prompt} for prompt: {prompt}")
                x = randn(1, 4, 64, 64, device=self.device, dtype=self.dtype)
                self.adapter.set_image_embedding(image_embedding)
                if self.config.adapter.use_pooled_text_embedding:
                    self.adapter.set_pooled_text_embedding(pooled_clip_text_embedding/self.config.adapter.pooled_text_div_factor)
                for step in sd.steps:
                    x = sd(
                        x=x,
                        step=step,
                        clip_text_embedding=clip_text_embedding,
                        condition_scale=self.config.test_ldm.condition_scale
                    )
                canvas_image.paste(sd.lda.decode_latents(x=x), box=(0, 512 * i))
            images[prompt] = canvas_image
        # log images to wandb
        self.wandb_log(data=images)
        self.adapter.scale = self.config.adapter.scale
    @register_callback()
    def compute_grad_norms(self, config: CallbackConfig) -> ComputeGradNormCallback:
        return ComputeGradNormCallback()
    @register_callback()
    def compute_param_norms(self, config: CallbackConfig) -> ComputeParamNormCallback:
        return ComputeParamNormCallback()
    @register_callback()
    def save_adapter(self, config: CallbackConfig) -> SaveAdapterCallback:
        return SaveAdapterCallback()

    def __init__(
        self,
        config: AdapterLatentDiffusionConfig,
    ) -> None:
        # if initializing after, the on_init_end methods do not get called for the extended callbacks. So all these callbacks
        # can't have on_init
        super().__init__(config=config)





if __name__ == "__main__":
    import sys

    config_path = sys.argv[1]
    config = AdapterLatentDiffusionConfig.load_from_toml(toml_path=config_path)
    trainer = AdapterLatentDiffusionTrainer(config=config)
    trainer.train()
