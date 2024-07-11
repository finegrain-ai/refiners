from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch import Tensor
from typing_extensions import TypeVar

from refiners.fluxion.utils import image_to_tensor, load_from_safetensors, manual_seed, no_grad
from refiners.foundationals.clip.concepts import ConceptExtender
from refiners.foundationals.latent_diffusion.lora import SDLoraManager
from refiners.foundationals.latent_diffusion.multi_diffusion import DiffusionTarget, MultiDiffusion, Size
from refiners.foundationals.latent_diffusion.solvers.dpm import DPMSolver
from refiners.foundationals.latent_diffusion.solvers.solver import Solver
from refiners.foundationals.latent_diffusion.stable_diffusion_1.controlnet import SD1ControlnetAdapter
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import (
    StableDiffusion_1,
)

Name = str


@dataclass(kw_only=True)
class UpscalerCheckpoints:
    """
    Checkpoints for the multi upscaler.

    Attributes:
        unet: The path to the Stable Diffusion 1 UNet checkpoint.
        clip_text_encoder: The path to the CLIP text encoder checkpoint.
        lda: The path to the LDA checkpoint.
        controlnet_tile: The path to the controlnet tile checkpoint.
        negative_embedding: The path to the negative embedding checkpoint.
        negative_embedding_key: The key for the negative embedding. If the negative embedding is a dictionary, this
            key is used to access the negative embedding. You can use a dot-separated path to access nested dictionaries.
        loras: A dictionary of LORAs to load. The key is the name of the LORA and the value is the path to the LORA
            checkpoint.
    """

    unet: Path
    clip_text_encoder: Path
    lda: Path
    controlnet_tile: Path
    negative_embedding: Path | None = None
    negative_embedding_key: str | None = None
    loras: dict[Name, Path] | None = None


@dataclass(kw_only=True)
class UpscalerTarget(DiffusionTarget):
    clip_text_embedding: Tensor
    controlnet_condition: Tensor
    condition_scale: float = 7.0


T = TypeVar("T", bound=DiffusionTarget)


class MultiUpscalerAbstract(MultiDiffusion[T], ABC):
    def __init__(self, checkpoints: UpscalerCheckpoints, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        self.sd = self.load_stable_diffusion(checkpoints)
        self.manager = self.load_loras(checkpoints.loras)
        self.controlnet = self.load_controlnet(checkpoints)
        self.negative_embedding_token = self.load_negative_embedding(
            checkpoints.negative_embedding, checkpoints.negative_embedding_key
        )

    @abstractmethod
    def compute_targets(
        self,
        image: Image.Image,
        latent_size: Size,
        tile_size: Size,
        num_inference_steps: int,
        first_step: int,
        condition_scale: float,
        clip_text_embedding: torch.Tensor,
    ) -> Sequence[T]: ...

    @abstractmethod
    def diffuse_target(self, x: Tensor, step: int, target: T) -> Tensor: ...

    def load_stable_diffusion(self, checkpoints: UpscalerCheckpoints) -> StableDiffusion_1:
        sd = StableDiffusion_1(device=self.device, dtype=self.dtype)
        sd.unet.load_from_safetensors(checkpoints.unet)
        sd.clip_text_encoder.load_from_safetensors(checkpoints.clip_text_encoder)
        sd.lda.load_from_safetensors(checkpoints.lda)
        return sd

    def load_controlnet(self, checkpoints: UpscalerCheckpoints) -> SD1ControlnetAdapter:
        return SD1ControlnetAdapter(
            target=self.sd.unet,
            name="tile",
            weights=load_from_safetensors(checkpoints.controlnet_tile),
        ).inject()

    def load_loras(self, loras: dict[Name, Path] | None) -> SDLoraManager | None:
        if loras is None:
            return

        manager = SDLoraManager(self.sd)
        for name, path in loras.items():
            manager.add_loras(name, tensors=load_from_safetensors(path))

        return manager

    def load_negative_embedding(self, path: Path | None, key: str | None) -> str:
        if path is None:
            return ""

        embeddings: Tensor | dict[str, Any] = torch.load(path, weights_only=True)  # type: ignore

        if isinstance(embeddings, dict):
            assert key is not None, "Key must be provided to access the negative embedding."
            key_sequence = key.split(".")
            for key in key_sequence:
                assert (
                    key in embeddings
                ), f"Key {key} not found in the negative embedding dictionary. Available keys: {list(embeddings.keys())}"
                embeddings = embeddings[key]

        assert isinstance(
            embeddings, torch.Tensor
        ), f"The negative embedding must be a tensor, found {type(embeddings)}."
        assert embeddings.ndim == 2, f"The negative embedding must be a 2D tensor, found {embeddings.ndim}D tensor."

        extender = ConceptExtender(self.sd.clip_text_encoder)
        negative_embedding_token = ", "
        for i, embedding in enumerate(embeddings):
            extender.add_concept(token=f"<{i}>", embedding=embedding)
            negative_embedding_token += f"<{i}> "
        extender.inject()

        return negative_embedding_token

    @no_grad()
    def compute_clip_text_embedding(self, prompt: str, negative_prompt: str, offload_to_cpu: bool = True) -> Tensor:
        """Compute the CLIP text embedding for the prompt and negative prompt.

        Args:
            prompt: The prompt to use for the upscaling.
            negative_prompt: The negative prompt to use for the upscaling.
            offload_to_cpu: Whether to offload the model to the CPU after computing the embedding.
        """
        if self.negative_embedding_token:
            negative_prompt += self.negative_embedding_token

        self.sd.clip_text_encoder.to(device=self.device, dtype=self.dtype)
        clip_text_embedding = self.sd.compute_clip_text_embedding(
            text=prompt,
            negative_text=negative_prompt,
        )
        if offload_to_cpu:
            self.sd.clip_text_encoder.to(torch.device("cpu"))

        return clip_text_embedding

    def diffuse_upscaler_target(self, x: Tensor, step: int, target: UpscalerTarget) -> Tensor:
        self.sd.solver = target.solver
        self.controlnet.set_controlnet_condition(target.controlnet_condition)
        return self.sd(
            x=x,
            step=step,
            clip_text_embedding=target.clip_text_embedding,
            condition_scale=target.condition_scale,
        )

    @staticmethod
    def resize_modulo_8(image: Image.Image, size: int = 768, on_short: bool = True) -> Image.Image:
        """
        Resize an image respecting the aspect ratio and ensuring the size is a multiple of 8.

        The `on_short` parameter determines whether the resizing is based on the shortest side.
        """
        assert size % 8 == 0, "Size must be a multiple of 8 because this is the latent compression size."
        side_size = min(image.size) if on_short else max(image.size)
        scale = size / (side_size * 8)
        new_size = (int(image.width * scale) * 8, int(image.height * scale) * 8)
        return image.resize(new_size, resample=Image.Resampling.LANCZOS)  # type: ignore

    @no_grad()
    def pre_upscale(self, image: Image.Image, upscale_factor: float, **_: Any) -> Image.Image:
        """
        Pre-upscale an image before the actual upscaling process.

        You can override this method to implement custom pre-upscaling logic like using a ESRGAN model like in the
        original implementation.
        """

        return image.resize(
            (int(image.width * upscale_factor), int(image.height * upscale_factor)),
            resample=Image.Resampling.LANCZOS,  # type: ignore
        )

    def compute_upscaler_targets(
        self,
        image: Image.Image,
        latent_size: Size,
        tile_size: Size,
        num_inference_steps: int,
        first_step: int,
        condition_scale: float,
        clip_text_embedding: torch.Tensor,
    ) -> Sequence[UpscalerTarget]:
        tiles = MultiDiffusion.generate_latent_tiles(size=latent_size, tile_size=tile_size, min_overlap=8)

        targets: Sequence[UpscalerTarget] = []
        for tile in tiles:
            pixel_box = (tile.left * 8, tile.top * 8, tile.right * 8, tile.bottom * 8)
            pixel_tile = image.crop(pixel_box)
            solver = self.sd.solver.rebuild(num_inference_steps=num_inference_steps, first_inference_step=first_step)

            target = UpscalerTarget(
                tile=tile,
                solver=solver,
                start_step=first_step,
                condition_scale=condition_scale,
                controlnet_condition=image_to_tensor(pixel_tile, device=self.device, dtype=self.dtype),
                clip_text_embedding=clip_text_embedding,
            )
            targets.append(target)

        return targets

    def diffuse_targets(
        self,
        targets: Sequence[T],
        image: Image.Image,
        latent_size: Size,
        first_step: int,
        autoencoder_tile_length: int,
    ) -> Image.Image:
        noise = torch.randn(size=(1, 4, *latent_size), device=self.device, dtype=self.dtype)
        with self.sd.lda.tiled_inference(image, (autoencoder_tile_length, autoencoder_tile_length)):
            latents = self.sd.lda.tiled_image_to_latents(image)
            x = self.sd.solver.add_noise(x=latents, noise=noise, step=first_step)

            for step in self.sd.steps:
                x = self(x, noise=noise, step=step, targets=targets)

            return self.sd.lda.tiled_latents_to_image(x)

    @no_grad()
    def upscale(
        self,
        image: Image.Image,
        prompt: str = "masterpiece, best quality, highres",
        negative_prompt: str = "worst quality, low quality, normal quality",
        upscale_factor: float = 2,
        downscale_size: int = 768,
        tile_size: tuple[int, int] = (144, 112),
        denoise_strength: float = 0.35,
        condition_scale: float = 6,
        controlnet_scale: float = 0.6,
        controlnet_scale_decay: float = 0.825,
        loras_scale: dict[Name, float] | None = None,
        solver_type: type[Solver] = DPMSolver,
        num_inference_steps: int = 18,
        autoencoder_tile_length: int = 1024,
        seed: int = 37,
    ) -> Image.Image:
        """
        Upscale an image using the multi upscaler.

        Default settings follow closely to the original implementation https://github.com/philz1337x/clarity-upscaler/

        Args:
            image: The image to upscale.
            prompt: The prompt to use for the upscaling.
            negative_prompt: The negative prompt to use for the upscaling. Original default has a weight of 2.0, but
                using prompt weighting is no supported yet in Refiners.
            upscale_factor: The factor to upscale the image by.
            downscale_size: The size to downscale the image along is short side to before upscaling. Must be a
                multiple of 8 because of latent compression.
            tile_size: The size (H, W) of the tiles to use for latent diffusion. The smaller the tile size, the more "fractal"
                the upscaling will be.
            denoise_strength: The strength of the denoising. A value of 0.0 means no denoising (so nothing happens),
                while a value of 1.0 means full denoising and maximum creativity.
            condition_scale: The scale of the condition. Higher values will create images with more contrast. This
                parameter is called "dynamic" or "HDR" in the original implementation.
            controlnet_scale: The scale of the Tile Controlnet. This parameter is called "resemblance" in the original
                implementation.
            controlnet_scale_decay: Applies an exponential decay to the controlnet scale on the blocks of the UNet. This
                has the effect of diminishing the controlnet scale in a subtle way. The default value is 0.825
                corresponding to the "Prompt is more important" parameter in the original implementation.
            loras_scale: The scale of the LORAs. This is a dictionary where the key is the name of the LORA and the value
                is the scale.
            solver_type: The type of solver to use for the latent diffusion. The default is the DPM solver.
            num_inference_steps: The number of inference steps to use for the latent diffusion. This is a trade-off
                between quality and speed.
            autoencoder_tile_length: The length of the autoencoder tiles. It shouldn't affect the end result, but
                lowering it can reduce GPU memory usage (but increase computation time).
            seed: The seed to use for the random number generator.
        """
        manual_seed(seed)

        # update controlnet scale
        self.controlnet.scale = controlnet_scale
        self.controlnet.scale_decay = controlnet_scale_decay

        # update LoRA scales
        if self.manager is not None and loras_scale is not None:
            self.manager.update_scales(loras_scale)

        # update the solver
        first_step = int(num_inference_steps * (1 - denoise_strength))
        self.sd.solver = solver_type(
            num_inference_steps=num_inference_steps,
            first_inference_step=first_step,
            device=self.device,
            dtype=self.dtype,
        )

        # compute clip text embedding
        clip_text_embedding = self.compute_clip_text_embedding(prompt=prompt, negative_prompt=negative_prompt)

        # prepare the image for the upscale
        image = self.resize_modulo_8(image, size=downscale_size)
        image = self.pre_upscale(image, upscale_factor=upscale_factor)

        # compute the latent size and tile size
        latent_size = Size(height=image.height // 8, width=image.width // 8)
        tile_size = Size(height=tile_size[0], width=tile_size[1])

        # split the image into tiles
        targets: Sequence[DiffusionTarget] = self.compute_targets(
            image=image,
            latent_size=latent_size,
            tile_size=tile_size,
            num_inference_steps=num_inference_steps,
            first_step=first_step,
            condition_scale=condition_scale,
            clip_text_embedding=clip_text_embedding,
        )

        # diffuse the tiles
        return self.diffuse_targets(
            targets=targets,
            image=image,
            latent_size=latent_size,
            first_step=first_step,
            autoencoder_tile_length=autoencoder_tile_length,
        )


class MultiUpscaler(MultiUpscalerAbstract[UpscalerTarget]):
    def diffuse_target(self, x: Tensor, step: int, target: UpscalerTarget) -> Tensor:
        return self.diffuse_upscaler_target(x=x, step=step, target=target)

    def compute_targets(
        self,
        image: Image.Image,
        latent_size: Size,
        tile_size: Size,
        num_inference_steps: int,
        first_step: int,
        condition_scale: float,
        clip_text_embedding: torch.Tensor,
    ) -> Sequence[UpscalerTarget]:
        return self.compute_upscaler_targets(
            image=image,
            latent_size=latent_size,
            tile_size=tile_size,
            num_inference_steps=num_inference_steps,
            first_step=first_step,
            condition_scale=condition_scale,
            clip_text_embedding=clip_text_embedding,
        )
