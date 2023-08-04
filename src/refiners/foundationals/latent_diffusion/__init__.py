from typing import TypeVar
from torch import cat, float32, randn, tensor, device as Device, dtype as DType, Size, Tensor
from PIL import Image
import numpy as np

from refiners.fluxion.utils import image_to_tensor, interpolate
from refiners.fluxion.layers.module import Module
from refiners.foundationals.latent_diffusion.auto_encoder import (
    LatentDiffusionAutoencoder,
)
from refiners.foundationals.clip.text_encoder import (
    CLIPTextEncoder,
    CLIPTextEncoderL,
)
from refiners.foundationals.latent_diffusion.schedulers import Scheduler, DPMSolver
from refiners.foundationals.latent_diffusion.unet import UNet


TLatentDiffusionModel = TypeVar("TLatentDiffusionModel", bound="LatentDiffusionModel")

__all__ = [
    "LatentDiffusionModel",
    "UNet",
    "DPMSolver",
    "Scheduler",
    "CLIPTextEncoder",
    "LatentDiffusionAutoencoder",
]


class LatentDiffusionModel(Module):
    def __init__(
        self,
        unet: UNet,
        lda: LatentDiffusionAutoencoder,
        clip_text_encoder: CLIPTextEncoder,
        scheduler: Scheduler,
        device: Device | str = "cpu",
        dtype: DType = float32,
    ):
        super().__init__()
        self.device: Device = device if isinstance(device, Device) else Device(device)
        self.dtype = dtype
        self.unet = unet.to(self.device, dtype=self.dtype)
        self.lda = lda.to(self.device, dtype=self.dtype)
        self.clip_text_encoder = clip_text_encoder.to(self.device, dtype=self.dtype)
        self.scheduler = scheduler.to(self.device, dtype=self.dtype)

    def set_num_inference_steps(self, num_inference_steps: int):
        initial_diffusion_rate = self.scheduler.initial_diffusion_rate
        final_diffusion_rate = self.scheduler.final_diffusion_rate
        device, dtype = self.scheduler.device, self.scheduler.dtype
        self.scheduler = self.scheduler.__class__(
            num_inference_steps,
            initial_diffusion_rate=initial_diffusion_rate,
            final_diffusion_rate=final_diffusion_rate,
        ).to(device=device, dtype=dtype)

    def init_latents(
        self,
        size: tuple[int, int],
        init_image: Image.Image | None = None,
        first_step: int = 0,
        noise: Tensor | None = None,
    ) -> Tensor:
        if noise is None:
            height, width = size
            noise = randn(1, 4, height // 8, width // 8, device=self.device)
        assert list(noise.shape[2:]) == [
            size[0] // 8,
            size[1] // 8,
        ], f"noise shape is not compatible: {noise.shape}, with size: {size}"
        if init_image is None:
            return noise
        encoded_image = self.lda.encode_image(init_image.resize(size))
        return self.scheduler.add_noise(encoded_image, noise, self.steps[first_step])

    @property
    def steps(self) -> list[int]:
        return self.scheduler.steps

    @property
    def timestep_embeddings(self) -> Tensor:
        return self.timestep_encoder(self.scheduler.timesteps)

    @property
    def unconditional_clip_text_embeddings(self) -> Tensor:
        return self.clip_text_encoder.unconditional_text_embedding

    def compute_text_embedding(self, text: str) -> Tensor:
        return self.clip_text_encoder.encode(text)

    def forward(
        self,
        x: Tensor,
        step: int,
        clip_text_embedding: Tensor,
        negative_clip_text_embedding: Tensor | None = None,
        condition_scale: float = 7.5,
    ) -> Tensor:
        timestep = self.scheduler.timesteps[step].unsqueeze(0)
        self.unet.set_timestep(timestep)

        negative_clip_text_embedding = (
            self.clip_text_encoder.unconditional_text_embedding
            if negative_clip_text_embedding is None
            else negative_clip_text_embedding
        )

        clip_text_embeddings = cat((negative_clip_text_embedding, clip_text_embedding))

        self.unet.set_clip_text_embedding(clip_text_embeddings)
        latents = cat((x, x))  # for classifier-free guidance
        unconditional_prediction, conditional_prediction = self.unet(latents).chunk(2)

        # classifier-free guidance
        noise = unconditional_prediction + condition_scale * (conditional_prediction - unconditional_prediction)
        x = x.narrow(dim=1, start=0, length=4)  # support > 4 channels for inpainting
        return self.scheduler(x, noise=noise, step=step)

    def structural_copy(self: TLatentDiffusionModel) -> TLatentDiffusionModel:
        return self.__class__(
            unet=self.unet.structural_copy(),
            lda=self.lda.structural_copy(),
            clip_text_encoder=self.clip_text_encoder.structural_copy(),
            scheduler=self.scheduler,
            device=self.device,
            dtype=self.dtype,
        )


class StableDiffusion_1(LatentDiffusionModel):
    def __init__(
        self,
        unet: UNet | None = None,
        lda: LatentDiffusionAutoencoder | None = None,
        clip_text_encoder: CLIPTextEncoderL | None = None,
        scheduler: Scheduler | None = None,
        device: Device | str = "cpu",
        dtype: DType = float32,
    ):
        unet = unet or UNet(in_channels=4, clip_embedding_dim=768)
        lda = lda or LatentDiffusionAutoencoder()
        clip_text_encoder = clip_text_encoder or CLIPTextEncoderL()
        scheduler = scheduler or DPMSolver(num_inference_steps=30)

        super().__init__(
            unet,
            lda,
            clip_text_encoder=clip_text_encoder,
            scheduler=scheduler,
            device=device,
            dtype=dtype,
        )


class StableDiffusion_1_Inpainting(StableDiffusion_1):
    def __init__(
        self,
        unet: UNet | None = None,
        lda: LatentDiffusionAutoencoder | None = None,
        clip_text_encoder: CLIPTextEncoderL | None = None,
        scheduler: Scheduler | None = None,
        device: Device | str = "cpu",
        dtype: DType = float32,
    ):
        self.mask_latents: Tensor | None = None
        self.target_image_latents: Tensor | None = None
        super().__init__(unet, lda, clip_text_encoder, scheduler, device, dtype)

    def forward(
        self,
        x: Tensor,
        step: int,
        clip_text_embedding: Tensor,
        negative_clip_text_embedding: Tensor | None = None,
        condition_scale: float = 7.5,
    ):
        assert self.mask_latents is not None
        assert self.target_image_latents is not None
        x = cat((x, self.mask_latents, self.target_image_latents), dim=1)
        return super().forward(x, step, clip_text_embedding, negative_clip_text_embedding, condition_scale)

    def set_inpainting_conditions(
        self,
        target_image: Image.Image,
        mask: Image.Image,
        latents_size: tuple[int, int] = (64, 64),
    ) -> tuple[Tensor, Tensor]:
        target_image = target_image.convert("RGB")
        mask = mask.convert("L")

        mask_tensor = tensor(np.array(mask).astype(np.float32) / 255.0).to(self.device)
        mask_tensor = (mask_tensor > 0.5).unsqueeze(0).unsqueeze(0).to(dtype=self.dtype)
        self.mask_latents = interpolate(mask_tensor, Size(latents_size))

        init_image_tensor = image_to_tensor(target_image, device=self.device, dtype=self.dtype) * 2 - 1
        masked_init_image = init_image_tensor * (1 - mask_tensor)
        self.target_image_latents = self.lda.encode(masked_init_image)

        return self.mask_latents, self.target_image_latents
