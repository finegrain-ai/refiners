import torch
from refiners.fluxion.utils import image_to_tensor, interpolate
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
from refiners.foundationals.latent_diffusion.auto_encoder import LatentDiffusionAutoencoder
from refiners.foundationals.latent_diffusion.model import LatentDiffusionModel
from refiners.foundationals.latent_diffusion.schedulers.dpm_solver import DPMSolver
from refiners.foundationals.latent_diffusion.schedulers.scheduler import Scheduler
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
from PIL import Image
import numpy as np
from torch import device as Device, dtype as DType, Tensor


class StableDiffusion_1(LatentDiffusionModel):
    unet: SD1UNet
    clip_text_encoder: CLIPTextEncoderL

    def __init__(
        self,
        unet: SD1UNet | None = None,
        lda: LatentDiffusionAutoencoder | None = None,
        clip_text_encoder: CLIPTextEncoderL | None = None,
        scheduler: Scheduler | None = None,
        device: Device | str = "cpu",
        dtype: DType = torch.float32,
    ) -> None:
        unet = unet or SD1UNet(in_channels=4)
        lda = lda or LatentDiffusionAutoencoder()
        clip_text_encoder = clip_text_encoder or CLIPTextEncoderL()
        scheduler = scheduler or DPMSolver(num_inference_steps=30)

        super().__init__(
            unet=unet,
            lda=lda,
            clip_text_encoder=clip_text_encoder,
            scheduler=scheduler,
            device=device,
            dtype=dtype,
        )

    def compute_clip_text_embedding(self, text: str, negative_text: str = "") -> Tensor:
        conditional_embedding = self.clip_text_encoder(text)
        if text == negative_text:
            return torch.cat(tensors=(conditional_embedding, conditional_embedding), dim=0)

        negative_embedding = self.clip_text_encoder(negative_text or "")
        return torch.cat(tensors=(negative_embedding, conditional_embedding), dim=0)

    def set_unet_context(self, timestep: Tensor, clip_text_embedding: Tensor, *_: Tensor) -> None:
        self.unet.set_timestep(timestep=timestep)
        self.unet.set_clip_text_embedding(clip_text_embedding=clip_text_embedding)


class StableDiffusion_1_Inpainting(StableDiffusion_1):
    def __init__(
        self,
        unet: SD1UNet | None = None,
        lda: LatentDiffusionAutoencoder | None = None,
        clip_text_encoder: CLIPTextEncoderL | None = None,
        scheduler: Scheduler | None = None,
        device: Device | str = "cpu",
        dtype: DType = torch.float32,
    ) -> None:
        self.mask_latents: Tensor | None = None
        self.target_image_latents: Tensor | None = None
        super().__init__(
            unet=unet, lda=lda, clip_text_encoder=clip_text_encoder, scheduler=scheduler, device=device, dtype=dtype
        )

    def forward(
        self,
        x: Tensor,
        step: int,
        clip_text_embedding: Tensor,
        *_: Tensor,
        condition_scale: float = 7.5,
    ) -> Tensor:
        assert self.mask_latents is not None
        assert self.target_image_latents is not None
        x = torch.cat(tensors=(x, self.mask_latents, self.target_image_latents), dim=1)
        return super().forward(
            x=x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=condition_scale,
        )

    def set_inpainting_conditions(
        self,
        target_image: Image.Image,
        mask: Image.Image,
        latents_size: tuple[int, int] = (64, 64),
    ) -> tuple[Tensor, Tensor]:
        target_image = target_image.convert(mode="RGB")
        mask = mask.convert(mode="L")

        mask_tensor = torch.tensor(data=np.array(object=mask).astype(dtype=np.float32) / 255.0).to(device=self.device)
        mask_tensor = (mask_tensor > 0.5).unsqueeze(dim=0).unsqueeze(dim=0).to(dtype=self.dtype)
        self.mask_latents = interpolate(x=mask_tensor, factor=torch.Size(latents_size))

        init_image_tensor = image_to_tensor(image=target_image, device=self.device, dtype=self.dtype) * 2 - 1
        masked_init_image = init_image_tensor * (1 - mask_tensor)
        self.target_image_latents = self.lda.encode(x=masked_init_image)

        return self.mask_latents, self.target_image_latents
