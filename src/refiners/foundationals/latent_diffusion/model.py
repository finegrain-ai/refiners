from abc import ABC, abstractmethod
from typing import Protocol, TypeVar
from torch import Tensor, device as Device, dtype as DType
from PIL import Image
import torch
import refiners.fluxion.layers as fl
from refiners.foundationals.latent_diffusion.auto_encoder import LatentDiffusionAutoencoder
from refiners.foundationals.latent_diffusion.schedulers.scheduler import Scheduler


T = TypeVar("T", bound="fl.Module")


class UNetInterface(Protocol):
    def set_timestep(self, timestep: Tensor) -> None:
        ...

    def set_clip_text_embedding(self, clip_text_embedding: Tensor) -> None:
        ...

    def __call__(self, x: Tensor) -> Tensor:
        ...


class TextEncoderInterface(Protocol):
    def __call__(self, text: str) -> Tensor | tuple[Tensor, Tensor]:
        ...


TLatentDiffusionModel = TypeVar("TLatentDiffusionModel", bound="LatentDiffusionModel")


class LatentDiffusionModel(fl.Module, ABC):
    def __init__(
        self,
        unet: UNetInterface,
        lda: LatentDiffusionAutoencoder,
        clip_text_encoder: TextEncoderInterface,
        scheduler: Scheduler,
        device: Device | str = "cpu",
        dtype: DType = torch.float32,
    ) -> None:
        super().__init__()
        self.device: Device = device if isinstance(device, Device) else Device(device=device)
        self.dtype = dtype
        assert isinstance(unet, fl.Module)
        self.unet = unet.to(device=self.device, dtype=self.dtype)
        self.lda = lda.to(device=self.device, dtype=self.dtype)
        assert isinstance(clip_text_encoder, fl.Module)
        self.clip_text_encoder = clip_text_encoder.to(device=self.device, dtype=self.dtype)
        self.scheduler = scheduler.to(device=self.device, dtype=self.dtype)

    def set_num_inference_steps(self, num_inference_steps: int) -> None:
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
            noise = torch.randn(1, 4, height // 8, width // 8, device=self.device)
        assert list(noise.shape[2:]) == [
            size[0] // 8,
            size[1] // 8,
        ], f"noise shape is not compatible: {noise.shape}, with size: {size}"
        if init_image is None:
            return noise
        encoded_image = self.lda.encode_image(image=init_image.resize(size=size))
        return self.scheduler.add_noise(x=encoded_image, noise=noise, step=self.steps[first_step])

    @property
    def steps(self) -> list[int]:
        return self.scheduler.steps

    @abstractmethod
    def set_unet_context(self, timestep: Tensor, clip_text_embedding: Tensor, *args: Tensor) -> None:
        ...

    def forward(
        self,
        x: Tensor,
        step: int,
        clip_text_embedding: Tensor,
        *args: Tensor,
        condition_scale: float = 7.5,
    ) -> Tensor:
        timestep = self.scheduler.timesteps[step].unsqueeze(dim=0)
        self.set_unet_context(timestep=timestep, clip_text_embedding=clip_text_embedding, *args)

        latents = torch.cat(tensors=(x, x))  # for classifier-free guidance
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
