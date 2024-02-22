from abc import ABC, abstractmethod
from typing import TypeVar

import torch
from PIL import Image
from torch import Tensor, device as Device, dtype as DType

import refiners.fluxion.layers as fl
from refiners.foundationals.latent_diffusion.auto_encoder import LatentDiffusionAutoencoder
from refiners.foundationals.latent_diffusion.solvers import Solver

TLatentDiffusionModel = TypeVar("TLatentDiffusionModel", bound="LatentDiffusionModel")


class LatentDiffusionModel(fl.Module, ABC):
    def __init__(
        self,
        unet: fl.Chain,
        lda: LatentDiffusionAutoencoder,
        clip_text_encoder: fl.Chain,
        solver: Solver,
        device: Device | str = "cpu",
        dtype: DType = torch.float32,
    ) -> None:
        super().__init__()
        self.device: Device = device if isinstance(device, Device) else Device(device=device)
        self.dtype = dtype
        self.unet = unet.to(device=self.device, dtype=self.dtype)
        self.lda = lda.to(device=self.device, dtype=self.dtype)
        self.clip_text_encoder = clip_text_encoder.to(device=self.device, dtype=self.dtype)
        self.solver = solver.to(device=self.device, dtype=self.dtype)

    def set_inference_steps(self, num_steps: int, first_step: int = 0) -> None:
        self.solver = self.solver.rebuild(num_inference_steps=num_steps, first_inference_step=first_step)

    def init_latents(
        self,
        size: tuple[int, int],
        init_image: Image.Image | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        height, width = size
        if noise is None:
            noise = torch.randn(1, 4, height // 8, width // 8, device=self.device)
        assert list(noise.shape[2:]) == [
            height // 8,
            width // 8,
        ], f"noise shape is not compatible: {noise.shape}, with size: {size}"
        if init_image is None:
            return noise
        encoded_image = self.lda.image_to_latents(image=init_image.resize(size=(width, height)))
        return self.solver.add_noise(
            x=encoded_image,
            noise=noise,
            step=self.solver.first_inference_step,
        )

    @property
    def steps(self) -> list[int]:
        return self.solver.inference_steps

    @abstractmethod
    def set_unet_context(self, *, timestep: Tensor, clip_text_embedding: Tensor, **_: Tensor) -> None:
        ...

    @abstractmethod
    def set_self_attention_guidance(self, enable: bool, scale: float = 1.0) -> None:
        ...

    @abstractmethod
    def has_self_attention_guidance(self) -> bool:
        ...

    @abstractmethod
    def compute_self_attention_guidance(
        self, x: Tensor, noise: Tensor, step: int, *, clip_text_embedding: Tensor, **kwargs: Tensor
    ) -> Tensor:
        ...

    def forward(
        self, x: Tensor, step: int, *, clip_text_embedding: Tensor, condition_scale: float = 7.5, **kwargs: Tensor
    ) -> Tensor:
        timestep = self.solver.timesteps[step].unsqueeze(dim=0)
        self.set_unet_context(timestep=timestep, clip_text_embedding=clip_text_embedding, **kwargs)

        latents = torch.cat(tensors=(x, x))  # for classifier-free guidance
        # scale latents for solvers that need it
        latents = self.solver.scale_model_input(latents, step=step)
        unconditional_prediction, conditional_prediction = self.unet(latents).chunk(2)

        # classifier-free guidance
        predicted_noise = unconditional_prediction + condition_scale * (
            conditional_prediction - unconditional_prediction
        )
        x = x.narrow(dim=1, start=0, length=4)  # support > 4 channels for inpainting

        if self.has_self_attention_guidance():
            predicted_noise += self.compute_self_attention_guidance(
                x=x, noise=unconditional_prediction, step=step, clip_text_embedding=clip_text_embedding, **kwargs
            )

        return self.solver(x, predicted_noise=predicted_noise, step=step)

    def structural_copy(self: TLatentDiffusionModel) -> TLatentDiffusionModel:
        return self.__class__(
            unet=self.unet.structural_copy(),
            lda=self.lda.structural_copy(),
            clip_text_encoder=self.clip_text_encoder.structural_copy(),
            solver=self.solver,
            device=self.device,
            dtype=self.dtype,
        )
