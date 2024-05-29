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
        classifier_free_guidance: bool = True,
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
        self.classifier_free_guidance = classifier_free_guidance

    def set_inference_steps(self, num_steps: int, first_step: int = 0) -> None:
        """Set the steps of the diffusion process.

        Args:
            num_steps: The number of inference steps.
            first_step: The first inference step, used for image-to-image diffusion.
                You may be used to setting a float in `[0, 1]` called `strength` instead,
                which is an abstraction for this. The first step is
                `round((1 - strength) * (num_steps - 1))`.
        """
        self.solver = self.solver.rebuild(num_inference_steps=num_steps, first_inference_step=first_step)

    @staticmethod
    def sample_noise(
        size: tuple[int, ...],
        device: Device | None = None,
        dtype: DType | None = None,
        offset_noise: float | None = None,
    ) -> torch.Tensor:
        """Sample noise from a normal distribution with an optional offset.

        Args:
            size: The size of the noise tensor.
            device: The device to put the noise tensor on.
            dtype: The data type of the noise tensor.
            offset_noise: The offset of the noise tensor.
                Useful at training time, see https://www.crosslabs.org/blog/diffusion-with-offset-noise.
        """
        noise = torch.randn(size=size, device=device, dtype=dtype)
        if offset_noise is not None:
            noise += offset_noise * torch.randn(size=(size[0], size[1], 1, 1), device=device, dtype=dtype)
        return noise

    def init_latents(
        self,
        size: tuple[int, int],
        init_image: Image.Image | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Initialize the latents for the diffusion process.

        Args:
            size: The size of the latent (in pixel space).
            init_image: The image to use as initialization for the latents.
            noise: The noise to add to the latents.
        """
        height, width = size
        latent_height = height // 8
        latent_width = width // 8

        if noise is None:
            noise = LatentDiffusionModel.sample_noise(
                size=(1, 4, latent_height, latent_width),
                device=self.device,
                dtype=self.dtype,
            )

        assert list(noise.shape[2:]) == [
            latent_height,
            latent_width,
        ], f"noise shape is not compatible: {noise.shape}, with size: {size}"

        if init_image is None:
            latent = noise
        else:
            resized = init_image.resize(size=(width, height))  # type: ignore
            encoded_image = self.lda.image_to_latents(resized)
            latent = self.solver.add_noise(
                x=encoded_image,
                noise=noise,
                step=self.solver.first_inference_step,
            )

        return self.solver.scale_model_input(latent, step=-1)

    @property
    def steps(self) -> list[int]:
        return self.solver.inference_steps

    @abstractmethod
    def set_unet_context(self, *, timestep: Tensor, clip_text_embedding: Tensor, **_: Tensor) -> None: ...

    @abstractmethod
    def set_self_attention_guidance(self, enable: bool, scale: float = 1.0) -> None: ...

    @abstractmethod
    def has_self_attention_guidance(self) -> bool: ...

    @abstractmethod
    def compute_self_attention_guidance(
        self, x: Tensor, noise: Tensor, step: int, *, clip_text_embedding: Tensor, **kwargs: Tensor
    ) -> Tensor: ...

    def forward(
        self, x: Tensor, step: int, *, clip_text_embedding: Tensor, condition_scale: float = 7.5, **kwargs: Tensor
    ) -> Tensor:
        if self.classifier_free_guidance:
            assert clip_text_embedding.shape[0] % 2 == 0, f"invalid batch size: {clip_text_embedding.shape[0]}"

        timestep = self.solver.timesteps[step].unsqueeze(dim=0)
        self.set_unet_context(timestep=timestep, clip_text_embedding=clip_text_embedding, **kwargs)

        latents = torch.cat(tensors=(x, x)) if self.classifier_free_guidance else x
        # scale latents for solvers that need it
        latents = self.solver.scale_model_input(latents, step=step)

        if self.classifier_free_guidance:
            unconditional_prediction, conditional_prediction = self.unet(latents).chunk(2)
            predicted_noise = unconditional_prediction + condition_scale * (
                conditional_prediction - unconditional_prediction
            )
            x = x.narrow(dim=1, start=0, length=4)  # support > 4 channels for inpainting
            if self.has_self_attention_guidance():
                predicted_noise += self.compute_self_attention_guidance(
                    x=x,
                    noise=unconditional_prediction,
                    step=step,
                    clip_text_embedding=clip_text_embedding,
                    **kwargs,
                )
        else:
            predicted_noise = self.unet(latents)
            x = x.narrow(dim=1, start=0, length=4)  # support > 4 channels for inpainting

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
