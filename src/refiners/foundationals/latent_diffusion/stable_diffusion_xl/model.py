import torch
from torch import Tensor, device as Device, dtype as DType

from refiners.foundationals.latent_diffusion.auto_encoder import LatentDiffusionAutoencoder
from refiners.foundationals.latent_diffusion.model import LatentDiffusionModel
from refiners.foundationals.latent_diffusion.solvers import DDIM, Solver
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.self_attention_guidance import SDXLSAGAdapter
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.text_encoder import DoubleTextEncoder
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet


class SDXLAutoencoder(LatentDiffusionAutoencoder):
    """Stable Diffusion XL autoencoder model.

    Attributes:
        encoder_scale: The encoder scale to use.
    """

    encoder_scale: float = 0.13025


class StableDiffusion_XL(LatentDiffusionModel):
    """Stable Diffusion XL model.

    Attributes:
        unet: The U-Net model.
        clip_text_encoder: The text encoder.
        lda: The image autoencoder.
    """

    unet: SDXLUNet
    clip_text_encoder: DoubleTextEncoder
    lda: SDXLAutoencoder

    def __init__(
        self,
        unet: SDXLUNet | None = None,
        lda: SDXLAutoencoder | None = None,
        clip_text_encoder: DoubleTextEncoder | None = None,
        solver: Solver | None = None,
        device: Device | str = "cpu",
        dtype: DType = torch.float32,
    ) -> None:
        """Initializes the model.

        Args:
            unet: The SDXLUNet U-Net model to use.
            lda: The SDXLAutoencoder image autoencoder to use.
            clip_text_encoder: The DoubleTextEncoder text encoder to use.
            solver: The solver to use.
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """
        unet = unet or SDXLUNet(in_channels=4)
        lda = lda or SDXLAutoencoder()
        clip_text_encoder = clip_text_encoder or DoubleTextEncoder()
        solver = solver or DDIM(num_inference_steps=30)

        super().__init__(
            unet=unet,
            lda=lda,
            clip_text_encoder=clip_text_encoder,
            solver=solver,
            device=device,
            dtype=dtype,
        )

    def compute_clip_text_embedding(self, text: str, negative_text: str | None = None) -> tuple[Tensor, Tensor]:
        """Compute the CLIP text embedding associated with the given prompt and negative prompt.

        Args:
            text: The prompt to compute the CLIP text embedding of.
            negative_text: The negative prompt to compute the CLIP text embedding of.
                If not provided, the negative prompt is assumed to be empty (i.e., `""`).
        """
        conditional_embedding, conditional_pooled_embedding = self.clip_text_encoder(text)
        if text == negative_text:
            return torch.cat(tensors=(conditional_embedding, conditional_embedding), dim=0), torch.cat(
                tensors=(conditional_pooled_embedding, conditional_pooled_embedding), dim=0
            )

        # TODO: when negative_text is None, use zero tensor?
        negative_embedding, negative_pooled_embedding = self.clip_text_encoder(negative_text or "")

        return torch.cat(tensors=(negative_embedding, conditional_embedding), dim=0), torch.cat(
            tensors=(negative_pooled_embedding, conditional_pooled_embedding), dim=0
        )

    @property
    def default_time_ids(self) -> Tensor:
        """The default time IDs to use."""
        # [original_height, original_width, crop_top, crop_left, target_height, target_width]
        # See https://arxiv.org/abs/2307.01952 > 2.2 Micro-Conditioning
        time_ids = torch.tensor(data=[1024, 1024, 0, 0, 1024, 1024], device=self.device)
        return time_ids.repeat(2, 1)

    def set_unet_context(
        self,
        *,
        timestep: Tensor,
        clip_text_embedding: Tensor,
        pooled_text_embedding: Tensor,
        time_ids: Tensor,
        **_: Tensor,
    ) -> None:
        """Set the various context parameters required by the U-Net model.

        Args:
            timestep: The timestep to set.
            clip_text_embedding: The CLIP text embedding to set.
            pooled_text_embedding: The pooled CLIP text embedding to set.
            time_ids: The time IDs to set.
        """
        self.unet.set_timestep(timestep=timestep)
        self.unet.set_clip_text_embedding(clip_text_embedding=clip_text_embedding)
        self.unet.set_pooled_text_embedding(pooled_text_embedding=pooled_text_embedding)
        self.unet.set_time_ids(time_ids=time_ids)

    def forward(
        self,
        x: Tensor,
        step: int,
        *,
        clip_text_embedding: Tensor,
        pooled_text_embedding: Tensor,
        time_ids: Tensor,
        condition_scale: float = 5.0,
        **kwargs: Tensor,
    ) -> Tensor:
        return super().forward(
            x=x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            time_ids=time_ids,
            condition_scale=condition_scale,
            **kwargs,
        )

    def set_self_attention_guidance(self, enable: bool, scale: float = 1.0) -> None:
        """Sets the self-attention guidance.

        See [[arXiv:2210.00939] Improving Sample Quality of Diffusion Models Using Self-Attention Guidance](https://arxiv.org/abs/2210.00939)
        for more details.

        Args:
            enable: Whether to enable self-attention guidance or not.
            scale: The scale to use.
        """
        if enable:
            if sag := self._find_sag_adapter():
                sag.scale = scale
            else:
                SDXLSAGAdapter(target=self.unet, scale=scale).inject()
        else:
            if sag := self._find_sag_adapter():
                sag.eject()

    def has_self_attention_guidance(self) -> bool:
        """Whether the model has self-attention guidance or not."""
        return self._find_sag_adapter() is not None

    def _find_sag_adapter(self) -> SDXLSAGAdapter | None:
        """Finds the self-attention guidance adapter, if any."""
        for p in self.unet.get_parents():
            if isinstance(p, SDXLSAGAdapter):
                return p
        return None

    def compute_self_attention_guidance(
        self,
        x: Tensor,
        noise: Tensor,
        step: int,
        *,
        clip_text_embedding: Tensor,
        pooled_text_embedding: Tensor,
        time_ids: Tensor,
        **kwargs: Tensor,
    ) -> Tensor:
        """Compute the self-attention guidance.

        Args:
            x: The input tensor.
            noise: The noise tensor.
            step: The step to compute the self-attention guidance at.
            clip_text_embedding: The CLIP text embedding to compute the self-attention guidance with.
            pooled_text_embedding: The pooled CLIP text embedding to compute the self-attention guidance with.
            time_ids: The time IDs to compute the self-attention guidance with.

        Returns:
            The computed self-attention guidance.
        """
        sag = self._find_sag_adapter()
        assert sag is not None

        degraded_latents = sag.compute_degraded_latents(
            solver=self.solver,
            latents=x,
            noise=noise,
            step=step,
            classifier_free_guidance=True,
        )

        negative_text_embedding, _ = clip_text_embedding.chunk(2)
        negative_pooled_embedding, _ = pooled_text_embedding.chunk(2)
        timestep = self.solver.timesteps[step].unsqueeze(dim=0)
        time_ids, _ = time_ids.chunk(2)

        self.set_unet_context(
            timestep=timestep,
            clip_text_embedding=negative_text_embedding,
            pooled_text_embedding=negative_pooled_embedding,
            time_ids=time_ids,
        )
        if "ip_adapter" in self.unet.provider.contexts:
            # this implementation is a bit hacky, it should be refactored in the future
            ip_adapter_context = self.unet.use_context("ip_adapter")
            image_embedding_copy = ip_adapter_context["clip_image_embedding"].clone()
            ip_adapter_context["clip_image_embedding"], _ = ip_adapter_context["clip_image_embedding"].chunk(2)
            degraded_noise = self.unet(degraded_latents)
            ip_adapter_context["clip_image_embedding"] = image_embedding_copy
        else:
            degraded_noise = self.unet(degraded_latents)

        return sag.scale * (noise - degraded_noise)
