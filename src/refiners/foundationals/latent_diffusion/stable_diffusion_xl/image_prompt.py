from torch import Tensor

from refiners.foundationals.clip.image_encoder import CLIPImageEncoderH
from refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock2d
from refiners.foundationals.latent_diffusion.image_prompt import ImageProjection, IPAdapter, PerceiverResampler
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet


class SDXLIPAdapter(IPAdapter[SDXLUNet]):
    """Image Prompt adapter for the Stable Diffusion XL U-Net model."""

    def __init__(
        self,
        target: SDXLUNet,
        clip_image_encoder: CLIPImageEncoderH | None = None,
        image_proj: ImageProjection | PerceiverResampler | None = None,
        scale: float = 1.0,
        fine_grained: bool = False,
        weights: dict[str, Tensor] | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            target: The SDXLUNet model to adapt.
            clip_image_encoder: The CLIP image encoder to use.
            image_proj: The image projection to use.
            scale: The scale to use for the image prompt.
            fine_grained: Whether to use fine-grained image prompt.
            weights: The weights of the IPAdapter.
        """
        clip_image_encoder = clip_image_encoder or CLIPImageEncoderH(device=target.device, dtype=target.dtype)

        if image_proj is None:
            cross_attn_2d = target.ensure_find(CrossAttentionBlock2d)
            image_proj = (
                ImageProjection(
                    clip_image_embedding_dim=clip_image_encoder.output_dim,
                    clip_text_embedding_dim=cross_attn_2d.context_embedding_dim,
                    device=target.device,
                    dtype=target.dtype,
                )
                if not fine_grained
                else PerceiverResampler(
                    latents_dim=1280,  # not `cross_attn_2d.context_embedding_dim` in this case
                    num_attention_layers=4,
                    num_attention_heads=20,
                    head_dim=64,
                    num_tokens=16,
                    input_dim=clip_image_encoder.embedding_dim,  # = dim before final projection
                    output_dim=cross_attn_2d.context_embedding_dim,
                    device=target.device,
                    dtype=target.dtype,
                )
            )
        elif fine_grained:
            assert isinstance(image_proj, PerceiverResampler)

        super().__init__(
            target=target,
            clip_image_encoder=clip_image_encoder,
            image_proj=image_proj,
            scale=scale,
            fine_grained=fine_grained,
            weights=weights,
        )
