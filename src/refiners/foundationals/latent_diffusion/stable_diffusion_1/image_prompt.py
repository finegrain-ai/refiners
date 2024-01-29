from torch import Tensor

from refiners.foundationals.clip.image_encoder import CLIPImageEncoderH
from refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock2d
from refiners.foundationals.latent_diffusion.image_prompt import ImageProjection, IPAdapter, PerceiverResampler
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
from refiners.foundationals.dinov2 import ViT

def get_sd1_image_proj(image_encoder: CLIPImageEncoderH | ViT, target: SD1UNet, cross_attn_2d: CrossAttentionBlock2d, fine_grained: bool) -> ImageProjection | PerceiverResampler:
    return (
        ImageProjection(
            image_embedding_dim=image_encoder.output_dim,
            clip_text_embedding_dim=cross_attn_2d.context_embedding_dim,
            device=target.device,
            dtype=target.dtype,
        )
        if not fine_grained
        else PerceiverResampler(
            latents_dim=cross_attn_2d.context_embedding_dim,
            num_attention_layers=4,
            num_attention_heads=12,
            head_dim=64,
            num_tokens=16,
            input_dim=image_encoder.embedding_dim,  # = dim before final projection
            output_dim=cross_attn_2d.context_embedding_dim,
            device=target.device,
            dtype=target.dtype,
        )
    )
class SD1IPAdapter(IPAdapter[SD1UNet]):
    def __init__(
        self,
        target: SD1UNet,
        image_encoder: CLIPImageEncoderH | ViT | None = None,
        image_proj: ImageProjection | PerceiverResampler | None = None,
        scale: float = 1.0,
        fine_grained: bool = False,
        weights: dict[str, Tensor] | None = None,
        strict: bool = True,
        use_timestep_embedding: bool = False,
        use_pooled_text_embedding: bool = False,
        train_image_proj: bool = False
    ) -> None:
        image_encoder = image_encoder or CLIPImageEncoderH(device=target.device, dtype=target.dtype)

        if image_proj is None:
            cross_attn_2d = target.ensure_find(CrossAttentionBlock2d)
            image_proj = get_sd1_image_proj(image_encoder, target, cross_attn_2d, fine_grained)
        elif fine_grained:
            assert isinstance(image_proj, PerceiverResampler)

        super().__init__(
            target=target,
            image_encoder=image_encoder,
            image_proj=image_proj,
            scale=scale,
            fine_grained=fine_grained,
            weights=weights,
            strict=strict,
            use_timestep_embedding=use_timestep_embedding,
            use_pooled_text_embedding=use_pooled_text_embedding,
            train_image_proj=train_image_proj
        )
