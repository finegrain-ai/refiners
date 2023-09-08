from torch import Tensor

from refiners.foundationals.latent_diffusion.image_prompt import IPAdapter
from refiners.foundationals.latent_diffusion.stable_diffusion_1 import SD1UNet

from refiners.foundationals.clip.image_encoder import CLIPImageEncoderH


class SD1IPAdapter(IPAdapter[SD1UNet]):
    def __init__(
        self,
        target: SD1UNet,
        clip_image_encoder: CLIPImageEncoderH | None = None,
        scale: float = 1.0,
        weights: dict[str, Tensor] | None = None,
    ) -> None:
        super().__init__(
            target=target,
            clip_image_encoder=clip_image_encoder or CLIPImageEncoderH(device=target.device, dtype=target.dtype),
            scale=scale,
            weights=weights,
        )
