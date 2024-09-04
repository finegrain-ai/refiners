from torch import Tensor

from refiners.foundationals.latent_diffusion.ella_adapter import ELLA, ELLAAdapter
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet


class SD1ELLAAdapter(ELLAAdapter[SD1UNet]):
    """[`ELLA`][refiners.foundationals.latent_diffusion.ella_adapter.ELLA] adapter for Stable Diffusion 1.5."""

    def __init__(self, target: SD1UNet, weights: dict[str, Tensor] | None = None) -> None:
        """Initialize the adapter.

        Args:
            target: The target model to adapt.
            weights: The weights of the ELLA adapter (see `scripts/conversion/convert_ella_adapter.py`).
        """
        latents_encoder = ELLA(
            time_channel=320,
            timestep_embedding_dim=768,
            width=768,
            num_layers=6,
            num_heads=8,
            num_latents=64,
            input_dim=2048,
            device=target.device,
            dtype=target.dtype,
        )
        super().__init__(target=target, latents_encoder=latents_encoder, weights=weights)
