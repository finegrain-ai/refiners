from refiners.foundationals.latent_diffusion.scale_crafter import SDScaleCrafterAdapter
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import  SD1UNet
from torch import Tensor

class SD1ScaleCrafterAdapter(SDScaleCrafterAdapter[SD1UNet]):
    def __init__(self, target: SD1UNet, dilation_settings: dict[str, float], inflate_settings: list[str], noise_damped_dilation_settings: dict[str, float], noise_damped_inflate_settings: dict[str, str], inflate_transform: Tensor | None = None, inflate_timestep: int = 0, dilation_timestep: int = 700, noise_damped_timestep: int = 700, progressive: bool =False) -> None:
        super().__init__(
            target, dilation_settings, inflate_settings, noise_damped_dilation_settings, noise_damped_inflate_settings, inflate_transform, inflate_timestep, dilation_timestep, noise_damped_timestep, progressive
        )