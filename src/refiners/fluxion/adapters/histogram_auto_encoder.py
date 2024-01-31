from PIL import Image
from torch import Tensor, device as Device, dtype as DType
from refiners.fluxion.layers.basics import Unsqueeze, Squeeze
from refiners.foundationals.latent_diffusion.auto_encoder import Encoder, Decoder
from refiners.fluxion.layers import Chain


class HistogramAutoEncoder(Chain):
    encoder_scale = 1

    def __init__(
        self,
        latent_dim: int = 8,
        resnet_sizes: list[int] = [8, 4],
        num_groups: int = 4,
        n_down_samples: int = 2,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        
        histogram_dim = 1
        spatial_dims = 3
        self.n_down_samples = n_down_samples
        self.latent_dim = latent_dim
        
        super().__init__(
            Chain(
                Unsqueeze(dim=1),
                Encoder(
                    spatial_dims = spatial_dims, 
                    num_groups = num_groups, 
                    resnet_sizes = resnet_sizes,
                    input_channels = histogram_dim,
                    latent_dim = latent_dim,
                    device=device, 
                    n_down_samples=n_down_samples,
                     slide_end = latent_dim,
                    dtype=dtype
                )
            ),
            Chain(
                Decoder(
                    spatial_dims = spatial_dims, 
                    num_groups = num_groups, 
                    resnet_sizes = resnet_sizes,
                    output_channels = histogram_dim,
                     n_up_samples=n_down_samples,
                    latent_dim = latent_dim,
                    device=device, 
                    dtype=dtype
                ),
                Squeeze(dim=1)    
            )
        )
    
    def encode(self, x: Tensor) -> Tensor:
        encoder = self[0]
        x = self.encoder_scale * encoder(x)
        return x

    def decode(self, x: Tensor) -> Tensor:
        decoder = self[1]
        x = decoder(x / self.encoder_scale)
        return x

    def image_to_latent(self, image: Image.Image) -> Tensor:
        return self.images_to_latents([image])

    def images_to_latents(self, images: list[Image.Image]) -> Tensor:
        histograms = self.histogram_extractor.images_to_histograms(images)
        return self.encode(histograms)
    
    @property
    def compression_rate(self) -> float:
        return (2**self.n_down_samples)**3 / self.latent_dim
