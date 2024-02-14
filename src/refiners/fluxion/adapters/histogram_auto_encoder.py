from PIL import Image
from refiners.fluxion.layers.converter import Converter
from refiners.fluxion.utils import summarize_tensor
from torch import Tensor, device as Device, dtype as DType, zeros_like, cat
from torch.nn import Softmax
from refiners.fluxion.layers.basics import Reshape, Unsqueeze, Squeeze
from refiners.foundationals.latent_diffusion.auto_encoder import Encoder, Decoder
from refiners.fluxion.layers import Chain



class HistogramAutoEncoder(Chain):
    encoder_scale = 1

    def __init__(
        self,
        latent_dim: int = 8,
        resnet_sizes: list[int] = [4, 4, 8],
        num_groups: int = 4,
        color_bits: int = 8,
        n_down_samples: int = 2,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        
        histogram_dim = 1
        spatial_dims = 3
        self.n_down_samples = n_down_samples
        self.latent_dim = latent_dim
        self.color_bits = color_bits
        cube_size = 2 ** color_bits
        
        super().__init__(
            Chain(
                Converter(),
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
    
    def encode_sequence(self, x: Tensor) -> Tensor:
        return self.encode(x).reshape(x.shape[0], 1, -1)
    
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
    def compression_rate(self) -> int:
        return (2**self.n_down_samples)**3 / self.latent_dim
    
    @property
    def embedding_dim(self) -> int:
        color_size = 2**self.color_bits
        
        embedding_dim = color_size**3 / self.compression_rate
        return int(embedding_dim)
    
    def unconditionnal_embedding_like(self, x: Tensor) -> Tensor:
        numel: int = x.numel()
        if numel == 0:
            raise ValueError("Cannot compute histogram embedding for empty tensor")        
        return (zeros_like(x) + 1.0) / numel
    
    def compute_histogram_embedding(
        self,
        x: Tensor,
        negative_histogram: None | Tensor = None,
    ) -> Tensor:
        conditional_embedding = self.encode_sequence(x)
        if x == negative_histogram:
            return cat(tensors=(conditional_embedding, conditional_embedding), dim=0)

        if negative_histogram is None:
            # a uniform palette with all the colors at the same frequency
            negative_histogram = self.unconditionnal_embedding_like(x)

        negative_embedding = self.encode_sequence(negative_histogram)
        return cat(tensors=(negative_embedding, conditional_embedding), dim=0)