import torch

from refiners.fluxion.adapters.histogram import HistogramExtractor, HistogramDistance
from refiners.fluxion.adapters.histogram_auto_encoder import HistogramAutoEncoder


def test_histogram_auto_encoder() -> None:
    color_bits = 5
    color_size = 2**color_bits
    batch_size = 1
    latent_dim = 8
    resnet_sizes = [4, 4, 4, 4, 4]
    n_down_samples = 4
    img = torch.randint(0, color_size, (batch_size, 3, 224, 224), dtype=torch.uint8).float()
    
    out_size = color_size / 2**n_down_samples
    extractor = HistogramExtractor(color_bits=color_bits)

    histogram = extractor(img)
    encoder = HistogramAutoEncoder(
        latent_dim=latent_dim, 
        resnet_sizes=resnet_sizes,
        n_down_samples=n_down_samples
    )
    
    encoded = encoder.encode(histogram)
    decoded = encoder.decode(encoded)
    
    assert encoded.shape == (batch_size, latent_dim, out_size, out_size, out_size), "decoded shape should be the same as histogram shape"
    assert decoded.shape == histogram.shape, "decoded shape should be the same as histogram shape"
    