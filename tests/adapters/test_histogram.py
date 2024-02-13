import torch

from refiners.fluxion.adapters.histogram import HistogramDistance, HistogramEncoder, HistogramExtractor, ColorLoss, histogram_to_histo_channels, sorted_channels_to_histo_channels, tensor_to_sorted_channels
from refiners.fluxion.utils import image_to_tensor, tensor_to_image
from PIL import Image
import numpy as np

def test_histogram_extractor() -> None:
    color_bits = 3
    color_size = 2**color_bits
    img = torch.randint(0, color_size, (1, 3, 224, 224), dtype=torch.uint8).float() / color_size

    extractor = HistogramExtractor(color_bits=color_bits)

    histogram = extractor(img)
    assert histogram.shape == (1, color_size, color_size, color_size)
    assert abs(histogram.sum().item() - 1.0) < 1e-4, "histogram sum should equal 1.0"

    img_black = torch.zeros((1, 3, 224, 224), dtype=torch.uint8).float()
    histogram_black = extractor(img_black)
    assert abs(histogram_black[0, 0, 0, 0] - 1.0) < 1e-4, "histogram_zero should be 1.0 at 0,0,0,0"
    assert abs(histogram_black.sum() - 1.0) < 1e-4, "histogram sum should equal 1.0"

    img_white_normalized = torch.ones((1, 3, 224, 224)) 
    img_white = img_white_normalized
    histogram_white = extractor(img_white)
    assert abs(histogram_white[0, -1, -1, -1] - 1.0) < 1e-4, "histogram_white should be 1.0 at -1,-1,-1,-1"
    assert abs(histogram_white.sum() - 1.0) < 1e-4, "histogram sum should equal 1.0"

    imarray = np.random.rand(256,256,3) * 255
    image = Image.fromarray(imarray.astype('uint8')).convert('RGB')
    histogram_img = extractor.images_to_histograms([image, image])
    assert abs(histogram_img.sum() - 2.0) < 1e-5, "histogram sum should equal 1.0"

def test_images_histogram_extractor() -> None:
    color_bits = 3

    extractor = HistogramExtractor(color_bits=color_bits)

    img_white = tensor_to_image(torch.ones((1, 3, 224, 224)))
    
    histogram_white = extractor.images_to_histograms([img_white])
    assert abs(histogram_white[0, -1, -1, -1] - 1.0) < 1e-4, "histogram_white should be 1.0 at -1,-1,-1,-1"
    assert abs(histogram_white.sum() - 1.0) < 1e-4, "histogram sum should equal 1.0"

    img_black = tensor_to_image(torch.ones((1, 3, 224, 224))*-1)
    histogram_black = extractor.images_to_histograms([img_black])
    
    assert abs(histogram_black[0, 0, 0, 0] - 1.0) < 1e-4, "histogram_zero should be 1.0 at 0,0,0,0"
    assert abs(histogram_black.sum() - 1.0) < 1e-4, "histogram sum should equal 1.0"

def test_histogram_distance() -> None:
    distance = HistogramDistance()
    color_bits = 2
    color_size = 2**color_bits
    batch_size = 10

    histo1 = torch.rand((batch_size, color_size, color_size, color_size))
    sum1 = histo1.sum()
    histo1 = histo1 / sum1

    histo2 = torch.rand((batch_size, color_size, color_size, color_size))
    sum2 = histo2.sum()
    histo2 = histo2 / sum2

    dist_same = distance(histo1, histo1)
    assert abs(dist_same) < 1e-4, "distance between himself should be 0.0"
    
    dist_diff = distance(histo1, histo2)
    assert dist_diff >= 0.0, "distance should more than 0.0"
    
    dist_bhattacharyya = distance.bhattacharyya(histo1, histo2)
    assert dist_bhattacharyya >= 0.0, "distance bhattacharyya should be more than 0"

    dist_kl_div = distance.kl_div(histo1, histo2)
    assert dist_kl_div >= 0.0, "distance kl div himself should more than 0.0"

    dist_intersection_same = distance.intersection(histo1, histo1)
    assert abs(dist_intersection_same - 1.0) < 1e-6, "distance intersection should be 1"
    
    dist_correlation_same = distance.correlation(histo1, histo1)
    assert dist_correlation_same > 0.0, "distance correlation should be more than 0"


def test_histogram_encoder() -> None:
    batch_size = 2
    patch_size = 16
    color_bits = 6
    cube_size = 2**color_bits
    histo1 = torch.rand((batch_size, cube_size, cube_size, cube_size))
    sum1 = histo1.sum()
    histo1 = histo1 / sum1

    embedding_dim = 768
    n_patch = cube_size // patch_size
    
    encoder = HistogramEncoder(color_bits=color_bits, patch_size=patch_size, embedding_dim=embedding_dim)
    embedding = encoder(histo1)
    assert embedding.shape == (batch_size, n_patch**3 + 1, embedding_dim), "embedding shape should be (batch_size, ouput_dim)"

def test_color_loss() -> None:
    
    img_white_normalized = torch.ones((2, 3, 224, 224)) 
    img_black_normalized = torch.zeros((2, 3, 224, 224)) 

    color_loss = ColorLoss()

    assert color_loss(img_white_normalized, img_black_normalized) == 1.0, "White and black image should give loss = 1"

def test_sorted_channels() -> None:
    img_white_normalized = torch.ones((2, 3, 224, 224)) 
    img_black_normalized = torch.zeros((2, 3, 224, 224)) 
    
    color_bits = 5
    extractor = HistogramExtractor(color_bits=color_bits)
    
    sorted_channels = tensor_to_sorted_channels(img_white_normalized)
    histo_channels1 = sorted_channels_to_histo_channels(sorted_channels, color_bits=color_bits)
    
    histogram = extractor(img_white_normalized)
    histo_channels2 = histogram_to_histo_channels(histogram)
    
    assert len(histo_channels1) == len(histo_channels2), "histo_channels1 and histo_channels2 should have the same length"
    assert len(histo_channels1) == 3, "histo_channels1 and histo_channels2 should have length 3"
    
    for i in range(len(histo_channels1)):
        assert histo_channels1[i].shape == histo_channels2[i].shape, "histo_channels1 and histo_channels2 should have the same shape"
        assert histo_channels1[i].shape == (2, 2**color_bits), "histo_channels1 and histo_channels2 should have shape (2, 2**color_bits)"
        assert torch.allclose(histo_channels1[i], histo_channels2[i]), "histo_channels1 and histo_channels2 should be close"
