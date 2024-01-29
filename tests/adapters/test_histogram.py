import torch

from refiners.fluxion.adapters.histogram import HistogramDistance, HistogramEncoder, HistogramExtractor, ColorLoss
from refiners.fluxion.utils import image_to_tensor, tensor_to_image


def test_histogram_extractor() -> None:
    color_bits = 3
    color_size = 2**color_bits
    img = torch.randint(0, color_size, (1, 3, 224, 224), dtype=torch.uint8).float()

    extractor = HistogramExtractor(color_bits=color_bits)

    histogram = extractor(img)
    assert histogram.shape == (1, color_size, color_size, color_size)
    assert abs(histogram.sum().item() - 1.0) < 1e-4, "histogram sum should equal 1.0"

    img_black = torch.zeros((1, 3, 224, 224), dtype=torch.uint8).float()
    histogram_black = extractor(img_black)
    assert abs(histogram_black[0, 0, 0, 0] - 1.0) < 1e-4, "histogram_zero should be 1.0 at 0,0,0,0"
    assert abs(histogram_black.sum() - 1.0) < 1e-4, "histogram sum should equal 1.0"

    img_white_normalized = torch.ones((1, 3, 224, 224)) 
    img_white = img_white_normalized * (color_size - 1)
    histogram_white = extractor(img_white)
    assert abs(histogram_white[0, -1, -1, -1] - 1.0) < 1e-4, "histogram_white should be 1.0 at -1,-1,-1,-1"
    assert abs(histogram_white.sum() - 1.0) < 1e-4, "histogram sum should equal 1.0"
    
    decoded_histogram_white = img_white_normalized * 2 - 1
    histogram_white2 = extractor.from_decoded(decoded_histogram_white)
    distance = HistogramDistance()    
    assert distance(histogram_white2, histogram_white) == 0.0, "distance between himself should be 0.0"

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
    batch_size = 2

    histo1 = torch.rand((batch_size, color_size, color_size, color_size))
    sum1 = histo1.sum()
    histo1 = histo1 / sum1

    histo2 = torch.rand((batch_size, color_size, color_size, color_size))
    sum2 = histo2.sum()
    histo2 = histo2 / sum2

    dist_same = distance(histo1, histo1)
    assert dist_same == 0.0, "distance between himself should be 0.0"


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
