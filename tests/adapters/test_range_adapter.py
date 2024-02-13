import torch

from refiners.fluxion.adapters.adapter import Adapter
from refiners.fluxion.layers import Chain, Linear
from refiners.foundationals.latent_diffusion.range_adapter import RangeEncoder


class DummyLinearAdapter(Chain, Adapter[Linear]):
    def __init__(self, target: Linear):
        with self.setup_adapter(target):
            super().__init__(target)


def test_range_encoder_dtype_after_adaptation(test_device: torch.device):  # FG-433
    dtype = torch.float64
    chain = Chain(RangeEncoder(320, 1280, device=test_device, dtype=dtype))

    range_encoder = chain.layer("RangeEncoder", RangeEncoder)
    adaptee = range_encoder.layer("Linear_1", Linear)
    adapter = DummyLinearAdapter(adaptee).inject(range_encoder)

    assert adapter.parent == chain.RangeEncoder

    x = torch.tensor([42], device=test_device)
    y = chain(x)
    assert y.dtype == dtype
