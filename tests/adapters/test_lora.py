from refiners.adapters.lora import Lora, LoraAdapter
from torch import randn, allclose
import refiners.fluxion.layers as fl


def test_lora() -> None:
    chain = fl.Chain(
        fl.Chain(
            fl.Linear(in_features=1, out_features=1),
            fl.Linear(in_features=1, out_features=1),
        ),
        fl.Linear(in_features=1, out_features=2),
    )
    x = randn(1, 1)
    y = chain(x)

    lora_adapter = LoraAdapter(chain.Chain.Linear_1)
    lora_adapter.inject(chain.Chain)

    assert isinstance(lora_adapter[1], Lora)
    assert allclose(input=chain(x), other=y)
    assert lora_adapter.parent == chain.Chain

    lora_adapter.eject()
    assert isinstance(chain.Chain[0], fl.Linear)
    assert len(chain) == 2

    lora_adapter.inject(chain.Chain)
    assert isinstance(chain.Chain[0], LoraAdapter)
