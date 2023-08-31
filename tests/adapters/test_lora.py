from refiners.adapters.lora import Lora, SingleLoraAdapter, LoraAdapter
from torch import randn, allclose
import refiners.fluxion.layers as fl


def test_single_lora_adapter() -> None:
    chain = fl.Chain(
        fl.Chain(
            fl.Linear(in_features=1, out_features=1),
            fl.Linear(in_features=1, out_features=1),
        ),
        fl.Linear(in_features=1, out_features=2),
    )
    x = randn(1, 1)
    y = chain(x)

    lora_adapter = SingleLoraAdapter(chain.Chain.Linear_1).inject(chain.Chain)

    assert isinstance(lora_adapter[1], Lora)
    assert allclose(input=chain(x), other=y)
    assert lora_adapter.parent == chain.Chain

    lora_adapter.eject()
    assert isinstance(chain.Chain[0], fl.Linear)
    assert len(chain) == 2

    lora_adapter.inject(chain.Chain)
    assert isinstance(chain.Chain[0], SingleLoraAdapter)


def test_lora_adapter() -> None:
    chain = fl.Chain(
        fl.Chain(
            fl.Linear(in_features=1, out_features=1),
            fl.Linear(in_features=1, out_features=1),
        ),
        fl.Linear(in_features=1, out_features=2),
    )

    LoraAdapter[fl.Chain](chain, sub_targets=chain.walk(fl.Linear), rank=1, scale=1.0).inject()

    assert len(list(chain.layers(Lora))) == 3
