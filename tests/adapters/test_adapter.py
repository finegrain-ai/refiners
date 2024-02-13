import pytest

from refiners.fluxion.adapters.adapter import Adapter
from refiners.fluxion.layers import Chain, Linear


class DummyLinearAdapter(Chain, Adapter[Linear]):
    def __init__(self, target: Linear):
        with self.setup_adapter(target):
            super().__init__(target)


class DummyChainAdapter(Chain, Adapter[Chain]):
    def __init__(self, target: Chain):
        with self.setup_adapter(target):
            super().__init__(target)


@pytest.fixture
def chain() -> Chain:
    return Chain(Chain(Linear(2, 2)))


def test_weighted_module_adapter_insertion(chain: Chain):
    parent = chain.layer("Chain", Chain)
    adaptee = parent.layer("Linear", Linear)

    adapter = DummyLinearAdapter(adaptee).inject(parent)

    assert adapter.parent == parent
    assert adapter in iter(parent)
    assert adaptee not in iter(parent)

    adapter.eject()
    assert adapter.parent is None
    assert adapter not in iter(parent)
    assert adaptee in iter(parent)


def test_chain_adapter_insertion(chain: Chain):
    parent = chain
    adaptee = parent.layer("Chain", Chain)

    adapter = DummyChainAdapter(adaptee)
    assert adaptee.parent == parent

    adapter.inject()
    assert adapter.parent == parent
    assert adaptee.parent == adapter
    assert adapter in iter(parent)
    assert adaptee not in iter(parent)

    adapter.eject()
    assert adapter.parent is None
    assert adaptee.parent == parent
    assert adapter not in iter(parent)
    assert adaptee in iter(parent)


def test_weighted_module_adapter_structural_copy(chain: Chain):
    parent = chain.layer("Chain", Chain)
    adaptee = parent.layer("Linear", Linear)

    DummyLinearAdapter(adaptee).inject(parent)

    clone = chain.structural_copy()
    cloned_adapter = clone.layer(("Chain", "DummyLinearAdapter"), DummyLinearAdapter)
    assert cloned_adapter.parent == clone.Chain
    assert cloned_adapter.target == adaptee


def test_chain_adapter_structural_copy(chain: Chain):
    # Chain adapters cannot be copied by default.
    adapter = DummyChainAdapter(chain.layer("Chain", Chain)).inject()

    with pytest.raises(RuntimeError):
        chain.structural_copy()

    adapter.eject()
    chain.structural_copy()
