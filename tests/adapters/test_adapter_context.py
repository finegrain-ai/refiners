import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.adapter import Adapter
from refiners.fluxion.context import Contexts


class ContextAdapter(fl.Chain, Adapter[fl.Chain]):
    def __init__(self, target: fl.Chain):
        with self.setup_adapter(target):
            super().__init__(
                fl.Lambda(lambda: 42),
                fl.SetContext("foo", "bar"),
            )


class ContextChain(fl.Chain):
    def init_context(self) -> Contexts:
        return {"foo": {"bar": None}}


def test_adapter_can_access_parent_context():
    chain = ContextChain(fl.Chain(), fl.UseContext("foo", "bar"))
    adaptee = chain.layer("Chain", fl.Chain)
    ContextAdapter(adaptee).inject(chain)

    assert chain() == 42
