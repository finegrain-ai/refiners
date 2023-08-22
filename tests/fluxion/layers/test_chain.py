import torch
import refiners.fluxion.layers as fl


def test_chain_remove_replace():
    chain = fl.Chain(
        fl.Linear(1, 1),
        fl.Linear(1, 1),
        fl.Chain(
            fl.Linear(1, 1),
            fl.Linear(1, 1),
            fl.Chain(fl.Linear(1, 1), fl.Linear(1, 1)),
        ),
        fl.Conv2d(1, 1, 1),
    )
    assert len(chain) == 4
    assert len(chain.Chain) == 3

    chain.remove(chain[-1])
    assert len(chain) == 3
    assert len(chain.Chain) == 3

    assert isinstance(chain.Chain.Chain[1], fl.Linear)
    chain.Chain.Chain.replace(chain.Chain.Chain[1], fl.Conv2d(1, 1, 1))
    assert len(chain) == 3
    assert len(chain.Chain) == 3
    assert isinstance(chain.Chain.Chain[1], fl.Conv2d)


def test_chain_structural_copy():
    m = fl.Chain(
        fl.Sum(
            fl.Linear(in_features=4, out_features=8),
            fl.Linear(in_features=4, out_features=8),
        ),
        fl.Linear(in_features=8, out_features=12),
    )

    x = torch.randn(7, 4)
    y = m(x)
    assert y.shape == (7, 12)

    m2 = m.structural_copy()

    assert m.Linear == m2.Linear
    assert m.Sum.Linear_1 == m2.Sum.Linear_1
    assert m.Sum.Linear_2 == m2.Sum.Linear_2

    assert m.Sum != m2.Sum
    assert m != m2

    assert m.Sum.parent == m
    assert m2.Sum.parent == m2

    y2 = m2(x)
    assert y2.shape == (7, 12)
    torch.equal(y2, y)


def test_chain_find():
    chain = fl.Chain(
        fl.Linear(1, 1),
    )

    assert isinstance(chain.find(fl.Linear), fl.Linear)
    assert chain.find(fl.Conv2d) is None


def test_chain_slice() -> None:
    chain = fl.Chain(
        fl.Linear(in_features=1, out_features=1),
        fl.Linear(in_features=1, out_features=1),
        fl.Linear(in_features=1, out_features=1),
        fl.Chain(
            fl.Linear(in_features=1, out_features=1),
            fl.Linear(in_features=1, out_features=1),
        ),
        fl.Linear(in_features=1, out_features=1),
    )

    x = torch.randn(1, 1)
    sliced_chain = chain[1:4]

    assert len(chain) == 5
    assert len(sliced_chain) == 3
    assert chain[:-1](x).shape == (1, 1)


def test_chain_layers() -> None:
    chain = fl.Chain(
        fl.Chain(fl.Chain(fl.Chain())),
        fl.Chain(),
        fl.Linear(in_features=1, out_features=1),
    )

    assert len(list(chain.layers(fl.Chain))) == 2
    assert len(list(chain.layers(fl.Chain, recurse=True))) == 4
