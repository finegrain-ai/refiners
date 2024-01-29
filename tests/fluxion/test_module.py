import refiners.fluxion.layers as fl


def test_module_get_path() -> None:
    chain = fl.Chain(
        fl.Sum(
            fl.Linear(1, 1),
            fl.Linear(1, 1),
        ),
        fl.Sum(),
    )

    assert chain.Sum_1.Linear_2.get_path(parent=chain.Sum_1) == "Chain.Sum_1.Linear_2"
    assert chain.Sum_1.Linear_2.get_path(parent=chain.Sum_1, top=chain.Sum_1) == "Sum.Linear_2"
    assert chain.Sum_1.get_path() == "Chain.Sum_1"
