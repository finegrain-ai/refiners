import refiners.fluxion.layers as fl


def test_module_get_path() -> None:
    chain = fl.Chain(
        fl.Sum(
            fl.Linear(1, 1),
            fl.Linear(1, 1),
        ),
        fl.Sum(),
    )

    sum_1 = chain.layer("Sum_1", fl.Sum)
    linear_2 = sum_1.layer("Linear_2", fl.Linear)

    assert linear_2.get_path(parent=sum_1) == "Chain.Sum_1.Linear_2"
    assert linear_2.get_path(parent=sum_1, top=sum_1) == "Sum.Linear_2"
    assert sum_1.get_path() == "Chain.Sum_1"


def test_module_basic_attributes() -> None:
    class MyModule(fl.Module):
        def __init__(self, spam: int = 0, foo: list[str | int] = ["bar", "qux", 42]) -> None:
            self.spam = spam
            self.foo = foo
            self.chunky = "bacon"

    m = MyModule(spam=3995)
    assert str(m) == "MyModule(spam=3995)"
    assert m.basic_attributes() == {"chunky": "bacon", "foo": ["bar", "qux", 42], "spam": 3995}
