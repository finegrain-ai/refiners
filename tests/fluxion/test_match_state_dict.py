from refiners.fluxion.match_state_dict import match_state_dict


def test_match_state_dict_simple() -> None:
    source = {"DropoutAdapter.Linear.weight": "foo", "DropoutAdapter.Linear.bias": "bar"}
    target = {"Linear.weight": "init", "Linear.bias": "init"}

    out = match_state_dict(source, target)
    assert out == {"Linear.weight": "foo", "Linear.bias": "bar"}


def test_match_state_dict_simple_reverse() -> None:
    source = {"Linear.weight": "foo", "Linear.bias": "bar"}
    target = {"DropoutAdapter.Linear.weight": "init", "DropoutAdapter.Linear.bias": "init"}

    out = match_state_dict(source, target)
    assert out == {"DropoutAdapter.Linear.weight": "foo", "DropoutAdapter.Linear.bias": "bar"}


def test_match_state_dict_asym_dropout() -> None:
    source = {"DropoutAdapter.Linear.weight": "foo", "Linear.weight": "bar"}
    target = {"Linear_1.weight": "init", "Linear_2.weight": "init"}

    out = match_state_dict(source, target)
    assert out == {"Linear_1.weight": "foo", "Linear_2.weight": "bar"}


def test_match_state_dict_asym_dropout_reverse() -> None:
    source = {"Linear_1.weight": "foo", "Linear_2.weight": "bar"}
    target = {"DropoutAdapter.Linear.weight": "init", "Linear.weight": "init"}

    out = match_state_dict(source, target)
    assert out == {"DropoutAdapter.Linear.weight": "foo", "Linear.weight": "bar"}


def test_match_state_dict_chain() -> None:
    source = {
        "Chain.Chain_1.Linear": "foo1",
        "Chain.Chain_1.Conv": "bar1",
        "Chain.Chain_2.Linear": "foo2",
        "Chain.Chain_2.Conv": "bar2",
    }

    target = {"Chain.Linear_1": "init", "Chain.Conv_1": "init", "Chain.Linear_2": "init", "Chain.Conv_2": "init"}

    out = match_state_dict(source, target)
    assert out == {"Chain.Linear_1": "foo1", "Chain.Conv_1": "bar1", "Chain.Linear_2": "foo2", "Chain.Conv_2": "bar2"}


def test_match_state_dict_chain_reverse() -> None:
    source = {"Chain.Linear_1": "foo1", "Chain.Conv_1": "bar1", "Chain.Linear_2": "foo2", "Chain.Conv_2": "bar2"}

    target = {
        "Chain.Chain_1.Linear": "init",
        "Chain.Chain_1.Conv": "init",
        "Chain.Chain_2.Linear": "init",
        "Chain.Chain_2.Conv": "init",
    }

    out = match_state_dict(source, target)
    assert out == {
        "Chain.Chain_1.Linear": "foo1",
        "Chain.Chain_1.Conv": "bar1",
        "Chain.Chain_2.Linear": "foo2",
        "Chain.Chain_2.Conv": "bar2",
    }
