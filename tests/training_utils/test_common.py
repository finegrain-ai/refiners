import random

import pytest
import torch

from refiners.training_utils.common import (
    Epoch,
    Iteration,
    Step,
    TimeValue,
    TimeValueInput,
    parse_number_unit_field,
    scoped_seed,
)


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        ("3 : steP", Step(3)),
        ("5: epoch", Epoch(5)),
        (" 7:Iteration", Iteration(7)),
    ],
)
def test_time_value_from_str(input_value: str, expected_output: TimeValue) -> None:
    result = TimeValue.from_str(input_value)
    assert result == expected_output


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        ("10: step", Step(10)),
        ("20 :epoch", Epoch(20)),
        ("30: Iteration", Iteration(30)),
        (50, Step(50)),
        (Epoch(200), Epoch(200)),
    ],
)
def test_parse_number_unit_field(input_value: TimeValueInput, expected_output: TimeValue) -> None:
    result = parse_number_unit_field(input_value)
    assert result == expected_output


@pytest.mark.parametrize(
    "invalid_input",
    [
        "invalid:input",
        "10: invalid",
        "10",
        None,
    ],
)
def test_parse_number_unit_field_invalid_input(invalid_input: TimeValueInput):
    with pytest.raises(ValueError):
        parse_number_unit_field(invalid_input)


@scoped_seed(seed=37)
def pick_a_number() -> int:
    return int(torch.randint(0, 100, (1,)).item())


@pytest.mark.parametrize(
    "seed, expected_output",
    [
        (42, 42),
        (37, 31),
        (0, 44),
    ],
)
def test_scoped_seed_with_specific_seed(seed: int, expected_output: int) -> None:
    with scoped_seed(seed):
        assert torch.randint(0, 100, (1,)).item() == expected_output


@pytest.mark.parametrize(
    "seed, expected_output",
    [
        (42, 81),
        (37, 87),
        (0, 49),
    ],
)
def test_scoped_seed_with_random_module(seed: int, expected_output: int) -> None:
    with scoped_seed(seed):
        assert random.randint(0, 100) == expected_output


def test_scoped_seed_with_function_call() -> None:
    assert pick_a_number() == 31

    with scoped_seed(37):
        assert pick_a_number() == 31


def test_scoped_seed_with_callable_seed() -> None:
    with scoped_seed(pick_a_number):
        assert pick_a_number() == 31

    def add_10(n: int) -> int:
        return n + 10

    @scoped_seed(seed=add_10)
    def pick_a_number_greater_than_n_plus_10(n: int) -> int:
        return int(torch.randint(n, 100, (1,)).item())

    assert pick_a_number_greater_than_n_plus_10(10) == 81


def test_scoped_seed_restore_state() -> None:
    random.seed(37)
    with scoped_seed(42):
        random.randint(0, 100)
    assert random.randint(0, 100) == 87


def test_import_training_utils() -> None:
    try:
        import refiners.training_utils
    except ImportError:
        pytest.fail("Failed to import refiners.training_utils")

    assert refiners.training_utils is not None
