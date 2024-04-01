import pytest

from refiners.training_utils.common import TimeUnit, TimeValue, TimeValueInput, parse_number_unit_field


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        ("10: step", TimeValue(number=10, unit=TimeUnit.STEP)),
        ("20 :epoch", TimeValue(number=20, unit=TimeUnit.EPOCH)),
        ("30: Iteration", TimeValue(number=30, unit=TimeUnit.ITERATION)),
        (50, TimeValue(number=50, unit=TimeUnit.DEFAULT)),
        ({"number": 100, "unit": "STEP"}, TimeValue(number=100, unit=TimeUnit.STEP)),
        (TimeValue(number=200, unit=TimeUnit.EPOCH), TimeValue(number=200, unit=TimeUnit.EPOCH)),
    ],
)
def test_parse_number_unit_field(input_value: TimeValueInput, expected_output: TimeValue):
    result = parse_number_unit_field(input_value)
    assert result == expected_output


@pytest.mark.parametrize(
    "invalid_input",
    [
        "invalid:input",
        {"number": "not_a_number", "unit": "step"},
        {"invalid_key": 10},
        None,
    ],
)
def test_parse_number_unit_field_invalid_input(invalid_input: TimeValueInput):
    with pytest.raises(ValueError):
        parse_number_unit_field(invalid_input)
