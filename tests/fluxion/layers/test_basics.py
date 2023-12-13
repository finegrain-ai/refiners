import torch

from refiners.fluxion.layers.basics import Slicing


def test_slicing_positive_indices() -> None:
    x = torch.randn(5, 5, 5)
    slicing_layer = Slicing(dim=0, start=2, end=5)
    sliced = slicing_layer(x)
    expected = x[2:5, :]
    assert torch.equal(sliced, expected)


def test_slicing_negative_indices() -> None:
    x = torch.randn(5, 5, 5)
    slicing_layer = Slicing(dim=1, start=-3, end=-1)
    sliced = slicing_layer(x)
    expected = x[:, -3:-1]
    assert torch.equal(sliced, expected)


def test_none_end_slicing() -> None:
    x = torch.randn(2, 1000, 400)
    slicing_layer = Slicing(dim=1, start=1)
    sliced = slicing_layer(x)
    expected = x[:, 1:, :]
    assert torch.equal(sliced, expected)


def test_slicing_step() -> None:
    x = torch.randn(5, 5, 5)
    slicing_layer = Slicing(dim=1, start=0, end=5, step=2)
    sliced = slicing_layer(x)
    expected = x[:, 0:5:2]
    assert torch.equal(sliced, expected)


def test_slicing_empty_slice() -> None:
    x = torch.randn(5, 5, 5)
    slicing_layer = Slicing(dim=1, start=3, end=3)
    sliced = slicing_layer(x)
    expected = x[:, 3:3]
    assert torch.equal(sliced, expected)


def test_slicing_full_dimension() -> None:
    x = torch.randn(5, 5, 5)
    slicing_layer = Slicing(dim=2, start=0, end=5)
    sliced = slicing_layer(x)
    expected = x[:, :, :]
    assert torch.equal(sliced, expected)


def test_slicing_step_greater_than_range() -> None:
    x = torch.randn(5, 5, 5)
    slicing_layer = Slicing(dim=1, start=1, end=3, step=4)
    sliced = slicing_layer(x)
    expected = x[:, 1:3:4]
    assert torch.equal(sliced, expected)


def test_slicing_reversed_start_end() -> None:
    x = torch.randn(5, 5, 5)
    slicing_layer = Slicing(dim=1, start=4, end=2)
    sliced = slicing_layer(x)
    expected = x[:, 4:2]
    assert torch.equal(sliced, expected)


def test_slicing_out_of_bounds_indices() -> None:
    x = torch.randn(5, 5, 5)
    slicing_layer = Slicing(dim=1, start=-10, end=10)
    sliced = slicing_layer(x)
    expected = x[:, -10:10]
    assert torch.equal(sliced, expected)
