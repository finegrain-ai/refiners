from pathlib import Path

import pytest
from torch import Tensor, load as torch_load, randn, tensor  # type: ignore

from refiners.training_utils.batch import BaseBatch


def test_inherit_no_attribute() -> None:
    with pytest.raises(ValueError) as excinfo:

        class NoAttrBatch(BaseBatch):  # type: ignore
            pass

    assert "At least one attribute is required for 'NoAttrBatch'" == str(excinfo.value)


def test_inherit_non_list_errors() -> None:
    with pytest.raises(TypeError) as excinfo:

        class StrBatch(BaseBatch):  # type: ignore
            foo: str

    assert "Type of 'foo' must be 'Tensor' or 'list', got 'str'" == str(excinfo.value)


class MockBatch(BaseBatch):
    foo: Tensor
    bar: Tensor
    indices: list[int]


def test_single_batch() -> None:
    single = MockBatch(foo=randn(1, 10), bar=randn(1, 5), indices=[0])
    assert len(single) == 1


def test_l5_batch() -> None:
    l5 = MockBatch(foo=randn(5, 10), bar=randn(5, 5), indices=[0, 1, 2, 3, 4])
    assert len(l5) == 5
    total = 0
    for single in l5:
        assert len(single) == 1
        total += 1

    assert total == 5


def test_attr_type_error() -> None:
    with pytest.raises(TypeError) as excinfo:
        MockBatch(foo="foo", bar=randn(5, 5), indices=[0, 1, 2, 3, 4])  # type: ignore

    assert "Invalid type for attribute 'foo': Expected 'Tensor', got 'str'" == str(excinfo.value)


def test_extra_missing_attr() -> None:
    with pytest.raises(ValueError) as excinfo:
        MockBatch(foo=randn(5, 10), bar=randn(5, 5))  # type: ignore

    assert "Missing required attribute 'indices'" == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        MockBatch(foo=randn(5, 10), bar=randn(5, 5), indices=[0, 1, 2], extra=["a", "b", "c"])  # type: ignore

    assert "Attribute 'extra' is not valid" == str(excinfo.value)


def test_inhomogeneous_sizes_raise_errors() -> None:
    with pytest.raises(ValueError) as excinfo:
        MockBatch(foo=randn(5, 10), bar=randn(6, 5), indices=[0, 1, 2, 3, 4])

    assert "Attribute 'bar' has size 6, expected 5" == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        MockBatch(foo=randn(5, 10), bar=randn(5, 5), indices=[0, 1, 2])

    assert "Attribute 'indices' has size 3, expected 5" == str(excinfo.value)


def test_empty_attr_raise_errors() -> None:
    with pytest.raises(ValueError) as excinfo:
        MockBatch(foo=randn(0, 10), bar=randn(6, 5), indices=[0, 1, 2, 3, 4])

    assert "Attribute 'foo' is empty, empty attributes are not permitted" == str(excinfo.value)


def test_load_save_batch() -> None:
    loaded = MockBatch(foo=randn(3, 10), bar=randn(3, 5), indices=[1, 2, 3])
    tmp_filename = Path(".pytest_cache/test_batch_tmp.pt")
    loaded.save(tmp_filename)
    second_loaded = MockBatch.load(tmp_filename)
    assert second_loaded == loaded


def test_equality() -> None:
    b1 = MockBatch(foo=randn(3, 10), bar=randn(3, 5), indices=[1, 2, 3])
    b2 = b1.clone()
    assert b2 == b1
    assert not b1 != b2
    b3 = MockBatch(foo=randn(3, 10), bar=randn(3, 5), indices=[1, 2, 3])
    assert b3 != b1


def test_collate() -> None:
    b1 = MockBatch(foo=randn(3, 10), bar=randn(3, 5), indices=[1, 2, 3])
    b2 = MockBatch(foo=randn(5, 10), bar=randn(5, 5), indices=[4, 5, 6, 7, 8])
    b3 = MockBatch(foo=randn(1, 10), bar=randn(1, 5), indices=[9])
    collated = MockBatch.collate([b1, b2, b3])
    assert len(collated) == 9


def test_add() -> None:
    b1 = MockBatch(foo=randn(3, 10), bar=randn(3, 5), indices=[1, 2, 3])
    b2 = MockBatch(foo=randn(5, 10), bar=randn(5, 5), indices=[4, 5, 6, 7, 8])
    collated = MockBatch.collate([b1, b2])
    added = b1 + b2
    assert added == collated


def test_slicing() -> None:
    b1 = MockBatch(foo=randn(3, 10), bar=randn(3, 5), indices=[1, 2, 3])
    b1_0 = b1[0]
    assert len(b1_0) == 1
    b1_21 = b1[1:3]
    assert len(b1_21) == 2
    assert MockBatch.collate([b1_0, b1_21]) == b1

    b1_0_2 = b1[[0, 2]]
    assert len(b1_0_2) == 2
    assert MockBatch.collate([b1_0_2[0], b1[1], b1_0_2[1]]) == b1


def test_repr() -> None:
    b1 = MockBatch(foo=tensor([[1, 2]]), bar=tensor([[5, 6]]), indices=[1])
    assert (
        repr(b1)
        == "MockBatch(size=1)[foo=Tensor(shape=(1, 2), dtype=int64, device=cpu, min=1.00, max=2.00, mean=1.50, std=0.71, norm=2.24, grad=False),bar=Tensor(shape=(1, 2), dtype=int64, device=cpu, min=5.00, max=6.00, mean=5.50, std=0.71, norm=7.81, grad=False),indices=[1]]"
    )


def test_getattr_setattr() -> None:
    tensor1 = tensor([[1, 2]])
    tensor2 = tensor([[5, 6]])
    list1 = [10]
    list2 = [20]
    b1 = MockBatch(foo=tensor1, bar=tensor1, indices=list1)
    assert (b1.foo == tensor1).all()
    assert b1.indices == list1

    b1.foo = tensor2
    b1.indices = list2
    assert (b1.foo == tensor2).all()
    assert b1.indices == list2
