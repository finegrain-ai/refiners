from json import load
from torch import Tensor, randn, tensor; load as torch_load
from refiners.training_utils.batch import BaseBatch
from pathlib import Path
import pytest

def test_inherit_no_attribute() -> None:
    with pytest.raises(ValueError) as excinfo:
        class NoAttrBatch(BaseBatch):
            pass
    assert "attribute 'foo' is not compatible with Batch, type hint should be list or Tensor" in str(excinfo.value)


def test_inherit_non_list_errors() -> None:
    with pytest.raises(ValueError) as excinfo:
        class StrBatch(BaseBatch):
            foo: str
    assert "attribute 'foo' is not compatible with Batch, type hint should be list or Tensor" in str(excinfo.value)

class TestBatch(BaseBatch):
    foo: Tensor
    bar: Tensor
    indices: list[int]

def test_single_batch() -> None:
    single = TestBatch(foo=randn(1, 10), bar=randn(1, 5), indices=[0])
    assert len(single) == 1

def test_l5_batch() -> None:
    l5 = TestBatch(foo=randn(5, 10), bar=randn(5, 5), indices=[0, 1, 2, 3, 4])
    assert len(l5) == 5
    total = 0
    for single in l5:
        assert len(single) == 1
        total += 1
    
    assert total == 5

def test_extra_missing_attr() -> None:
    with pytest.raises(ValueError) as excinfo:
        TestBatch(foo=randn(5, 10), bar=randn(6, 5))
    
    assert "Attribute 'indices' is missing" in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        TestBatch(foo=randn(5, 10), bar=randn(5, 5), indices=[0, 1, 2], extra=["a", "b", "c"])
    
    assert "Attribute 'extra' is not valid" in str(excinfo.value)
    
def test_inhomogeneous_sizes_raise_errors() -> None:
    with pytest.raises(ValueError) as excinfo:
        TestBatch(foo=randn(5, 10), bar=randn(6, 5), indices=[0, 1, 2, 3, 4])
    
    assert "Attribute 'bar' has size 6, expected 5" in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        TestBatch(foo=randn(5, 10), bar=randn(5, 5), indices=[0, 1, 2])
    
    assert "Attribute 'indices' has size 3, expected 5" in str(excinfo.value)
    
def test_empty_attr_raise_errors() -> None:
    with pytest.raises(ValueError) as excinfo:
        TestBatch(foo=randn(0, 10), bar=randn(6, 5), indices=[0, 1, 2, 3, 4])
    
    assert "Attribute 'foo' is empty, empty attributes are not permitted" in str(excinfo.value)

def test_load_save_batch() -> None:
    loaded = TestBatch(foo=randn(3, 10), bar=randn(3, 5), indices=[1,2,3])
    tmp_filename = Path(".pytest_cache/test_batch_tmp.pt")
    loaded.save(tmp_filename)
    second_loaded = TestBatch.load(tmp_filename)
    assert second_loaded == loaded

def test_equality() -> None:
    b1 = TestBatch(foo=randn(3, 10), bar=randn(3, 5), indices=[1,2,3])
    b2 = b1.clone()
    assert b2 == b1
    assert not b1 != b2
    b3 = TestBatch(foo=randn(3, 10), bar=randn(3, 5), indices=[1,2,3])
    assert b3 != b1
