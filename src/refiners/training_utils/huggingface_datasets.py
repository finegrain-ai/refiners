from datasets import load_dataset as _load_dataset, VerificationMode  # type: ignore
from typing import Any, Generic, Protocol, TypeVar, cast

__all__ = ["load_hf_dataset", "HuggingfaceDataset"]


T = TypeVar("T", covariant=True)


class HuggingfaceDataset(Generic[T], Protocol):
    def __getitem__(self, index: int) -> T: ...

    def __len__(self) -> int: ...


def load_hf_dataset(
    path: str, revision: str = "main", split: str = "train", use_verification: bool = False
) -> HuggingfaceDataset[Any]:
    verification_mode = VerificationMode.BASIC_CHECKS if use_verification else VerificationMode.NO_CHECKS
    dataset = _load_dataset(path=path, revision=revision, split=split, verification_mode=verification_mode)
    return cast(HuggingfaceDataset[Any], dataset)
