from typing import Callable, TypeVar

from pydantic import BaseModel, ConfigDict, PositiveInt
from torch.utils.data import DataLoader, Dataset

BatchT = TypeVar("BatchT")


class DataLoaderConfig(BaseModel):
    batch_size: PositiveInt = 1
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    drop_last: bool = False
    shuffle: bool = True

    model_config = ConfigDict(extra="forbid")
    # TODO: Add more validation to the config


class DatasetFromCallable(Dataset[BatchT]):
    """
    A wrapper around the `get_item` method to create a [`torch.utils.data.Dataset`][torch.utils.data.Dataset].
    """

    def __init__(self, get_item: Callable[[int], BatchT], length: int) -> None:
        assert length > 0, "Dataset length must be greater than 0."
        self.length = length
        self.get_item = get_item

    def __getitem__(self, index: int) -> BatchT:
        return self.get_item(index)

    def __len__(self) -> int:
        return self.length


def create_data_loader(
    get_item: Callable[[int], BatchT],
    length: int,
    config: DataloaderConfig,
    collate_fn: Callable[[list[BatchT]], BatchT] | None = None,
) -> DataLoader[BatchT]:
    return DataLoader(
        DatasetFromCallable(get_item, length),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        drop_last=config.drop_last,
        shuffle=config.shuffle,
        collate_fn=collate_fn,
    )
