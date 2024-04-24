import pytest
from pydantic import ValidationError
from torch.utils.data import DataLoader

from refiners.training_utils.data_loader import DataLoaderConfig, DatasetFromCallable, create_data_loader


def get_item(index: int) -> int:
    return index * 2


@pytest.fixture
def config() -> DataLoaderConfig:
    return DataLoaderConfig(batch_size=2, num_workers=2, persistent_workers=True)


def test_dataloader_config_valid(config: DataLoaderConfig) -> None:
    assert config.batch_size == 2
    assert config.num_workers == 2
    assert config.persistent_workers == True


def test_dataloader_config_invalid() -> None:
    with pytest.raises(ValidationError):
        DataLoaderConfig(num_workers=0, prefetch_factor=2)

    with pytest.raises(ValidationError):
        DataLoaderConfig(num_workers=0, persistent_workers=True)


def test_dataset_from_callable():
    dataset = DatasetFromCallable(get_item, 200)
    assert len(dataset) == 200
    assert dataset[0] == 0
    assert dataset[5] == 10


def test_create_data_loader(config: DataLoaderConfig) -> None:
    data_loader = create_data_loader(get_item, 100, config)
    assert isinstance(data_loader, DataLoader)


def test_create_data_loader_with_collate_fn(config: DataLoaderConfig) -> None:
    def collate_fn(batch: list[int]):
        return sum(batch)

    data_loader = create_data_loader(get_item, 20, config=config, collate_fn=collate_fn)
    assert isinstance(data_loader, DataLoader)
