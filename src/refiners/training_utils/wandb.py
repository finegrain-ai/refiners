from typing import Any

import wandb
from PIL import Image

__all__ = [
    "WandbLogger",
    "WandbLoggable",
]


number = float | int
WandbLoggable = number | Image.Image | list[number] | dict[str, list[number]]


def convert_to_wandb(value: WandbLoggable) -> Any:
    match value:
        case Image.Image():
            return convert_to_wandb_image(value=value)
        case list():
            return convert_to_wandb_histogram(value=value)
        case dict():
            return convert_to_wandb_table(value=value)
        case _:
            return value


def convert_to_wandb_image(value: Image.Image) -> wandb.Image:
    return wandb.Image(data_or_path=value)


def convert_to_wandb_histogram(value: list[number]) -> wandb.Histogram:
    return wandb.Histogram(sequence=value)


def convert_to_wandb_table(value: dict[str, list[number]]) -> wandb.Table:
    assert all(
        isinstance(v, list) and len(v) == len(next(iter(value.values()))) for v in value.values()
    ), "Expected a dictionary of lists of the same size"
    columns = list(value.keys())
    data_rows = list(zip(*value.values()))
    return wandb.Table(columns=columns, data=data_rows)


class WandbLogger:
    def __init__(self, init_config: dict[str, Any] = {}) -> None:
        self.wandb_run = wandb.init(**init_config)  # type: ignore

    def log(self, data: dict[str, WandbLoggable], step: int) -> None:
        converted_data = {key: convert_to_wandb(value=value) for key, value in data.items()}
        self.wandb_run.log(converted_data, step=step)  # type: ignore

    def update_summary(self, key: str, value: Any) -> None:
        self.wandb_run.summary[key] = value  # type: ignore

    @property
    def project_name(self) -> str:
        return self.wandb_run.project_name()  # type: ignore

    @property
    def run_name(self) -> str:
        return self.wandb_run.name or ""  # type: ignore
