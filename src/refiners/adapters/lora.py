import refiners.fluxion.layers as fl
from refiners.adapters.adapter import Adapter

from torch.nn.init import zeros_, normal_
from torch import Tensor, device as Device, dtype as DType


class Lora(fl.Chain):
    structural_attrs = ["in_features", "out_features", "rank", "scale"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale: float = 1.0

        super().__init__(
            fl.Linear(in_features=in_features, out_features=rank, bias=False, device=device, dtype=dtype),
            fl.Linear(in_features=rank, out_features=out_features, bias=False, device=device, dtype=dtype),
            fl.Lambda(func=self.scale_outputs),
        )

        normal_(tensor=self.Linear_1.weight, std=1 / self.rank)
        zeros_(tensor=self.Linear_2.weight)

    def scale_outputs(self, x: Tensor) -> Tensor:
        return x * self.scale

    def set_scale(self, scale: float) -> None:
        self.scale = scale

    def load_weights(self, down_weight: Tensor, up_weight: Tensor) -> None:
        self.Linear_1.weight = down_weight
        self.Linear_2.weight = up_weight


class LoraAdapter(fl.Sum, Adapter[fl.Linear]):
    structural_attrs = ["in_features", "out_features", "rank", "scale"]

    def __init__(
        self,
        target: fl.Linear,
        rank: int = 16,
        scale: float = 1.0,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.in_features = target.in_features
        self.out_features = target.out_features
        self.rank = rank
        self.scale = scale
        with self.setup_adapter(target):
            super().__init__(
                target,
                Lora(
                    in_features=target.in_features,
                    out_features=target.out_features,
                    rank=rank,
                    device=device,
                    dtype=dtype,
                ),
            )
        self.Lora.set_scale(scale=scale)

    def add_lora(self, lora: Lora) -> None:
        self.append(module=lora)

    def load_lora_weights(self, up_weight: Tensor, down_weight: Tensor, index: int = 0) -> None:
        self[index + 1].load_weights(up_weight=up_weight, down_weight=down_weight)


def load_lora_weights(model: fl.Chain, weights: list[Tensor]) -> None:
    assert len(weights) % 2 == 0, "Number of weights must be even"
    assert (
        len(list(model.layers(layer_type=Lora))) == len(weights) // 2
    ), "Number of Lora layers must match number of weights"
    for i, lora in enumerate(iterable=model.layers(layer_type=Lora)):
        assert (
            lora.rank == weights[i * 2].shape[1]
        ), f"Rank of Lora layer {lora.rank} must match shape of weights {weights[i*2].shape[1]}"
        lora.load_weights(up_weight=weights[i * 2], down_weight=weights[i * 2 + 1])
