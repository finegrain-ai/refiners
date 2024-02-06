from abc import ABC, abstractmethod
from typing import Any, Generic, Iterator, TypeVar, cast

from torch import Tensor, device as Device, dtype as DType
from torch.nn import Parameter as TorchParameter
from torch.nn.init import normal_, zeros_

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.adapter import Adapter

T = TypeVar("T", bound=fl.WeightedModule)


class Lora(Generic[T], fl.Chain, ABC):
    """Low-Rank Adaptation (LoRA) layer.

    This layer is composed of two [`WeightedModule`][refiners.fluxion.layers.WeightedModule]:

    - `down`: initialized with a random normal distribution
    - `up`: initialized with zeros

    Note:
        This layer is not meant to be used directly.
        Instead, use one of its subclasses:

        - [`LinearLora`][refiners.fluxion.adapters.lora.LinearLora]
        - [`Conv2dLora`][refiners.fluxion.adapters.lora.Conv2dLora]
    """

    def __init__(
        self,
        name: str,
        /,
        rank: int = 16,
        scale: float = 1.0,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize the LoRA layer.

        Args:
            name: The name of the LoRA.
            rank: The rank of the LoRA.
            scale: The scale of the LoRA.
            device: The device of the LoRA weights.
            dtype: The dtype of the LoRA weights.
        """
        self.name = name
        self._rank = rank
        self._scale = scale

        super().__init__(
            *self.lora_layers(device=device, dtype=dtype),
            fl.Multiply(scale),
        )

        normal_(tensor=self.down.weight, std=1 / self.rank)
        zeros_(tensor=self.up.weight)

    @abstractmethod
    def lora_layers(self, device: Device | str | None = None, dtype: DType | None = None) -> tuple[T, T]:
        """Create the down and up layers of the LoRA.

        Args:
            device: The device of the LoRA weights.
            dtype: The dtype of the LoRA weights.
        """
        ...

    @property
    def down(self) -> T:
        """The down layer."""
        down_layer = self[0]
        assert isinstance(down_layer, fl.WeightedModule)
        return cast(T, down_layer)

    @property
    def up(self) -> T:
        """The up layer."""
        up_layer = self[1]
        assert isinstance(up_layer, fl.WeightedModule)
        return cast(T, up_layer)

    @property
    def rank(self) -> int:
        """The rank of the low-rank approximation."""
        return self._rank

    @property
    def scale(self) -> float:
        """The scale of the low-rank approximation."""
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value
        self.ensure_find(fl.Multiply).scale = value

    @classmethod
    def from_weights(
        cls,
        name: str,
        /,
        down: Tensor,
        up: Tensor,
    ) -> "Lora[Any]":
        match (up.ndim, down.ndim):
            case (2, 2):
                return LinearLora.from_weights(name, up=up, down=down)
            case (4, 4):
                return Conv2dLora.from_weights(name, up=up, down=down)
            case _:
                raise ValueError(f"Unsupported weight shapes: up={up.shape}, down={down.shape}")

    @classmethod
    def from_dict(cls, name: str, /, state_dict: dict[str, Tensor]) -> dict[str, "Lora[Any]"]:
        """
        Create a dictionary of LoRA layers from a state dict.

        Expects the state dict to be a succession of down and up weights.
        """
        state_dict = {k: v for k, v in state_dict.items() if ".weight" in k}
        loras: dict[str, Lora[Any]] = {}
        for down_key, down_tensor, up_tensor in zip(
            list(state_dict.keys())[::2], list(state_dict.values())[::2], list(state_dict.values())[1::2]
        ):
            key = ".".join(down_key.split(".")[:-2])
            loras[key] = cls.from_weights(name, down=down_tensor, up=up_tensor)
        return loras

    @abstractmethod
    def is_compatible(self, layer: fl.WeightedModule, /) -> bool:
        ...

    def auto_attach(
        self, target: fl.Chain, exclude: list[str] | None = None
    ) -> "tuple[LoraAdapter, fl.Chain | None] | None":
        for layer, parent in target.walk(self.up.__class__):
            if isinstance(parent, Lora):
                continue

            if exclude is not None and any(
                [any([p.__class__.__name__ == e for p in parent.get_parents() + [parent]]) for e in exclude]
            ):
                continue

            if not self.is_compatible(layer):
                continue

            if isinstance(parent, LoraAdapter):
                if self.name in parent.names:
                    continue
                else:
                    parent.add_lora(self)
                    return parent, None

            return LoraAdapter(layer, self), parent

    def load_weights(self, down_weight: Tensor, up_weight: Tensor) -> None:
        """Load the (pre-trained) weights of the LoRA.

        Args:
            down_weight: The down weight.
            up_weight: The up weight.
        """
        assert down_weight.shape == self.down.weight.shape
        assert up_weight.shape == self.up.weight.shape
        self.down.weight = TorchParameter(down_weight.to(device=self.device, dtype=self.dtype))
        self.up.weight = TorchParameter(up_weight.to(device=self.device, dtype=self.dtype))


class LinearLora(Lora[fl.Linear]):
    """Low-Rank Adaptation (LoRA) layer for linear layers.

    This layer uses two [`Linear`][refiners.fluxion.layers.Linear] layers as its down and up layers.
    """

    def __init__(
        self,
        name: str,
        /,
        in_features: int,
        out_features: int,
        rank: int = 16,
        scale: float = 1.0,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize the LoRA layer.

        Args:
            name: The name of the LoRA.
            in_features: The number of input features.
            out_features: The number of output features.
            rank: The rank of the LoRA.
            scale: The scale of the LoRA.
            device: The device of the LoRA weights.
            dtype: The dtype of the LoRA weights.
        """
        self.in_features = in_features
        self.out_features = out_features

        super().__init__(
            name,
            rank=rank,
            scale=scale,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_weights(
        cls,
        name: str,
        /,
        down: Tensor,
        up: Tensor,
    ) -> "LinearLora":
        assert up.ndim == 2 and down.ndim == 2
        assert down.shape[0] == up.shape[1], f"Rank mismatch: down rank={down.shape[0]} and up rank={up.shape[1]}"
        lora = cls(
            name,
            in_features=down.shape[1],
            out_features=up.shape[0],
            rank=down.shape[0],
            device=up.device,
            dtype=up.dtype,
        )
        lora.load_weights(down_weight=down, up_weight=up)
        return lora

    def lora_layers(
        self, device: Device | str | None = None, dtype: DType | None = None
    ) -> tuple[fl.Linear, fl.Linear]:
        return (
            fl.Linear(
                in_features=self.in_features,
                out_features=self.rank,
                bias=False,
                device=device,
                dtype=dtype,
            ),
            fl.Linear(
                in_features=self.rank,
                out_features=self.out_features,
                bias=False,
                device=device,
                dtype=dtype,
            ),
        )

    def is_compatible(self, layer: fl.WeightedModule, /) -> bool:
        if isinstance(layer, fl.Linear):
            return layer.in_features == self.in_features and layer.out_features == self.out_features
        return False


class Conv2dLora(Lora[fl.Conv2d]):
    """Low-Rank Adaptation (LoRA) layer for 2D convolutional layers.

    This layer uses two [`Conv2d`][refiners.fluxion.layers.Conv2d] layers as its down and up layers.
    """

    def __init__(
        self,
        name: str,
        /,
        in_channels: int,
        out_channels: int,
        rank: int = 16,
        scale: float = 1.0,
        kernel_size: tuple[int, int] = (1, 3),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 1),
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize the LoRA layer.

        Args:
            name: The name of the LoRA.
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            rank: The rank of the LoRA.
            scale: The scale of the LoRA.
            kernel_size: The kernel size of the LoRA.
            stride: The stride of the LoRA.
            padding: The padding of the LoRA.
            device: The device of the LoRA weights.
            dtype: The dtype of the LoRA weights.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        super().__init__(
            name,
            rank=rank,
            scale=scale,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_weights(
        cls,
        name: str,
        /,
        down: Tensor,
        up: Tensor,
    ) -> "Conv2dLora":
        assert up.ndim == 4 and down.ndim == 4
        assert down.shape[0] == up.shape[1], f"Rank mismatch: down rank={down.shape[0]} and up rank={up.shape[1]}"
        down_kernel_size, up_kernel_size = down.shape[2], up.shape[2]
        # padding is set so the spatial dimensions are preserved (assuming stride=1 and kernel_size either 1 or 3)
        down_padding = 1 if down_kernel_size == 3 else 0
        up_padding = 1 if up_kernel_size == 3 else 0
        lora = cls(
            name,
            in_channels=down.shape[1],
            out_channels=up.shape[0],
            rank=down.shape[0],
            kernel_size=(down_kernel_size, up_kernel_size),
            padding=(down_padding, up_padding),
            device=up.device,
            dtype=up.dtype,
        )
        lora.load_weights(down_weight=down, up_weight=up)
        return lora

    def lora_layers(
        self, device: Device | str | None = None, dtype: DType | None = None
    ) -> tuple[fl.Conv2d, fl.Conv2d]:
        return (
            fl.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.rank,
                kernel_size=self.kernel_size[0],
                stride=self.stride[0],
                padding=self.padding[0],
                use_bias=False,
                device=device,
                dtype=dtype,
            ),
            fl.Conv2d(
                in_channels=self.rank,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size[1],
                stride=self.stride[1],
                padding=self.padding[1],
                use_bias=False,
                device=device,
                dtype=dtype,
            ),
        )

    def is_compatible(self, layer: fl.WeightedModule, /) -> bool:
        if (
            isinstance(layer, fl.Conv2d)
            and layer.in_channels == self.in_channels
            and layer.out_channels == self.out_channels
        ):
            # stride cannot be inferred from the weights, so we assume it's the same as the layer
            self.down.stride = layer.stride

            return True
        return False


class LoraAdapter(fl.Sum, Adapter[fl.WeightedModule]):
    """Adapter for LoRA layers.

    This adapter simply sums the target layer with the given LoRA layers.
    """

    def __init__(self, target: fl.WeightedModule, /, *loras: Lora[Any]) -> None:
        """Initialize the adapter.

        Args:
            target: The target layer.
            loras: The LoRA layers.
        """
        with self.setup_adapter(target):
            super().__init__(target, *loras)

    @property
    def lora_layers(self) -> Iterator[Lora[Any]]:
        """The LoRA layers."""
        return cast(Iterator[Lora[Any]], self.layers(Lora))

    @property
    def names(self) -> list[str]:
        """The names of the LoRA layers."""
        return [lora.name for lora in self.lora_layers]

    @property
    def loras(self) -> dict[str, Lora[Any]]:
        """The LoRA layers indexed by name."""
        return {lora.name: lora for lora in self.lora_layers}

    @property
    def scales(self) -> dict[str, float]:
        """The scales of the LoRA layers indexed by names."""
        return {lora.name: lora.scale for lora in self.lora_layers}

    @scales.setter
    def scale(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            self.loras[name].scale = value

    def add_lora(self, lora: Lora[Any], /) -> None:
        """Add a LoRA layer to the adapter.

        Raises:
            AssertionError: If the adapter already contains a LoRA layer with the same name.

        Args:
            lora: The LoRA layer to add.
        """
        assert lora.name not in self.names, f"LoRA layer with name {lora.name} already exists"
        self.append(lora)

    def remove_lora(self, name: str, /) -> Lora[Any] | None:
        """Remove a LoRA layer from the adapter.

        Note:
            If the adapter doesn't contain a LoRA layer with the given name, nothing happens and `None` is returned.

        Args:
            name: The name of the LoRA layer to remove.
        """
        if name in self.names:
            lora = self.loras[name]
            self.remove(lora)
            return lora
