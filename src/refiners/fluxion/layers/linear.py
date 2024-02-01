from torch import device as Device, dtype as DType
from torch.nn import Linear as _Linear

from refiners.fluxion.layers.activations import ReLU
from refiners.fluxion.layers.chain import Chain
from refiners.fluxion.layers.module import Module, WeightedModule


class Linear(_Linear, WeightedModule):
    """Linear layer.

    This layer wraps [`torch.nn.Linear`][torch.nn.Linear].

    Receives:
        Input (Float[Tensor, "batch in_features"]):

    Returns:
        Output (Float[Tensor, "batch out_features"]):

    Example:
        ```py
        linear = fl.Linear(in_features=32, out_features=128)

        tensor = torch.randn(2, 32)
        output = linear(tensor)

        assert output.shape == (2, 128)
        ```
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initializes the Linear layer.

        Args:
            in_features: The number of input features.
            out_features: The number of output features.
            bias: If True, adds a learnable bias to the output.
            device: The device to use for the linear layer.
            dtype: The dtype to use for the linear layer.
        """
        self.in_features = in_features
        self.out_features = out_features
        super().__init__(  # type: ignore
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )


class MultiLinear(Chain):
    """Multi-layer linear network.

    This layer wraps multiple [`torch.nn.Linear`][torch.nn.Linear] layers,
    with an [`Activation`][refiners.fluxion.layers.Activation] layer in between.

    Receives:
        Input (Float[Tensor, "batch input_dim"]):

    Returns:
        Output (Float[Tensor, "batch output_dim"]):

    Example:
        ```py
        linear = fl.MultiLinear(
            input_dim=32,
            output_dim=128,
            inner_dim=64,
            num_layers=3,
        )

        tensor = torch.randn(2, 32)
        output = linear(tensor)

        assert output.shape == (2, 128)
        ```
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        inner_dim: int,
        num_layers: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initializes the MultiLinear layer.

        Args:
            input_dim: The input dimension of the first linear layer.
            output_dim: The output dimension of the last linear layer.
            inner_dim: The output dimension of the inner linear layers.
            num_layers: The number of linear layers.
            device: The device to use for the linear layers.
            dtype: The dtype to use for the linear layers.
        """
        layers: list[Module] = []
        for i in range(num_layers - 1):
            layers.append(
                Linear(
                    in_features=input_dim if i == 0 else inner_dim,
                    out_features=inner_dim,
                    device=device,
                    dtype=dtype,
                )
            )
            layers.append(
                ReLU(),
            )
        layers.append(
            Linear(
                in_features=inner_dim,
                out_features=output_dim,
                device=device,
                dtype=dtype,
            )
        )

        super().__init__(layers)
