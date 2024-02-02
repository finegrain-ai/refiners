from abc import ABC
from enum import Enum

from torch import Tensor
from torch.nn.functional import (
    gelu,
    relu,
    sigmoid,
    silu,
)

from refiners.fluxion.layers.module import Module


class Activation(Module, ABC):
    """Base class for activation layers.

    Activation layers are layers that apply a (non-linear) function to their input.

    Receives:
        x (Tensor):

    Returns:
        (Tensor):
    """

    def __init__(self) -> None:
        super().__init__()


class SiLU(Activation):
    """Sigmoid Linear Unit activation function.

    See [[arXiv:1702.03118] Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning](https://arxiv.org/abs/1702.03118) for more details.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return silu(x)


class ReLU(Activation):
    """Rectified Linear Unit activation function.

    See [Rectified Linear Units Improve Restricted Boltzmann Machines](https://www.cs.toronto.edu/%7Efritz/absps/reluICML.pdf)
    and [Cognitron: A self-organizing multilayered neural network](https://link.springer.com/article/10.1007/BF00342633)

    Example:
        ```py
        relu = fl.ReLU()

        tensor = torch.tensor([[-1.0, 0.0, 1.0]])
        output = relu(tensor)

        expected_output = torch.tensor([[0.0, 0.0, 1.0]])
        assert torch.equal(output, expected_output)
        ```
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


class GeLUApproximation(Enum):
    """Approximation methods for the Gaussian Error Linear Unit activation function.

    Attributes:
        NONE: No approximation, use the original formula.
        TANH: Use the tanh approximation.
        SIGMOID: Use the sigmoid approximation.
    """

    NONE = "none"
    TANH = "tanh"
    SIGMOID = "sigmoid"


class GeLU(Activation):
    """Gaussian Error Linear Unit activation function.

    This activation can be quite expensive to compute, a few approximations are available,
    see [`GeLUApproximation`][refiners.fluxion.layers.activations.GeLUApproximation].

    See [[arXiv:1606.08415] Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415) for more details.

    Example:
        ```py
        gelu = fl.GeLU()

        tensor = torch.tensor([[-1.0, 0.0, 1.0]])
        output = gelu(tensor)
        ```
    """

    def __init__(
        self,
        approximation: GeLUApproximation = GeLUApproximation.NONE,
    ) -> None:
        super().__init__()
        self.approximation = approximation

    def forward(self, x: Tensor) -> Tensor:
        match self.approximation:
            case GeLUApproximation.NONE:
                return gelu(x, approximate="none")
            case GeLUApproximation.TANH:
                return gelu(x, approximate="tanh")
            case GeLUApproximation.SIGMOID:
                return x * sigmoid(1.702 * x)


class Sigmoid(Activation):
    """Sigmoid activation function.

    Example:
        ```py
        sigmoid = fl.Sigmoid()

        tensor = torch.tensor([[-1.0, 0.0, 1.0]])
        output = sigmoid(tensor)
        ```
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)


class GLU(Activation):
    """Gated Linear Unit activation function.

    See [[arXiv:2002.05202] GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) for more details.

    Example:
        ```py
        glu = fl.GLU()

        tensor = torch.tensor([[-1.0, 0.0, 1.0]])
        output = glu(tensor)
        ```
    """

    def __init__(self, activation: Activation) -> None:
        super().__init__()
        self.activation = activation

    def __repr__(self):
        return f"{self.__class__.__name__}(activation={self.activation})"

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] % 2 == 0, "Non-batch input dimension must be divisible by 2"
        output, gate = x.chunk(2, dim=-1)
        return output * self.activation(gate)
