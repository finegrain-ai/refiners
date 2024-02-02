import torch
from torch import Size, Tensor, device as Device, dtype as DType
from torch.nn import Parameter as TorchParameter

from refiners.fluxion.layers.module import Module, WeightedModule


class Identity(Module):
    """Identity operator layer.

    This layer simply returns the input tensor.

    Example:
        ```py
        identity = fl.Identity()

        tensor = torch.randn(10, 10)
        output = identity(tensor)

        assert torch.equal(tensor, output)
        ```
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class View(Module):
    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.view(*self.shape)


class GetArg(Module):
    """GetArg operation layer.

    This layer returns the nth tensor of the input arguments.

    Example:
        ```py
        get_arg = fl.GetArg(1)

        inputs = (
            torch.randn(10, 10),
            torch.randn(20, 20),
            torch.randn(30, 30),
        )
        output = get_arg(*inputs)

        assert id(inputs[1]) == id(output)
        ```
    """

    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index

    def forward(self, *args: Tensor) -> Tensor:
        return args[self.index]


class Flatten(Module):
    """Flatten operation layer.

    This layer flattens the input tensor between the given dimensions.
    See also [`torch.flatten`][torch.flatten].

    Example:
        ```py
        flatten = fl.Flatten(start_dim=1)

        tensor = torch.randn(10, 10, 10)
        output = flatten(tensor)

        assert output.shape == (10, 100)
        ```
    """

    def __init__(
        self,
        start_dim: int = 0,
        end_dim: int = -1,
    ) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        return torch.flatten(
            input=x,
            start_dim=self.start_dim,
            end_dim=self.end_dim,
        )


class Unflatten(Module):
    """Unflatten operation layer.

    This layer unflattens the input tensor at the given dimension with the given sizes.
    See also [`torch.unflatten`][torch.unflatten].

    Example:
        ```py
        unflatten = fl.Unflatten(dim=1)

        tensor = torch.randn(10, 100)
        output = unflatten(tensor, sizes=(10, 10))

        assert output_unflatten.shape == (10, 10, 10)
        ```
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor, sizes: Size) -> Tensor:
        return torch.unflatten(
            input=x,
            dim=self.dim,
            sizes=sizes,
        )


class Reshape(Module):
    """Reshape operation layer.

    This layer reshapes the input tensor to a specific shape (which must be compatible with the original shape).
    See also [torch.reshape][torch.reshape].

    Warning:
        The first dimension (batch dimension) is forcefully preserved.

    Example:
        ```py
        reshape = fl.Reshape(5, 2)

        tensor = torch.randn(2, 10, 1)
        output = reshape(tensor)

        assert output.shape == (2, 5, 2)
        ```
    """

    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return torch.reshape(
            input=x,
            shape=(x.shape[0], *self.shape),
        )


class Transpose(Module):
    """Transpose operation layer.

    This layer transposes the input tensor between the two given dimensions.
    See also [`torch.transpose`][torch.transpose].

    Example:
        ```py
        transpose = fl.Transpose(dim0=1, dim1=2)

        tensor = torch.randn(10, 20, 30)
        output = transpose(tensor)

        assert output.shape == (10, 30, 20)
        ```
    """

    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: Tensor) -> Tensor:
        return torch.transpose(
            input=x,
            dim0=self.dim0,
            dim1=self.dim1,
        )


class Permute(Module):
    """Permute operation layer.

    This layer permutes the input tensor according to the given dimensions.
    See also [`torch.permute`][torch.permute].

    Example:
        ```py
        permute = fl.Permute(2, 0, 1)

        tensor = torch.randn(10, 20, 30)
        output = permute(tensor)

        assert output.shape == (30, 10, 20)
        ```
    """

    def __init__(self, *dims: int) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return torch.permute(
            input=x,
            dims=self.dims,
        )


class Slicing(Module):
    """Slicing operation layer.

    This layer slices the input tensor at the given dimension between the given start and end indices.
    See also [`torch.index_select`][torch.index_select].

    Example:
        ```py
        slicing = fl.Slicing(dim=1, start=50)

        tensor = torch.randn(10, 100)
        output = slicing(tensor)

        assert output.shape == (10, 50)
        assert torch.allclose(output, tensor[:, 50:])
        ```
    """

    def __init__(
        self,
        dim: int = 0,
        start: int = 0,
        end: int | None = None,
        step: int = 1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.start = start
        self.end = end
        self.step = step

    def forward(self, x: Tensor) -> Tensor:
        dim_size = x.shape[self.dim]

        # compute start index
        start = self.start if self.start >= 0 else dim_size + self.start
        start = max(min(start, dim_size), 0)

        # compute end index
        end = self.end or dim_size
        end = end if end >= 0 else dim_size + end
        end = max(min(end, dim_size), 0)

        if start >= end:
            return self._get_empty_slice(x)

        # compute indices
        indices = torch.arange(
            start=start,
            end=end,
            step=self.step,
            device=x.device,
        )

        return torch.index_select(
            input=x,
            dim=self.dim,
            index=indices,
        )

    def _get_empty_slice(self, x: Tensor) -> Tensor:
        """Get an empty slice of the same shape as the input tensor (to mimic PyTorch's slicing behavior)."""

        shape = list(x.shape)
        shape[self.dim] = 0
        return torch.empty(*shape, device=x.device)


class Squeeze(Module):
    """Squeeze operation layer.

    This layer squeezes the input tensor at the given dimension.
    See also [`torch.squeeze`][torch.squeeze].

    Example:
        ```py
        squeeze = fl.Squeeze(dim=1)

        tensor = torch.randn(10, 1, 10)
        output = squeeze(tensor)

        assert output.shape == (10, 10)
        ```
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return torch.squeeze(
            input=x,
            dim=self.dim,
        )


class Unsqueeze(Module):
    """Unsqueeze operation layer.

    This layer unsqueezes the input tensor at the given dimension.
    See also [`torch.unsqueeze`][torch.unsqueeze].

    Example:
        ```py
        unsqueeze = fl.Unsqueeze(dim=1)

        tensor = torch.randn(10, 10)
        output = unsqueeze(tensor)

        assert output.shape == (10, 1, 10)
        ```
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return torch.unsqueeze(
            input=x,
            dim=self.dim,
        )


class Sin(Module):
    """Sine operator layer.

    This layer applies the sine function to the input tensor.
    See also [`torch.sin`][torch.sin].

    Example:
        ```py
        sin = fl.Sin()

        tensor = torch.tensor([0, torch.pi])
        output = sin(tensor)

        expected_output = torch.tensor([0.0, 0.0])
        assert torch.allclose(output, expected_output, atol=1e-6)
        ```
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(input=x)


class Cos(Module):
    """Cosine operator layer.

    This layer applies the cosine function to the input tensor.
    See also [`torch.cos`][torch.cos].

    Example:
        ```py
        cos = fl.Cos()

        tensor = torch.tensor([0, torch.pi])
        output = cos(tensor)

        expected_output = torch.tensor([1.0, -1.0])
        assert torch.allclose(output, expected_output, atol=1e-6)
        ```
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.cos(input=x)


class Multiply(Module):
    """Multiply operator layer.

    This layer scales and shifts the input tensor by the given scale and bias.

    Example:
        ```py
        multiply = fl.Multiply(scale=2, bias=1)

        tensor = torch.ones(1)
        output = multiply(tensor)

        assert torch.allclose(output, torch.tensor([3.0]))
        ```
    """

    def __init__(
        self,
        scale: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.bias = bias

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * x + self.bias


class Parameter(WeightedModule):
    """Parameter layer.

    This layer simple wraps a PyTorch [`Parameter`][torch.nn.parameter.Parameter].
    When called, it simply returns the [`Parameter`][torch.nn.parameter.Parameter] Tensor.

    Attributes:
        weight (torch.nn.parameter.Parameter): The parameter Tensor.
    """

    def __init__(
        self,
        *dims: int,
        requires_grad: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.weight = TorchParameter(
            requires_grad=requires_grad,
            data=torch.randn(
                *dims,
                device=device,
                dtype=dtype,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.weight.expand(x.shape[0], *self.dims)

    @property
    def requires_grad(self) -> bool:
        return self.weight.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self.weight.requires_grad = value
