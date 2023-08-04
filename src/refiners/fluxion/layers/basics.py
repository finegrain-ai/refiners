from refiners.fluxion.layers.module import Module, WeightedModule
from torch import randn, Tensor, Size, device as Device, dtype as DType
from torch.nn import Parameter as TorchParameter


class Identity(Module):
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

    def __repr__(self):
        shape_repr = ", ".join([repr(s) for s in self.shape])
        return f"{self.__class__.__name__}({shape_repr})"


class Flatten(Module):
    def __init__(self, start_dim: int = 0, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim, self.end_dim)

    def __repr__(self):
        return f"{self.__class__.__name__}(start_dim={repr(self.start_dim)}, end_dim={repr(self.end_dim)})"


class Unflatten(Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor, sizes: Size) -> Tensor:
        return x.unflatten(self.dim, sizes)  # type: ignore

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={repr(self.dim)})"


class Reshape(Module):
    """
    Reshape the input tensor to the given shape. The shape must be compatible with the input tensor shape. The batch
    dimension is preserved.
    """

    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], *self.shape)

    def __repr__(self):
        shape_repr = ", ".join([repr(s) for s in self.shape])
        return f"{self.__class__.__name__}({shape_repr})"


class Transpose(Module):
    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(self.dim0, self.dim1)

    def __repr__(self):
        return f"{self.__class__.__name__}(dim0={repr(self.dim0)}, dim1={repr(self.dim1)})"


class Permute(Module):
    def __init__(self, *dims: int) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(*self.dims)

    def __repr__(self):
        dims_repr = ", ".join([repr(d) for d in self.dims])
        return f"{self.__class__.__name__}({dims_repr})"


class Slicing(Module):
    def __init__(self, dim: int, start: int, length: int) -> None:
        super().__init__()
        self.dim = dim
        self.start = start
        self.length = length

    def forward(self, x: Tensor) -> Tensor:
        return x.narrow(self.dim, self.start, self.length)

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={repr(self.dim)}, start={repr(self.start)}, length={repr(self.length)})"


class Squeeze(Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.squeeze(self.dim)

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={repr(self.dim)})"


class Unsqueeze(Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.unsqueeze(self.dim)

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={repr(self.dim)})"


class Parameter(WeightedModule):
    """
    A layer that wraps a tensor as a parameter. This is useful to create a parameter that is not a weight or a bias.
    """

    def __init__(self, *dims: int, device: Device | str | None = None, dtype: DType | None = None) -> None:
        super().__init__()
        self.register_parameter("parameter", TorchParameter(randn(*dims, device=device, dtype=dtype)))

    @property
    def device(self) -> Device:
        return self.parameter.device

    @property
    def dtype(self) -> DType:
        return self.parameter.dtype

    def forward(self, _: Tensor) -> Tensor:
        return self.parameter

    def __repr__(self):
        dims_repr = ", ".join([repr(d) for d in list(self.parameter.shape)])
        return f"{self.__class__.__name__}({dims_repr}, device={repr(self.device)})"


class Buffer(WeightedModule):
    """
    A layer that wraps a tensor as a buffer. This is useful to create a buffer that is not a weight or a bias.

    Buffers are not trainable.
    """

    def __init__(self, *dims: int, device: Device | str | None = None, dtype: DType | None = None) -> None:
        super().__init__()
        self.register_buffer("buffer", randn(*dims, device=device, dtype=dtype))

    @property
    def device(self) -> Device:
        return self.buffer.device

    @property
    def dtype(self) -> DType:
        return self.buffer.dtype

    def forward(self, _: Tensor) -> Tensor:
        return self.buffer

    def __repr__(self):
        dims_repr = ", ".join([repr(d) for d in list(self.buffer.shape)])
        return f"{self.__class__.__name__}({dims_repr}, device={repr(self.device)})"
