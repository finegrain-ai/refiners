from pathlib import Path
from typing import Any, Type, TypeVar, cast, get_origin, get_type_hints

from torch import Tensor, cat, device as Device, dtype as DType, load as torch_load, save as torch_save  # type: ignore
from typing_extensions import dataclass_transform

from refiners.fluxion.utils import summarize_tensor  # type: ignore

T = TypeVar("T", bound="BaseBatch")
AttrType = Tensor | list[Any]


def simple_hint(hint: Type[Any]) -> Type[Any]:
    origin = get_origin(hint)
    if origin is None:  # for Tensor
        return hint
    return origin


@dataclass_transform()
class TypeCheckMeta(type):
    def __new__(cls, name: str, bases: tuple[type, ...], dct: dict[str, Any]) -> type:
        new_class: type["BaseBatch"] = super().__new__(cls, name, bases, dct)

        if new_class.__name__ == "BaseBatch":
            return new_class

        attr_types = new_class.attr_types()
        if len(attr_types) == 0:
            raise ValueError(f"At least one attribute is required for '{new_class.__name__}'")

        for attr_name, attr_type in attr_types.items():
            if not attr_type in [Tensor, list]:
                raise TypeError(f"Type of '{attr_name}' must be 'Tensor' or 'list', got '{attr_type.__name__}'")

        return new_class


class BaseBatch(metaclass=TypeCheckMeta):
    @classmethod
    def attr_types(cls: Type[T]) -> dict[str, Type[AttrType]]:
        type_hints = get_type_hints(cls)
        return {name: simple_hint(hint) for name, hint in type_hints.items()}

    def __init__(self, **kwargs: AttrType):
        attr_types = self.__class__.attr_types()

        size = None

        for attr_name in kwargs:
            if attr_name not in attr_types:
                raise ValueError(f"Attribute '{attr_name}' is not valid")

        for attr_name in attr_types:
            if attr_name not in kwargs:
                raise ValueError(f"Missing required attribute '{attr_name}'")

        for attr_name in kwargs:
            self.__setattr__(attr_name, kwargs[attr_name], check_size=False)
            new_size = self.attr_size(attr_name)
            if size is not None and size != new_size:
                raise ValueError(f"Attribute '{attr_name}' has size {new_size}, expected {size}")
            size = new_size

            if size == 0:
                raise ValueError(f"Attribute '{attr_name}' is empty, empty attributes are not permitted")

    def __getattr__(self, name: str) -> AttrType:
        if name in self.__class__.attr_types():
            return getattr(self, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any, check_size: bool = True) -> None:
        attr_types = self.__class__.attr_types()
        if name in attr_types:
            attr_type = attr_types[name]
            if not isinstance(value, attr_type):
                raise TypeError(
                    f"Invalid type for attribute '{name}': Expected '{attr_type.__name__}', got '{type(value).__name__}'"
                )
            
            new_size = len(value) if isinstance(value, list) else value.shape[0]

            if check_size and new_size != len(self):
                raise ValueError(f"Attribute '{name}' has size {new_size}, expected {len(self)}")

            super().__setattr__(name, value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @classmethod
    def collate(cls: Type[T], batch_list: list[T]) -> T:
        collated_attrs: dict[str, Any] = {}
        attr_types = cls.attr_types()

        if len(batch_list) == 0:
            raise ValueError(f"Cannot collate an empty list of {cls.__name__}")

        for attr_name, attr_type in attr_types.items():
            attr_list = [obj.__getattr__(attr_name) for obj in batch_list]

            if attr_type == Tensor:
                tensor_tuple = cast(tuple[Tensor, ...], tuple(attr_list))
                collated_attrs[attr_name] = cat(tensor_tuple, dim=0)
            else:
                collated_attrs[attr_name] = [item for sublist in attr_list for item in sublist]

        return cls(**collated_attrs)

    def __add__(self: T, other: Any) -> T:
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unsupported type for addition: {type(other)}")

        collated_attrs: dict[str, Any] = {}
        attr_types = self.__class__.attr_types()
        for attr_name, attr_type in attr_types.items():
            self_attr = getattr(self, attr_name)
            if attr_type == Tensor:
                collated_attrs[attr_name] = cat(tensors=(self_attr, getattr(other, attr_name)), dim=0)
            else:
                collated_attrs[attr_name] = self_attr + getattr(other, attr_name)

        return self.__class__(**collated_attrs)

    def to(self: T, device: Device | None = None, dtype: DType | None = None) -> T:
        attr_types = self.__class__.attr_types()
        for attr_name, attr_type in attr_types.items():
            if attr_type == Tensor:
                value = cast(Tensor, self.__getattr__(attr_name))
                self.__setattr__(attr_name, value.to(device, dtype))

        return self

    def attr_size(self, name: str) -> int:
        value = self.__getattr__(name)
        if isinstance(value, list):
            return len(value)
        else:
            return value.shape[0]

    def __len__(self) -> int:
        return self.attr_size(list(self.__class__.attr_types().keys())[0])

    def to_dict(self) -> dict[str, AttrType]:
        return {attr_name: getattr(self, attr_name) for attr_name in self.__class__.attr_types()}

    def split(self: T) -> list[T]:
        result: list[T] = []
        for i in range(len(self)):
            args = {attr_name: getattr(self, attr_name)[i : i + 1] for attr_name in self.__class__.attr_types()}
            result.append(self.__class__(**args))
        return result

    def __iter__(self):
        for batch in self.split():
            yield batch

    @classmethod
    def load(cls: Type[T], filename: Path, map_location: Device | None = None) -> T:
        return cls(**torch_load(filename, map_location=map_location))

    def save(self, filename: Path) -> None:
        torch_save(self.to_dict(), filename)

    def clone(self: T) -> T:
        return self.__class__(**self.to_dict())

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, self.__class__):
            return False
        dict = self.to_dict()
        other_dict = __value.to_dict()

        for attr_name in dict:
            if isinstance(dict[attr_name], Tensor):
                condition = dict[attr_name] == other_dict[attr_name]
                casted = cast(Tensor, condition)
                if not casted.all():
                    return False
            else:
                if dict[attr_name] != other_dict[attr_name]:
                    return False
        return True

    def __neq__(self, __value: object) -> bool:
        return not self.__eq__(__value)

    def __repr__(self) -> str:
        attr_strs: list[str] = []
        for attr_name in self.__class__.attr_types():
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, list):
                attr_strs.append(f"{attr_name}={attr_value}")
            else:
                attr_strs.append(f"{attr_name}={summarize_tensor(attr_value)}")
        return f"{self.__class__.__name__}(size={len(self)})[{','.join(attr_strs)}]"

    def __getitem__(self: T, key: slice | int | list[int] | list[bool]) -> T:
        if isinstance(key, slice):
            return self.__class__(
                **{attr_name: getattr(self, attr_name)[key] for attr_name in self.__class__.attr_types()}
            )
        elif isinstance(key, int):
            return self[key : key + 1]
        else:  # list
            if len(key) == 0:
                raise ValueError("Empty list is not valid")
            if isinstance(key[0], bool):
                if len(key) != len(self):
                    raise ValueError("Boolean list must have the same length as the batch")
                indices: list[int] = list(filter(lambda x: key[x], range(len(self))))
                if len(indices) == 0:
                    raise ValueError("Boolean list must have at least one true value")
                return self[indices]
            else:  # list[int]
                indices = cast(list[int], key)
                batch_list: list[T] = [self[i] for i in indices]
                return self.__class__.collate(batch_list)
