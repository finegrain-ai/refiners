from pathlib import Path
from typing import Any, Type, TypeVar, cast, dataclass_transform, get_origin, get_type_hints

from torch import Tensor, cat, device as Device, dtype as DType, load as torch_load, save as torch_save  # type: ignore

from refiners.fluxion.utils import summarize_tensor  # type: ignore

T = TypeVar("T", bound="BaseBatch")


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

        batch_attr_types = new_class.batch_attr_types()
        if len(batch_attr_types) == 0:
            raise ValueError(f"At least one attribute with type hint is required for {new_class.__name__}")

        for attr_key, attr_hint in batch_attr_types.items():
            if not simple_hint(attr_hint) in [Tensor, list]:
                raise TypeError(f"Type of '{attr_key}' must be Tensor or list, got {attr_hint}")

        return new_class


BatchAttrType = Tensor | list[Any]


class BaseBatch(metaclass=TypeCheckMeta):
    @classmethod
    def batch_attr_types(cls: Type[T]) -> dict[str, Type[BatchAttrType]]:
        type_hints = get_type_hints(cls)
        return {name: simple_hint(hint) for name, hint in type_hints.items() if name != "_length"}

    def __init__(self, **kwargs: BatchAttrType):
        batch_attr_types = self.__class__.batch_attr_types()

        size = None

        for arg_key in kwargs:
            if arg_key not in batch_attr_types:
                raise ValueError(f"Attribute '{arg_key}' is not valid")

        for type_key in batch_attr_types:
            if type_key not in kwargs:
                raise ValueError(f"Missing required attribute '{type_key}'")

        for arg_key in kwargs:
            self.__setattr__(arg_key, kwargs[arg_key], check_size=False)
            new_size = self.attr_length(arg_key)
            if size is not None and size != new_size:
                raise ValueError(f"Attribute '{arg_key}' has size {new_size}, expected {size}")
            size = new_size

            if size == 0:
                raise ValueError(f"Attribute '{arg_key}' is empty, empty attributes are not permitted")

        if size is None:
            raise ValueError(f"Empty batch is not valid")

    def __getattr__(self, name: str) -> BatchAttrType:
        if name in self.__class__.batch_attr_types():
            return getattr(self, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any, check_size: bool = True) -> None:
        batch_attr_types = self.__class__.batch_attr_types()
        if name in batch_attr_types:
            simple_type_hint = batch_attr_types[name]
            if not isinstance(value, simple_type_hint):
                raise TypeError(
                    f"Invalid type for attribute '{name}': Expected {simple_type_hint.__name__}, got {type(value).__name__}"
                )
            if isinstance(value, list):
                new_size = len(value)
            else:
                new_size = value.shape[0]

            if check_size and new_size != len(self):
                raise ValueError(f"Attribute '{name}' has size {new_size}, expected {len(self)}")

            super().__setattr__(name, value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @classmethod
    def collate(cls: Type[T], batch_list: list[T]) -> T:
        collated_attrs: dict[str, Any] = {}
        batch_attr_types = cls.batch_attr_types()

        if len(batch_list) == 0:
            raise ValueError(f"Cannot collate an empty list of {cls.__name__}")

        for type_key, simple_type_hint in batch_attr_types.items():
            attr_list: list[BatchAttrType] = [getattr(obj, type_key) for obj in batch_list]

            if simple_type_hint == Tensor:
                tensor_tuple = cast(tuple[Tensor, ...], tuple(attr_list))
                collated_attrs[type_key] = cat(tensor_tuple, dim=0)
            else:
                collated_attrs[type_key] = [item for sublist in attr_list for item in sublist]

        return cls(**collated_attrs)

    def __add__(self: T, other: Any) -> T:
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unsupported type for addition: {type(other)}")

        collated_attrs: dict[str, Any] = {}
        batch_attr_types = self.__class__.batch_attr_types()
        for type_key, simple_type_hint in batch_attr_types.items():
            self_attr = getattr(self, type_key)
            if simple_type_hint == Tensor:
                collated_attrs[type_key] = cat(tensors=(self_attr, getattr(other, type_key)), dim=0)
            else:
                collated_attrs[type_key] = self_attr + getattr(other, type_key)

        return self.__class__(**collated_attrs)

    def to(self: T, device: Device | None = None, dtype: DType | None = None) -> T:
        batch_attr_types = self.__class__.batch_attr_types()
        for type_key, simple_type_hint in batch_attr_types.items():
            value = getattr(self, type_key)
            if simple_type_hint == Tensor:
                setattr(self, type_key, value.to(device, dtype))
            else:
                setattr(self, type_key, value)

        return self

    def attr_length(self, name: str) -> int:
        value = self.__getattr__(name)
        if isinstance(value, list):
            return len(value)
        else:
            return value.shape[0]

    def __len__(self) -> int:
        return self.attr_length(list(self.__class__.batch_attr_types().keys())[0])

    def to_dict(self) -> dict[str, BatchAttrType]:
        return {type_key: getattr(self, type_key) for type_key in self.__class__.batch_attr_types()}

    def split(self: T) -> list[T]:
        result: list[T] = []
        l = len(self)
        for i in range(l):
            args = {type_key: getattr(self, type_key)[i : i + 1] for type_key in self.__class__.batch_attr_types()}
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

        for key in dict:
            if isinstance(dict[key], Tensor):
                condition = dict[key] == other_dict[key]
                casted = cast(Tensor, condition)
                if not casted.all():
                    return False
            else:
                if dict[key] != other_dict[key]:
                    return False
        return True

    def __neq__(self, __value: object) -> bool:
        return not self.__eq__(__value)

    def __repr__(self) -> str:
        attr_strs: list[str] = []
        for type_key in self.__class__.batch_attr_types():
            attr_value = getattr(self, type_key)
            if isinstance(attr_value, list):
                attr_strs.append(f"{type_key}={attr_value}")
            else:
                attr_strs.append(f"{type_key}={summarize_tensor(attr_value)}")
        return f"{self.__class__.__name__}(size={len(self)})[{','.join(attr_strs)}]"

    def __getitem__(self: T, key: slice | int | list[int] | list[bool]) -> T:
        if isinstance(key, slice):
            return self.__class__(
                **{type_key: getattr(self, type_key)[key] for type_key in self.__class__.batch_attr_types()}
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
