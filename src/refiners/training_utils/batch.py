from math import e
from typing import Type, TypeVar, get_type_hints, get_origin, cast
import attr

from torch import Tensor, cat, device as Device, dtype as DType, empty, stack
from pathlib import Path
from torch import load as torch_load, save as torch_save
T = TypeVar('T', bound='BaseBatch')

def simple_hint(hint: Type) -> Type:
    origin = get_origin(hint)
    if origin is None: # for Tensor
        return hint
    return origin

class TypeCheckMeta(type):
    def __new__(cls, name, bases, dct):
        new_class = super().__new__(cls, name, bases, dct)
        
        if new_class.__name__ == "BaseBatch":
            return new_class
        
        hints = get_type_hints(new_class)
        if len(hints) == 0:
            raise ValueError(f"At least one attribute with type hint is required for {new_class.__name__}")
        
        for attr, hint in hints.items():
            if not simple_hint(hint) in [Tensor, list]:
                raise TypeError(f"Type of {attr} must be Tensor or list, got {hint}")
        return new_class

class BaseBatch(metaclass=TypeCheckMeta):
    
    def __init__(self, **kwargs):

        type_hints = get_type_hints(self.__class__)

        size = None

        for key in type_hints:
            type_hint = type_hints[key]

            if key not in kwargs:
                raise ValueError(f"Missing required attribute: {key} in {kwargs}")

            value = kwargs[key]
            
            s_hint = simple_hint(type_hint)
            if not isinstance(value, s_hint):
                raise ValueError(f"Invalid type for attribute {key}: Expected {s_hint}, got {type(value)}")
            
            if isinstance(value, list):
                new_size = len(value)
            elif isinstance(value, Tensor):
                new_size = value.shape[0]
            else:
                raise ValueError(f"Unsupported attribute type for '{key}': '{type(value)}', should be list or Tensor.")

            if size is not None and size != new_size:
                raise ValueError(f"Attribute {key} has size {new_size}, expected {size}")
            
            size = new_size

            setattr(self, key, kwargs[key])
        
        if size is None:
            raise ValueError(f"Size of the batch cannot be None")
        
        self._length = size

    @classmethod
    def collate(cls: Type[T], batch_list: list[T]) -> T:
        collated_attrs = {}
        type_hints = get_type_hints(cls)
        
        l = len(batch_list)
        
        if l == 0:
            raise ValueError(f"Cannot collate an empty list of {cls.__name__}")
        
        for key in type_hints.keys():
            attr_list = [getattr(obj, key) for obj in batch_list]

            if isinstance(attr_list[0], Tensor):
                collated_attrs[key] = cat(tensors=tuple(attr_list), dim=0)
            elif isinstance(attr_list[0], list):
                collated_attrs[key] = [item for sublist in attr_list for item in sublist]
            else:
                raise ValueError(f"Unsupported attribute type for collation: {type(attr_list[0])}")
            
        collated_instance = cls(**collated_attrs)
        return collated_instance
    
    def to(self: T, device: Device | None = None, dtype: DType | None = None) -> T:
        for key in get_type_hints(self.__class__):
            value = getattr(self, key)
            if isinstance(value, Tensor):
                setattr(self, key, value.to(device, dtype))
            elif isinstance(value, list):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unsupported attribute type for to: {type(value)}")
        return self
    
    def __len__(self) -> int:
        return self._length
    
    def to_dict(self):
        return {attr: getattr(self, attr) for attr in get_type_hints(self.__class__)}
    
    def split(self: T) -> list[T]:
        result : list[T] = []
        l = len(self)
        for i in range(l):
            args = {key: getattr(self, key)[i:i+1] for key in get_type_hints(self.__class__)}
            result.append(self.__class__(**args))
        return result

    def __iter__(self):
        for batch in self.split():
            yield batch
    
    @classmethod
    def load(cls: Type[T], filename: Path) -> T:
        return cls(**torch_load(filename, map_location='cpu'))
    
    def save(self, filename: Path) -> None:
        torch_save(self.to_dict(), filename)

    def clone(self: T) -> T:
        return self.__class__(**self.to_dict())