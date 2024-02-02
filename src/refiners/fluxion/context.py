from typing import Any

from torch import Tensor

Context = dict[str, Any]
Contexts = dict[str, Context]


class ContextProvider:
    """A class that provides a context store."""

    def __init__(self) -> None:
        """Initializes the ContextProvider."""
        self.contexts: Contexts = {}

    def set_context(self, key: str, value: Context) -> None:
        """Store a value in the context.

        Args:
            key: The key of the context.
            value: The context.
        """
        self.contexts[key] = value

    def get_context(self, key: str) -> Any:
        """Retrieve a value from the context.

        Args:
            key: The key of the context.

        Returns:
            The context value.
        """
        return self.contexts.get(key)

    def update_contexts(self, new_contexts: Contexts) -> None:
        """Update or set the contexts with new contexts.

        Args:
            new_contexts: The new contexts.
        """
        for key, value in new_contexts.items():
            if key not in self.contexts:
                self.contexts[key] = value
            else:
                self.contexts[key].update(value)

    @staticmethod
    def create(contexts: Contexts) -> "ContextProvider":
        """Create a ContextProvider from a dict of contexts.

        Args:
            contexts: The contexts.

        Returns:
            A ContextProvider with the contexts.
        """
        provider = ContextProvider()
        provider.update_contexts(contexts)
        return provider

    def _get_repr_for_value(self, value: Any) -> str:
        if isinstance(value, Tensor):
            return f"Tensor(shape={value.shape}, dtype={value.dtype}, device={value.device})"
        return repr(value)

    def _get_repr_for_dict(self, context_dict: Context) -> dict[str, str]:
        return {key: self._get_repr_for_value(value) for key, value in context_dict.items()}

    def __repr__(self) -> str:
        contexts_repr = {key: self._get_repr_for_dict(value) for key, value in self.contexts.items()}
        return f"{self.__class__.__name__}(contexts={contexts_repr})"
