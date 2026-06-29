"""Exceptions related to deepcgp library."""

from pprint import pformat
from typing import Any


class DeepcgpError(Exception):
    """Base exception class for all deepcgp's errors."""

    message: str
    """Error message"""
    extra: dict[str, Any]
    """Error's extra information."""

    def __init__(
        self,
        message: str = "Unexpected error occurred",
        extra: dict[str, Any] | None = None,
    ):
        self.message = message
        self.extra = extra or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return a string representation of the error."""
        if self.extra:
            return f"{self.message}\nExtra information:\n{pformat(self.extra)}"
        return self.message


def _type_fullname(t: type) -> str:
    """Return the full name of a type.

    Prefixed with its module if not a builtin.
    """
    module = t.__module__
    if module and module != "builtins":
        return f"{module}.{t.__qualname__}"
    return t.__qualname__
