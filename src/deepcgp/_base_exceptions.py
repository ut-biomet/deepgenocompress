"""Exceptions related to deepcgp library."""

import reprlib
from pprint import pformat
from typing import Any


class _Raw:
    """Wraps a pre-formatted string so pformat inserts it verbatim, unquoted."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


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
            repr = reprlib.Repr()
            truncated_extra = {
                k: _Raw(repr.repr(v)) if isinstance(v, (list, tuple, set)) else v
                for k, v in self.extra.items()
            }
            formated_extra = pformat(truncated_extra, depth=2, sort_dicts=False)
            return f"{self.message}\nExtra information:\n{formated_extra}"
        return self.message


def _type_fullname(t: type) -> str:
    """Return the full name of a type.

    Prefixed with its module if not a builtin.
    """
    module = t.__module__
    if module and module != "builtins":
        return f"{module}.{t.__qualname__}"
    return t.__qualname__
