"""Warnings related to deepcgp library."""

import os
import warnings
from pprint import pformat
from typing import Any, overload

_LIBRARY_DIR = os.path.dirname(os.path.abspath(__file__))


class DeepcgpWarning(UserWarning):
    """Base warning class for all deepcgp's warnings."""

    message: str
    """Warning message."""

    extra: dict[str, Any]
    """Warning's extra information."""

    def __init__(
        self,
        message: str,
        extra: dict[str, Any] | None = None,
    ):
        self.message = message
        self.extra = extra or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return a string representation of the warning."""
        if self.extra:
            return f"{self.message}\nExtra information:\n{pformat(self.extra)}"
        return self.message

    pass


@overload
def _deepcgp_warn(
    message: str,
    category: type[Warning] | None = ...,
) -> None: ...
@overload
def _deepcgp_warn(
    message: Warning,
    category: Any = ...,
) -> None: ...
def _deepcgp_warn(message, category=None) -> None:
    """Issue a DeepcgpWarning warning."""
    if category is None:
        category = DeepcgpWarning
    warnings.warn(  # noqa: TID251
        message=message, category=category, skip_file_prefixes=(_LIBRARY_DIR,)
    )
