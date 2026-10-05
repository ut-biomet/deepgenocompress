"""Exceptions related to deepgenocompress library."""

from __future__ import annotations

import reprlib
from collections.abc import Mapping
from enum import StrEnum
from pprint import pformat
from typing import Any, ClassVar, TypedDict


class _Raw:
    """Wraps a pre-formatted string so pformat inserts it verbatim, unquoted."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


class DeepgenocompressError(Exception):
    """Base exception class for all deepgenocompress's errors."""

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


class DuplicatedMarkerIDsError(DeepgenocompressError):
    """Raised when provided data does not have unique marker ids."""

    class ExtraKeys(TypedDict, total=True):
        duplicated_ids: list[str]

    extra: ExtraKeys | dict[str, Any]

    def __init__(self, duplicated_ids: list[str]):
        n_duplicated = len(duplicated_ids)
        msg = (
            f"Found {n_duplicated} duplicated marker id(s). Marker ids must be unique."
        )
        super().__init__(message=msg, extra={"duplicated_ids": duplicated_ids})


def _type_fullname(t: type) -> str:
    """Return the full name of a type.

    Prefixed with its module if not a builtin.
    """
    module = t.__module__
    if module and module != "builtins":
        return f"{module}.{t.__qualname__}"
    return t.__qualname__


class _ReasonCodeEnum(StrEnum):
    """Base class for the ``ReasonCode`` enum nested in each _ReasonedError class."""

    def __repr__(self) -> str:
        """Return the string representation."""
        return self.name


class _ReasonedError(DeepgenocompressError):
    """Base class for errors that can have different "trigger reasons".

    Subclasses must define:

    - a nested ``ReasonCode`` enum (subclass of :class:`_ReasonCodeEnum`);
    - a ``_MESSAGES`` mapping from each reason to a ``str.format`` template.

    Subclasses may override :meth:`_derive_format_args` to compute values that
    appear in message templates but are not stored in ``extra`` as-is.
    The method should takes 2 arguments ``reason`` and ``extra`` and return
    a dict containing extra keys needed for formating the error message.

    Completeness of ``_MESSAGES`` is checked at class-definition time, so a missing
    message fails on import instead of when the error is raised.

    NOTE: While technically not needed `reason`, and `extra` should be specified
    when defining the subclass for them to be documented.

    Subclass Snippet:

    ```
    class MyError(_ReasonedError):
        # add MyError docstring here
        reason: ReasonCode
        extra: ExtraKeys | dict[str, Any]

        class ReasonCode(_ReasonCodeEnum):
            REASON_1 = auto()
            # add REASON_1 docstring here
            REASON_2 = auto()
            # add REASON_2 docstring here

        _MESSAGES: ClassVar[Mapping[ReasonCode, str]] = {
            ReasonCode.REASON_1: "err msg reason 1 {key_1}",
            ReasonCode.REASON_2: "err msg reason 1 {key_2_derived}",
        }

        class ExtraKeys(_ReasonedError.ExtraKeys, TypedDict, total=False):

            reason: MyError.ReasonCode
            # Which rule was violated.
            key_1: str
            # add key_1 docstring here
            key_2: type
            # add key_1 docstring here

        @classmethod
        def _derive_format_args(
            cls, reason: Any, extra: Mapping[str, Any]
        ) -> dict[str, Any]:
            r = cls.ReasonCode
            if reason is r.REASON_2:
                return {"key_2_derived": _type_fullname(extra["key_2"])}
            return {}
    ```
    """

    reason: ReasonCode
    """Which rule was violated."""
    extra: ExtraKeys | dict[str, Any]
    """Contextual values. See :attr:`ExtraKeys` for the list of availables keys."""

    class ReasonCode(_ReasonCodeEnum):
        """Possible invalid reasons."""

        pass

    _MESSAGES: ClassVar[Mapping[Any, str]]

    class ExtraKeys(TypedDict, total=False):
        """Names of the keys that may appear in :attr:`extra <_ReasonedError.extra>`.

        Exact availables keys depends on the error reason.
        """

        reason: _ReasonedError.ReasonCode
        """Which rule was violated."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        reason_enum = cls.__dict__.get("ReasonCode")
        messages = cls.__dict__.get("_MESSAGES")
        if reason_enum is None and messages is None:
            return  # intermediate abstract subclass, nothing to validate
        if reason_enum is None or messages is None:
            raise TypeError(
                f"{cls.__name__} must define both `ReasonCode` and `_MESSAGES`."
            )
        if missing := set(reason_enum) - set(messages):
            raise TypeError(
                f"{cls.__name__}._MESSAGES is missing: {sorted(map(repr, missing))}"
            )
        if unknown := set(messages) - set(reason_enum):
            unknown_reasons = sorted(map(repr, unknown))
            raise TypeError(
                f"{cls.__name__}._MESSAGES has unknown reasons: {unknown_reasons}"
            )

        # Make the subclass uses documentation defined here
        extra_keys = cls.__dict__.get("ExtraKeys")
        if extra_keys is not None and not extra_keys.__doc__:
            correct_ref = f"{cls.__name__}.extra"
            if _ReasonedError.ExtraKeys.__doc__:
                # fix reference to `extra`
                extra_keys.__doc__ = _ReasonedError.ExtraKeys.__doc__.replace(
                    "_ReasonedError.extra", correct_ref
                )

        if reason_enum is not None and not reason_enum.__doc__:
            reason_enum.__doc__ = _ReasonedError.ReasonCode.__doc__

    def __init__(self, reason: Any, extra: Mapping[str, Any] | None = None) -> None:
        self.reason = reason
        extra = dict(extra or {})
        derived = {
            k: v
            for k, v in self._derive_format_args(reason, extra).items()
            if v is not None
        }
        try:
            message = self._MESSAGES[reason].format(**{**extra, **derived})
        except KeyError as exc:
            self_name = type(self).__name__
            raise ValueError(
                f"{self_name}({reason!r}) is missing required extra or derived key"
                f" {exc}."
            ) from exc
        super().__init__(message=message, extra={"reason": reason, **extra})

    @classmethod
    def _derive_format_args(
        cls, reason: Any, extra: Mapping[str, Any]  # noqa: ARG003
    ) -> dict[str, Any]:
        """Return additional msg formating arguments.

        This method should takes 2 arguments ``reason`` and ``extra`` and return
        a dict containing extra keys needed for formating the error message.

        __init__ call this method automatically before building the message.
        """
        return {}
