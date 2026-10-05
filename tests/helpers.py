"""Helper function for tests."""

import importlib
import inspect
import pkgutil
from typing import Any, ClassVar

import pandas as pd

import deepgenocompress
from deepgenocompress._core.exceptions import _ReasonedError
from deepgenocompress.exceptions import DeepgenocompressError


def get_all_subclasses(parent_class) -> dict[str, type]:
    """Find every DeepgenocompressWarning subclass defined anywhere in the package."""
    subclasses = {}
    for module_info in pkgutil.walk_packages(
        deepgenocompress.__path__, prefix="deepgenocompress."
    ):
        module = importlib.import_module(module_info.name)
        for name, obj in vars(module).items():
            if not inspect.isclass(obj):
                continue
            if not issubclass(obj, parent_class):
                continue
            if obj.__module__ != module_info.name:
                continue  # only count it in the module where it's *defined*
            subclasses[name] = obj
    return subclasses


class ErrorTestBase:
    """Base class for testing custom errors with a `ReasonCode` and `_MESSAGES`.

    Subclasses (named `Test...` so pytest collects them) only define:
      - `error_class`: the error class under test
      - `test_cases`: {ReasonCode: {"expected_msg": str, "extra": dict (optional)}}
    """

    error_class: ClassVar[type[_ReasonedError]]
    test_cases: ClassVar[dict[Any, dict[str, Any]]]

    def pytest_generate_tests(self, metafunc):
        if "reason" in metafunc.fixturenames:
            reasons = list(metafunc.cls.error_class.ReasonCode)
            metafunc.parametrize("reason", reasons, ids=repr)

    def test_error_is_DeepgenocompressError(self):
        assert issubclass(self.error_class, DeepgenocompressError)

    def test_all_reason_codes_have_a_message(self):
        assert set(self.error_class._MESSAGES) == set(self.error_class.ReasonCode)

    def test_all_reason_codes_have_a_test_case(self):

        missing_cases = [
            c for c in set(self.error_class.ReasonCode) if c not in set(self.test_cases)
        ]
        assert missing_cases == [], f"missing cases: {missing_cases}"

    def test_error_messages_and_extra(self, reason):
        case = self.test_cases[reason]
        expected_msg = case["expected_msg"]
        extra = case.get("extra")

        if extra is None:
            err = self.error_class(reason=reason)
        else:
            err = self.error_class(reason=reason, extra=dict(extra))

        assert "reason" in err.extra
        assert err.extra["reason"] is reason

        for k, v in (extra or {}).items():
            assert k in err.extra
            if isinstance(v, pd.Index):
                assert v.equals(err.extra[k])
            else:
                assert v == err.extra[k]

        assert expected_msg in str(err)
