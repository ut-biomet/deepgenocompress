import pytest
from helpers import get_all_subclasses

import deepgenocompress.exceptions as exceptions_module
from deepgenocompress._core.exceptions import DeepgenocompressError

custom_errors = get_all_subclasses(DeepgenocompressError)
public_custom_errors = [name for name in custom_errors if not name.startswith("_")]


@pytest.mark.parametrize(
    "public_custom_errors",
    public_custom_errors,
)
def test_all_exceptions_are_exported_from_exceptions_module(public_custom_errors):
    exported_names = set(exceptions_module.__all__)

    assert public_custom_errors in exported_names


@pytest.mark.parametrize(
    "exported_name",
    list(exceptions_module.__all__),
)
def test_module_does_not_export_stale_names(exported_name):
    """Check that __all__ references something that actually exists."""
    assert hasattr(exceptions_module, exported_name), (
        f"`{exported_name}` is listed in __all__ but not actually importable "
        f"from deepgenocompress.exceptions"
    )
