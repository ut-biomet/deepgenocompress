import pytest
from helpers import get_all_subclasses

import deepcgp.exceptions as exceptions_module
from deepcgp._core.exceptions import DeepcgpError

custom_errors = get_all_subclasses(DeepcgpError)


@pytest.mark.parametrize(
    "custom_error",
    list(custom_errors.keys()),
)
def test_all_exceptions_are_exported_from_exceptions_module(custom_error):
    exported_names = set(exceptions_module.__all__)

    assert custom_error in exported_names


@pytest.mark.parametrize(
    "exported_name",
    list(exceptions_module.__all__),
)
def test_module_does_not_export_stale_names(exported_name):
    """Check that __all__ references something that actually exists."""
    assert hasattr(exceptions_module, exported_name), (
        f"`{exported_name}` is listed in __all__ but not actually importable "
        f"from deepcgp.exceptions"
    )
