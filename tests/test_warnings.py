import pytest
from helpers import get_all_subclasses

import deepcgp.warnings as warnings_module
from deepcgp._base_warnings import DeepcgpWarning

custom_warnings = get_all_subclasses(DeepcgpWarning)


@pytest.mark.parametrize(
    "custom_warnings",
    list(custom_warnings.keys()),
)
def test_all_warnings_are_exported_from_warning_module(custom_warnings):
    exported_names = set(warnings_module.__all__)

    assert custom_warnings in exported_names


@pytest.mark.parametrize(
    "exported_name",
    list(warnings_module.__all__),
)
def test_module_does_not_export_stale_names(exported_name):
    """Check that __all__ references something that actually exists."""
    assert hasattr(warnings_module, exported_name), (
        f"`{exported_name}` is listed in __all__ but not actually importable "
        f"from deepcgp.warnings"
    )
