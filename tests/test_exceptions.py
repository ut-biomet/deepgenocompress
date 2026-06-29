import importlib
import inspect
import pkgutil

import pytest

import deepcgp
import deepcgp.exceptions as exceptions_module
from deepcgp._base_exceptions import DeepcgpError


def _get_all_deepcgp_error_subclasses() -> dict[str, type]:
    """Find every DeepcgpError subclass defined anywhere in the deepcgp package."""
    custom_errors = {}
    for module_info in pkgutil.walk_packages(deepcgp.__path__, prefix="deepcgp."):
        module = importlib.import_module(module_info.name)
        for name, obj in vars(module).items():
            if not inspect.isclass(obj):
                continue
            if not issubclass(obj, DeepcgpError):
                continue
            if obj.__module__ != module_info.name:
                continue  # only count it in the module where it's *defined*
            custom_errors[name] = obj
    return custom_errors


custom_errors = _get_all_deepcgp_error_subclasses()


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
