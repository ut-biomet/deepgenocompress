"""Helper function for tests."""

import importlib
import inspect
import pkgutil

import deepcgp


def get_all_subclasses(parent_class) -> dict[str, type]:
    """Find every DeepcgpWarning subclass defined anywhere in the deepcgp package."""
    subclasses = {}
    for module_info in pkgutil.walk_packages(deepcgp.__path__, prefix="deepcgp."):
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
