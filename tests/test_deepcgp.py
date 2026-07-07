import pytest

import deepcgp


@pytest.mark.parametrize(
    "exported_name",
    list(deepcgp.__all__),
)
def test_module_does_not_export_stale_names(exported_name):
    """Check that __all__ references something that actually exists."""
    assert hasattr(deepcgp, exported_name), (
        f"`{exported_name}` is listed in __all__ but not actually importable "
        "from deepcgp."
    )
