import pytest

from deepcgp._core.utils import encoding_size
from deepcgp.warnings import DeepcgpWarning


class TestEncodingSize:

    @pytest.mark.parametrize("size", [0, 1, 4, 10])
    def test_returns_length_of_encoding_vectors(self, size):
        encoding_map = {
            "A": [0 for _ in range(size)],
            "C": [0 for _ in range(size)],
        }
        assert encoding_size(encoding_map) == size

    @pytest.mark.parametrize("size", [0, 1, 4, 10])
    def test_minimal_encoding_map(self, size):
        encoding_map = {
            0: [0 for _ in range(size)],
        }
        assert encoding_size(encoding_map) == size

    def test_raises_on_empty_encoding_map(self):
        with pytest.raises(StopIteration):
            encoding_size({})
