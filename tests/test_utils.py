import pytest

from deepgenocompress.exceptions import UnexpectedMarkerIdFormatError
from deepgenocompress.utils import MARKER_ID_FORMATS, build_marker_ids, encoding_size


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
        with pytest.raises(ValueError):
            encoding_size({})


@pytest.mark.parametrize("format", MARKER_ID_FORMATS)
def test_all_MARKER_ID_FORMATS_works_with_build_marker_ids(format):
    # all MARKER_ID_FORMATS should be expected
    marker_info = {
        "chrom": "chr1",
        "pos": 1234,
        "id": "marker_1",
        "ref": "T",
        "alt": ["C", "A"],
    }
    try:
        build_marker_ids(**marker_info, marker_id_format=format)
    except UnexpectedMarkerIdFormatError as e:
        pytest.fail(f"Unexpected UnexpectedMarkerIdFormatError raised: {e}")
