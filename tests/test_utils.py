from pathlib import Path

import pandas as pd
import pytest

from deepgenocompress import read_vcf
from deepgenocompress.exceptions import UnexpectedMarkerIdFormatError
from deepgenocompress.utils import (
    MARKER_ID_FORMATS,
    ExampleFiles,
    build_marker_ids,
    encoding_size,
)


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


example_names = [name for name in dir(ExampleFiles()) if not name.startswith("_")]


class TestExampleFiles:

    @pytest.mark.parametrize("example_name", example_names)
    def test_example_files_exists(self, example_name):
        exemple_file: Path = getattr(ExampleFiles(), example_name)
        assert exemple_file.exists(), f"{example_name}: {exemple_file} doesn't exists"
        assert exemple_file.is_file(), f"{example_name}: {exemple_file} is not a file"
        assert (
            exemple_file.stat().st_size > 0
        ), f"{example_name}: {exemple_file} is empty"

    @pytest.mark.parametrize("example_name", example_names)
    def test_example_files_can_be_loaded(self, example_name):
        exemple_file: Path = getattr(ExampleFiles(), example_name)
        if exemple_file.suffix == ".csv":
            data = pd.read_csv(exemple_file)

        if exemple_file.suffix == ".vcf":
            data = read_vcf(exemple_file)

        assert not data.empty

    def test_print(self):
        assert all(ex_name in str(ExampleFiles()) for ex_name in example_names)
        exemple_files = [
            str(getattr(ExampleFiles(), ex_name)) for ex_name in example_names
        ]
        assert all(f in str(ExampleFiles()) for f in exemple_files)
