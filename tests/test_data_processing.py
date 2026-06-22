import re
from itertools import cycle

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from deepcgp.data_processing import (
    _validate_encoding_map,
    build_one_hot_encoding_map,
    encode_snp_array,
)


@pytest.fixture
def basic_geno_array():
    return np.array(
        [
            ["G", "A", "T", "T", "A", "C", "A"],
            ["A", "C", "A", "T", "T", "A", "G"],
            ["T", "A", "G", "G", "T", "C", "A"],
        ]
    )


MISSING_VALUES_CASES = [
    pytest.param(["N"], id="one missing value: N"),
    pytest.param(["N", "."], id="two missing values: N and ."),
]


@pytest.fixture(params=MISSING_VALUES_CASES)
def missing_values_fixt(request):
    return request.param


@pytest.fixture
def basic_geno_array_with_missing_values(missing_values_fixt):
    """Build geno array with missing values by cycling the given list of missing
    values."""
    """
    eg. with ``["N"]`` (default) returns (ACGT masked):
        [
            [   , "N",    ,    ,    , "N",    ],
            [   , "N",    ,    ,    ,    ,    ],
            [   ,    ,    ,    , "N",    ,    ],
        ]
    eg. with ``["N", "."]`` returns (ACGT masked):
        [
            [   , "N",    ,    ,    , ".",    ],
            [   , "N",    ,    ,    ,    ,    ],
            [   ,    ,    ,    , ".",    ,    ],
        ]
    """
    assert isinstance(missing_values_fixt, list)
    m = cycle(missing_values_fixt).__next__
    return np.array(
        [
            ["G", m(), "T", "T", "A", m(), "A"],
            ["A", m(), "A", "T", "T", "A", "G"],
            ["T", "A", "G", "G", m(), "C", "A"],
        ]
    )


class TestValidateEncodingMap:
    """Tests for _validate_encoding_map functions."""

    def test_raise_if_not_dict(self):
        encoding_map = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
        with pytest.raises(TypeError, match="must be a dict"):
            _validate_encoding_map(encoding_map)  # pyright: ignore [reportArgumentType]

    def test_raise_if_encodings_are_not_list(self):
        encoding_map = {
            "A": (0, 0, 1),
            "B": (0, 1, 0),
            "C": (1, 0, 0),
        }
        with pytest.raises(ValueError, match="must be a list"):
            _validate_encoding_map(encoding_map)  # pyright: ignore [reportArgumentType]

    def test_raise_if_encodings_have_different_lengths(self):
        encoding_map = {
            "A": [1],
            "B": [0, 1],
            "C": [0, 0, 1],
        }
        with pytest.raises(ValueError, match="must have the same length"):
            _validate_encoding_map(encoding_map)

    def test_raise_if_encodings_are_not_int_or_float(self):
        encoding_map = {
            "A": ["0", "0", "1"],
            "B": ["0", "1", "0"],
            "C": ["1", "0", "0"],
        }
        with pytest.raises(ValueError, match="must be a list of numerical values"):
            _validate_encoding_map(encoding_map)  # pyright: ignore [reportArgumentType]

    def test_return_none_when_valid(self):
        encoding_map = {
            "A": [1, 0, 0],
            "B": [0, 1, 0],
            "C": [0, 0, 1],
        }
        assert _validate_encoding_map(encoding_map) is None

    def test_pass_with_float_values(self):
        encoding_map = {
            "A": [1.0, 0, 0],
            "B": [0, 1.0, 0],
            "C": [0, 0, 1.0],
        }
        _validate_encoding_map(encoding_map)

    def test_pass_with_non_str_keys(self):
        encoding_map = {
            0: [1, 0, 0],
            1: [0, 1, 0],
            2: [0, 0, 1],
        }
        _validate_encoding_map(encoding_map)

    def test_pass_with_duplicate_encodings(self):
        """To accept "synonyms"."""
        encoding_map = {
            "A": [1, 0, 0],
            "B": [0, 1, 0],  # <- same
            "C": [0, 1, 0],  # <- same
        }
        _validate_encoding_map(encoding_map)

    @pytest.mark.parametrize(
        "missing_vector",
        [
            ([0, 0, 1]),
            ([-1, 0, 1]),
        ],
        ids=["one_hot", "sum equal to 0"],
    )
    def test_raise_when_missing_values_not_a_0_vector(self, missing_vector):
        encoding_map = {
            "A": [1, 0, 0],
            "B": [0, 1, 0],
            ".": missing_vector,
        }

        expected_msg = (
            "Missing values not encoded with a vector of 0, for '.' "
            f"got {missing_vector}."
        )
        with pytest.raises(
            ValueError,
            match=re.escape(expected_msg),
        ):
            _validate_encoding_map(encoding_map, missing_values=["."])

    def test_pass_when_missing_values_is_a_0_vector(self):
        encoding_map = {
            "A": [1, 0, 0],
            "B": [0, 1, 0],
            ".": [0, 0, 0],
        }

        assert _validate_encoding_map(encoding_map, missing_values=["."]) is None

    def test_raise_if_encoding_map_is_empty(self):
        encoding_map = {}
        with pytest.raises(
            ValueError,
            match=re.escape("`encoding_map` is empty."),
        ):
            _validate_encoding_map(encoding_map)


class TestBuildOneHotEncodingMap:
    """Tests for build_one_hot_encoding_map functions."""

    def test_returns_dict(self, basic_geno_array):
        result = build_one_hot_encoding_map(basic_geno_array)
        assert isinstance(result, dict)

    def test_correct_keys(self, basic_geno_array):
        result = build_one_hot_encoding_map(basic_geno_array)
        assert set(result.keys()) == {"A", "C", "G", "T"}

    def test_encoding_length_matches_number_of_alleles(self, basic_geno_array):
        result = build_one_hot_encoding_map(basic_geno_array)
        n = len(result)
        assert all(len(encoding) == n for encoding in result.values())

    def test_encoding_is_one_hot(self, basic_geno_array):
        result = build_one_hot_encoding_map(basic_geno_array)
        for encoding in result.values():
            assert sum(encoding) == 1
            assert set(encoding) == {0, 1}

    def test_encodings_are_unique(self, basic_geno_array):
        result = build_one_hot_encoding_map(basic_geno_array)
        encodings = [tuple(encoding) for encoding in result.values()]
        assert len(encodings) == len(set(encodings))

    def test_with_single_allele(self):
        result = build_one_hot_encoding_map(np.array([["A", "A", "A"]]))
        assert result == {"A": [1]}

    def test_deterministic_regardless_of_order(self):
        array1 = np.array([["A", "C", "G", "T"]])
        array2 = np.array([["T", "G", "C", "A"]])
        assert build_one_hot_encoding_map(array1) == build_one_hot_encoding_map(array2)

    def test_pass_validate_encoding_map(self, basic_geno_array):
        result = build_one_hot_encoding_map(basic_geno_array)
        assert _validate_encoding_map(result) is None

    @pytest.mark.parametrize(
        "missing_values_fixt",
        [["N"]],
        ids=["one missing value: N"],
    )
    def test_default_exclusion(
        self, missing_values_fixt, basic_geno_array_with_missing_values
    ):
        result = build_one_hot_encoding_map(basic_geno_array_with_missing_values)
        assert _validate_encoding_map(result) is None
        assert all(m not in result for m in missing_values_fixt)
        assert result == {
            "A": [1, 0, 0, 0],
            "C": [0, 1, 0, 0],
            "G": [0, 0, 1, 0],
            "T": [0, 0, 0, 1],
        }

    @pytest.mark.parametrize(
        "missing_values_fixt",
        [["X"], ["-", ".", "NA"]],
        ids=["one missing value: X", "three missing values: -, . and NA"],
    )
    def test_exclusion_of_missing_values(
        self, missing_values_fixt, basic_geno_array_with_missing_values
    ):
        result = build_one_hot_encoding_map(
            basic_geno_array_with_missing_values, exclude=missing_values_fixt
        )
        assert _validate_encoding_map(result) is None
        assert all(m not in result for m in missing_values_fixt)
        assert result == {
            "A": [1, 0, 0, 0],
            "C": [0, 1, 0, 0],
            "G": [0, 0, 1, 0],
            "T": [0, 0, 0, 1],
        }

    def test_empty_array(self):
        empty_array = np.empty((0, 0))
        result = build_one_hot_encoding_map(empty_array)
        assert result == {}

    def test_with_empty_string(self):
        geno_array = np.array(
            [
                ["A", "C", "T"],
                ["G", "", "A"],
            ]
        )
        result = build_one_hot_encoding_map(geno_array)
        assert result == {
            "": [1, 0, 0, 0, 0],
            "A": [0, 1, 0, 0, 0],
            "C": [0, 0, 1, 0, 0],
            "G": [0, 0, 0, 1, 0],
            "T": [0, 0, 0, 0, 1],
        }

    def test_numeric_array(self):
        geno_array = np.array(
            [
                [2, 1, 2],
                [0, 0, 2],
            ]
        )
        result = build_one_hot_encoding_map(geno_array)
        assert result == {
            0: [1, 0, 0],
            1: [0, 1, 0],
            2: [0, 0, 1],
        }

    @pytest.mark.parametrize(
        "miss_val",
        [None, np.nan, pd.NA],
        ids=["None", "np.nan", "pd.NA"],
    )
    def test_with_special_missing_values(self, miss_val):
        """Those are always excluded even if not specifyed in excluded."""
        # `geno_array` must be numeric else np.nan is converted to the string "nan"
        geno_array = np.array(
            [
                [2.0, 1.0, 2.0],
                [0.0, miss_val, 2.0],
            ]
        )
        result = build_one_hot_encoding_map(geno_array, exclude=[])
        assert result == {
            0: [1, 0, 0],
            1: [0, 1, 0],
            2: [0, 0, 1],
        }

    @pytest.mark.parametrize(
        "miss_val",
        [None, np.nan, pd.NA],
        ids=["None", "np.nan", "pd.NA"],
    )
    def test_special_missing_values_can_be_given_in_exclude(self, miss_val):
        """Even if not necessary check it doesn't crash."""
        geno_array_str = np.array([["A", "C", "G", "T"]])
        result_str = build_one_hot_encoding_map(geno_array_str, exclude=[miss_val])
        assert result_str == {
            "A": [1, 0, 0, 0],
            "C": [0, 1, 0, 0],
            "G": [0, 0, 1, 0],
            "T": [0, 0, 0, 1],
        }

        geno_array_num = np.array([[0, 1, 2]])
        result_num = build_one_hot_encoding_map(geno_array_num, exclude=[miss_val])
        assert result_num == {
            0: [1, 0, 0],
            1: [0, 1, 0],
            2: [0, 0, 1],
        }

    def test_with_mixed_types(self):
        geno_array = np.array(
            [
                ["G", np.nan, 0, 0],
                [1, None, "A", pd.NA],
            ]
        )
        result = build_one_hot_encoding_map(geno_array)
        assert result == {
            0: [1, 0, 0, 0],
            1: [0, 1, 0, 0],
            "A": [0, 0, 1, 0],
            "G": [0, 0, 0, 1],
        }


class TestEncodeSnpArray:
    """Tests for encode_snp_array functions."""

    @pytest.fixture
    def basic_encoding_map(self):
        enc_map = {
            "A": [1, 0, 0, 0],
            "C": [0, 1, 0, 0],
            "G": [0, 0, 1, 0],
            "T": [0, 0, 0, 1],
        }
        _validate_encoding_map(enc_map)
        return enc_map

    def test_call_build_one_hot_encoding_with_default_params(
        self, basic_geno_array, mocker: MockerFixture
    ):
        """Test build_one_hot_encoding_map is called with default parameters."""
        mock_build_one_hot_encoding_map = mocker.patch(
            "deepcgp.data_processing.build_one_hot_encoding_map",
            wraps=build_one_hot_encoding_map,
        )
        encode_snp_array(basic_geno_array)
        mock_build_one_hot_encoding_map.assert_called_once_with(
            basic_geno_array,
            {"N"},  # default for `missing_values`
        )

    @pytest.mark.parametrize(
        "missing_values_fixt",
        [["X"], ["-", ".", "NA"]],
        ids=["one missing value: X", "three missing values: -, . and NA"],
    )
    def test_call_build_one_hot_encoding_map_with_provided_missing_values(
        self, missing_values_fixt, basic_geno_array, mocker: MockerFixture
    ):
        """Test build_one_hot_encoding_map is called if encoding_map is None."""
        mock_build_one_hot_encoding_map = mocker.patch(
            "deepcgp.data_processing.build_one_hot_encoding_map",
            wraps=build_one_hot_encoding_map,
        )
        encode_snp_array(
            basic_geno_array, missing_values=missing_values_fixt, encoding_map=None
        )
        mock_build_one_hot_encoding_map.assert_called_once_with(
            basic_geno_array,
            missing_values_fixt,
        )

    @pytest.mark.parametrize(
        "missing_values_fixt",
        [["X"], ["-", ".", "NA"]],
        ids=["one missing value: X", "three missing values: -, . and NA"],
    )
    def test_call_validate_encoding_map(
        self,
        missing_values_fixt,
        basic_geno_array,
        basic_encoding_map,
        mocker: MockerFixture,
    ):
        """Test _validate_encoding_map is called encoding_map is provided."""
        mock_validate = mocker.patch(
            "deepcgp.data_processing._validate_encoding_map",
            wraps=_validate_encoding_map,
        )
        encode_snp_array(
            basic_geno_array,
            missing_values=missing_values_fixt,
            encoding_map=basic_encoding_map,
        )
        mock_validate.assert_called_once_with(basic_encoding_map, missing_values_fixt)

    def test_use_correct_encoding(self, basic_encoding_map):
        geno_a = np.array([["A"]])
        result_a = encode_snp_array(geno_a, encoding_map=basic_encoding_map)
        np.testing.assert_array_equal(result_a, np.array([basic_encoding_map["A"]]))

        geno_c = np.array([["C"]])
        result_c = encode_snp_array(geno_c, encoding_map=basic_encoding_map)
        np.testing.assert_array_equal(result_c, np.array([basic_encoding_map["C"]]))

        geno_g = np.array([["G"]])
        result_g = encode_snp_array(geno_g, encoding_map=basic_encoding_map)
        np.testing.assert_array_equal(result_g, np.array([basic_encoding_map["G"]]))

        geno_t = np.array([["T"]])
        result_t = encode_snp_array(geno_t, encoding_map=basic_encoding_map)
        np.testing.assert_array_equal(result_t, np.array([basic_encoding_map["T"]]))

        geno_unkown = np.array([["N"]])
        with pytest.warns(
            UserWarning,
            match=(
                "The encoded array contains only zeros. This may indicate that all values "
                "in geno_array are missing or not present in encoding_map."
            ),
        ):
            result = encode_snp_array(geno_unkown, encoding_map=basic_encoding_map)
        np.testing.assert_array_equal(
            result, np.array([[0] * len(basic_encoding_map["A"])])
        )

    def test_missing_allele_are_encoded_with_vector_of_0(self, basic_encoding_map):
        """Missing allele as empty string."""
        encoding_values_len = len(basic_encoding_map["A"])
        geno_unkown = np.array([[""]])

        with pytest.warns(
            UserWarning,
            match=(
                "The encoded array contains only zeros. This may indicate that all values "
                "in geno_array are missing or not present in encoding_map."
            ),
        ):
            result = encode_snp_array(geno_unkown, encoding_map=basic_encoding_map)
        np.testing.assert_array_equal(result, np.array([[0] * encoding_values_len]))

    def test_use_correct_dtype(self, basic_geno_array, basic_encoding_map):
        result = encode_snp_array(basic_geno_array, encoding_map=basic_encoding_map)
        assert result.dtype == np.float32

    def test_with_empty_array(self, basic_encoding_map):
        geno_empty = np.array([[]])

        with pytest.warns(
            UserWarning,
            match=(
                "The encoded array contains only zeros. This may indicate that all values "
                "in geno_array are missing or not present in encoding_map."
            ),
        ):
            result = encode_snp_array(geno_empty, encoding_map=basic_encoding_map)
        np.testing.assert_array_equal(result, np.array([[]]))

    def test_with_single_row(self, basic_encoding_map):
        em = basic_encoding_map
        geno = np.array([["G", "A", "T"]])
        result = encode_snp_array(geno, encoding_map=basic_encoding_map)

        expected = np.array([em["G"] + em["A"] + em["T"]])
        np.testing.assert_array_equal(result, expected)

    def test_with_single_column(self, basic_encoding_map):
        em = basic_encoding_map
        geno = np.array([["G"], ["A"], ["T"]])
        expected = np.array([em["G"], em["A"], em["T"]])
        result = encode_snp_array(geno, encoding_map=basic_encoding_map)

        np.testing.assert_array_equal(result, expected)

    def test_with_multiple_rows_columns(self, basic_encoding_map):
        em = basic_encoding_map
        em["N"] = [0, 0, 0, 0]
        geno = np.array(
            [
                ["G", "A", "N", "T"],  # <- "N" explicitly in encoding map as 0s
                ["A", ".", "A", "A"],  # <- "." not in encoding map
            ]
        )
        expected = np.array(
            [
                em["G"] + em["A"] + [0] * 4 + em["T"],
                em["A"] + [0] * 4 + em["A"] + em["A"],
            ]
        )
        result = encode_snp_array(geno, encoding_map=basic_encoding_map)
        np.testing.assert_array_equal(result, expected)

    def test_with_numeric_geno_array(self):
        geno = np.array(
            [
                [0, 1, 2],
                [1, 2, -1],
            ]
        )
        em = build_one_hot_encoding_map(geno, exclude={-1})
        result = encode_snp_array(geno, missing_values={-1})
        expected = np.array(
            [
                em[0] + em[1] + em[2],
                em[1] + em[2] + [0] * 3,
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_with_nan(self):
        geno = np.array(
            [
                [0, 1, 2],
                [1, np.nan, 1],
            ]
        )
        em = build_one_hot_encoding_map(geno)
        result = encode_snp_array(geno)
        expected = np.array(
            [
                em[0] + em[1] + em[2],
                em[1] + [0] * 3 + em[1],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_with_None(self):
        geno = np.array(
            [
                [0, 1, 2],
                [1, None, 1],
            ]
        )
        em = build_one_hot_encoding_map(geno)
        result = encode_snp_array(geno)
        expected = np.array(
            [
                em[0] + em[1] + em[2],
                em[1] + [0] * 3 + em[1],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_with_mixed_types(self):
        geno = np.array(
            [
                ["G", np.nan, 0],
                [1, None, "A"],
            ]
        )
        em = build_one_hot_encoding_map(geno)
        result = encode_snp_array(geno)
        expected = np.array(
            [
                em["G"] + [0] * 4 + em[0],
                em[1] + [0] * 4 + em["A"],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_warns_with_only_missing_values(self):
        geno = np.array(
            [
                ["G", "A", "T"],
                ["A", "C", "A"],
            ]
        )
        encoding_map = {"Z": [1, 0], "Y": [0, 1]}

        with pytest.warns(
            UserWarning,
            match=(
                "The encoded array contains only zeros. This may indicate that all values "
                "in geno_array are missing or not present in encoding_map."
            ),
        ):
            result = encode_snp_array(geno, encoding_map=encoding_map)
        expected = np.array(
            [
                [0] * 2 * 3,
                [0] * 2 * 3,
            ]
        )
        np.testing.assert_array_equal(result, expected)


def test_public_api_exports():
    import deepcgp

    assert hasattr(deepcgp, "encode_snp_array")
    assert hasattr(deepcgp, "build_one_hot_encoding_map")
