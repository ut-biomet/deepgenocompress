import re
from pathlib import Path
from typing import get_args

import numpy as np
import pandas as pd
import pytest
from cyvcf2 import VCF
from pytest_lazy_fixtures import lf
from pytest_mock import MockerFixture

from deepgenocompress._core.data_processing import _validate_encoding_map
from deepgenocompress._core.vcf import (
    _MARKER_ID_FORMAT,
    DuplicatedMarkerIDsError,
    InvalidVcfDataError,
    UnexpectedMarkerIdFormatError,
    build_marker_ids,
    build_vcf_encoding_map,
    read_vcf,
    reindex_vcf_data,
    validate_vcf_data,
)
from deepgenocompress.exceptions import DeepgenocompressError


@pytest.fixture(params=get_args(_MARKER_ID_FORMAT))
def marker_id_format(request):
    return request.param


@pytest.fixture(
    params=[
        "basic_vcf_data",
        "empty_vcf_data",
        "no_rows_with_columns_vcf_data",
        "no_columns_with_rows_vcf_data",
        "single_row_vcf_data",
        "single_col_vcf_data",
        "single_cell_vcf_data",
    ]
)
def valid_vcf_data_with_edge_cases(request):
    markers_info = {
        "marker1": {
            "chrom": "1",
            "pos": 100,
            "id": "marker1",
            "ref": "A",
            "alt": ["T"],
        },
        "marker2": {
            "chrom": "1",
            "pos": 200,
            "id": "marker2",
            "ref": "G",
            "alt": ["C"],
        },
        "marker3": {
            "chrom": "3",
            "pos": 100,
            "id": "marker3",
            "ref": "T",
            "alt": ["C"],
        },
    }

    cases = {
        "basic_vcf_data": (
            {"marker1": ["A/A"], "marker2": ["G/G"], "marker3": ["C/T"]},
            ["sample1", "sample2"],
            ["marker1", "marker2", "marker3"],
        ),
        "empty_vcf_data": ({}, [], []),
        "no_rows_with_columns_vcf_data": (
            {"marker1": [], "marker2": [], "marker3": []},
            [],
            ["marker1", "marker2", "marker3"],
        ),
        "no_columns_with_rows_vcf_data": ({}, ["sample1", "sample2"], []),
        "single_row_vcf_data": (
            {"marker1": ["A/A"], "marker2": ["G/G"], "marker3": ["C/T"]},
            ["sample1"],
            ["marker1", "marker2", "marker3"],
        ),
        "single_col_vcf_data": (
            {"marker1": ["A/A", "A/T"]},
            ["sample1", "sample2"],
            ["marker1"],
        ),
        "single_cell_vcf_data": ({"marker1": ["A/A"]}, ["sample1"], ["marker1"]),
    }

    data, index, markers = cases[request.param]

    vcf_data = pd.DataFrame(data, index=index)
    vcf_data.attrs = {
        "use_bases": True,
        "markers_info": {m: markers_info[m] for m in markers},
    }
    return vcf_data


def _basic_valid_vcf_data(use_bases=True):
    if use_bases:
        vcf_data = pd.DataFrame(
            {
                "marker1": ["A/A", "A/T"],
                "marker2": ["G/G", "G/C"],
                "marker3": ["C/T", "T/C"],
            },
            index=["sample1", "sample2"],
        )
    else:
        vcf_data = pd.DataFrame(
            {
                "marker1": [2, 1],
                "marker2": [0, 1],
                "marker3": [1, 1],
            },
            index=["sample1", "sample2"],
        )

    vcf_data.attrs = {
        "use_bases": use_bases,
        "markers_info": {
            "marker1": {
                "chrom": "1",
                "pos": 100,
                "id": "marker1",
                "ref": "T",
                "alt": ["A"],
            },
            "marker3": {
                "chrom": "3",
                "pos": 100,
                "id": "marker3",
                "ref": "T",
                "alt": ["C"],
            },
            "marker2": {
                "chrom": "1",
                "pos": 200,
                "id": "marker2",
                "ref": "G",
                "alt": ["C"],
            },
        },
    }
    return vcf_data


@pytest.fixture
def basic_valid_vcf_data_bases():
    return _basic_valid_vcf_data()


@pytest.fixture
def basic_valid_vcf_data_no_bases():
    return _basic_valid_vcf_data(use_bases=False)


# list of invalid vcf_data:
# tuple of:
#   - case id,
#   - reason,
#   - extra information,
#   - expected message,
#   - invalid vcf_data
invalidVcfDataError_errors_cases = (
    (
        "not_a_dataframe",
        InvalidVcfDataError.ReasonCode.NOT_A_DATAFRAME,
        {"provided_type": type(None)},
        "`vcf_data` must be a pandas DataFrame, got 'NoneType'.",
        None,
    ),
    (
        "use_bases_attr_missing",
        InvalidVcfDataError.ReasonCode.USE_BASES_ATTR_MISSING,
        {},
        "attrs['use_bases'] is missing.",
        (lambda df: (df.attrs.pop("use_bases", None), df)[1])(_basic_valid_vcf_data()),
    ),
    (
        "use_bases_invalid_type",
        InvalidVcfDataError.ReasonCode.USE_BASES_INVALID_TYPE,
        {"provided_type": int},
        "attrs['use_bases'] must be a bool, got 'int'.",
        (lambda df: (df.attrs.update(use_bases=1), df)[1])(_basic_valid_vcf_data()),
    ),
    (
        "values_inconsistent_with_use_bases_true_1",
        InvalidVcfDataError.ReasonCode.VALUES_INCONSISTENT_WITH_USE_BASES,
        {"use_bases": True, "invalid_values": {np.int64(0), np.int64(1), np.int64(2)}},
        (
            "vcf_data contains value(s) inconsistent with use_bases=True: "
            "[np.int64(0), np.int64(1), np.int64(2)]."
        ),
        (lambda df: (df.attrs.update(use_bases=True), df)[1])(
            _basic_valid_vcf_data(use_bases=False)
        ),
    ),
    (
        # case where alleles are not ACTGN with /
        "values_inconsistent_with_use_bases_true_2",
        InvalidVcfDataError.ReasonCode.VALUES_INCONSISTENT_WITH_USE_BASES,
        {
            "use_bases": True,
            "invalid_values": {"X/A"},
        },
        "vcf_data contains value(s) inconsistent with use_bases=True: ['X/A'].",
        _basic_valid_vcf_data().assign(marker1=["X/A", "A/T"]),
    ),
    (
        # case where alleles are not ACTGN with |
        "values_inconsistent_with_use_bases_true_3",
        InvalidVcfDataError.ReasonCode.VALUES_INCONSISTENT_WITH_USE_BASES,
        {
            "use_bases": True,
            "invalid_values": {"A|Y"},
        },
        "vcf_data contains value(s) inconsistent with use_bases=True: ['A|Y'].",
        _basic_valid_vcf_data().assign(marker1=["A/A", "A|Y"]),
    ),
    (
        "values_inconsistent_with_use_bases_false",
        InvalidVcfDataError.ReasonCode.VALUES_INCONSISTENT_WITH_USE_BASES,
        {
            "use_bases": False,
            "invalid_values": {"G/G", "A/A", "A/T", "G/C", "C/T", "T/C"},
        },
        (
            "vcf_data contains value(s) inconsistent with use_bases=False: "
            "['A/A', 'A/T', 'C/T', 'G/C', 'G/G', 'T/C']."  # sorted for the message
        ),
        (lambda df: (df.attrs.update(use_bases=False), df)[1])(
            _basic_valid_vcf_data(use_bases=True)
        ),
    ),
    (
        "markers_info_attr_missing",
        InvalidVcfDataError.ReasonCode.MARKERS_INFO_ATTR_MISSING,
        {},
        "attrs['markers_info'] is missing.",
        (lambda df: (df.attrs.pop("markers_info", None), df)[1])(
            _basic_valid_vcf_data()
        ),
    ),
    (
        "markers_info_invalid_type",
        InvalidVcfDataError.ReasonCode.MARKERS_INFO_INVALID_TYPE,
        {"provided_type": str},
        "attrs['markers_info'] must be a dict, got 'str'.",
        (lambda df: (df.attrs.update(markers_info="info"), df)[1])(
            _basic_valid_vcf_data()
        ),
    ),
    (
        "markers_info_missing_columns",
        InvalidVcfDataError.ReasonCode.MARKERS_INFO_MISSING_COLUMNS,
        {"missing_columns": ["snp_extra_1", "snp_extra_2"]},
        "attrs['markers_info'] missing information for column(s): "
        "['snp_extra_1', 'snp_extra_2'].",
        _basic_valid_vcf_data().assign(
            snp_extra_1=["A/A", "A/A"], snp_extra_2=["A/A", "A/A"]
        ),
    ),
    (
        "marker_info_invalid_type",
        InvalidVcfDataError.ReasonCode.MARKER_INFO_INVALID_TYPE,
        {"marker_ids": ["marker1"]},
        "markers_info entries must be dict, invalid for 1 marker(s).",
        (lambda df: (df.attrs["markers_info"].update(marker1="not a dict"), df)[1])(
            _basic_valid_vcf_data()
        ),
    ),
    (
        "marker_info_missing_keys",
        InvalidVcfDataError.ReasonCode.MARKER_INFO_MISSING_KEYS,
        {"marker_ids": ["marker1"]},
        "markers_info entries is missing expected key(s) for 1 marker(s).",
        (lambda df: (df.attrs["markers_info"]["marker1"].pop("chrom", None), df)[1])(
            _basic_valid_vcf_data()
        ),
    ),
    (
        "marker_info_invalid_chrom",
        InvalidVcfDataError.ReasonCode.MARKER_INFO_INVALID_CHROM,
        {"marker_ids": ["marker1"]},
        "markers_info entries for 'chrom' must be a non-empty str, "
        "invalid for 1 marker(s).",
        (lambda df: (df.attrs["markers_info"]["marker1"].update(chrom=""), df)[1])(
            _basic_valid_vcf_data()
        ),
    ),
    (
        "marker_info_invalid_pos",
        InvalidVcfDataError.ReasonCode.MARKER_INFO_INVALID_POS,
        {"marker_ids": ["marker1"]},
        "markers_info entries for 'pos' must be a positive int, "
        "invalid for 1 marker(s).",
        (lambda df: (df.attrs["markers_info"]["marker1"].update(pos="A"), df)[1])(
            _basic_valid_vcf_data()
        ),
    ),
    (
        "marker_info_invalid_pos_bool",
        InvalidVcfDataError.ReasonCode.MARKER_INFO_INVALID_POS,
        {"marker_ids": ["marker1"]},
        "markers_info entries for 'pos' must be a positive int, "
        "invalid for 1 marker(s).",
        (lambda df: (df.attrs["markers_info"]["marker1"].update(pos=True), df)[1])(
            _basic_valid_vcf_data()
        ),
    ),
    (
        "marker_info_invalid_id",
        InvalidVcfDataError.ReasonCode.MARKER_INFO_INVALID_ID,
        {"marker_ids": ["marker1"]},
        "markers_info entries for 'id' must be a str or None, "
        "invalid for 1 marker(s).",
        (lambda df: (df.attrs["markers_info"]["marker1"].update(id=1), df)[1])(
            _basic_valid_vcf_data()
        ),
    ),
    (
        "marker_info_invalid_ref",
        InvalidVcfDataError.ReasonCode.MARKER_INFO_INVALID_REF,
        {"marker_ids": ["marker1"]},
        "markers_info entries for 'ref' must be a non-empty str, "
        "invalid for 1 marker(s).",
        (lambda df: (df.attrs["markers_info"]["marker1"].update(ref=1), df)[1])(
            _basic_valid_vcf_data()
        ),
    ),
    (
        "marker_info_invalid_alt",
        InvalidVcfDataError.ReasonCode.MARKER_INFO_INVALID_ALT,
        {"marker_ids": ["marker1"]},
        "markers_info entries for 'alt' must be a list of str, "
        "invalid for 1 marker(s).",
        (lambda df: (df.attrs["markers_info"]["marker1"].update(alt="A"), df)[1])(
            _basic_valid_vcf_data()
        ),
    ),
    (
        "duplicated_markers_1",
        InvalidVcfDataError.ReasonCode.DUPLICATED_MARKERS,
        {"duplicated": ["marker1"]},
        "Duplicated marker id(s) in columns: ['marker1'].",
        (lambda df: pd.concat([df, df[["marker1"]]], axis=1))(_basic_valid_vcf_data()),
    ),
    (
        "duplicated_markers_2",
        InvalidVcfDataError.ReasonCode.DUPLICATED_MARKERS,
        {"duplicated": ["marker1", "marker2"]},
        "Duplicated marker id(s) in columns: ['marker1', 'marker2'].",
        (lambda df: pd.concat([df, df[["marker1", "marker2"]]], axis=1))(
            _basic_valid_vcf_data()
        ),
    ),
)


@pytest.fixture(params=[id for id, _, _, _, _ in invalidVcfDataError_errors_cases])
def invalid_vcf_data(request):

    invalidVcfDataError_errors_cases_dict = {
        i: {"reason": r, "extra": e, "expected_message": m, "data": d}
        for i, r, e, m, d in invalidVcfDataError_errors_cases
    }
    return invalidVcfDataError_errors_cases_dict.get(request.param)


class TestUnexpectedMarkerIdFormatError:
    def test_error_is_DeepgenocompressError(self):
        assert issubclass(UnexpectedMarkerIdFormatError, DeepgenocompressError)

    def test_error_messages_and_extra(self):
        err = UnexpectedMarkerIdFormatError(
            provided="toto",
        )

        assert "provided" in err.extra
        assert err.extra["provided"] == "toto"

        expected_msg = (
            "Unexpected marker_id_format: 'toto'. Expected values are: "
            "id, pos, ref_alt, alleles."
        )

        assert expected_msg in str(err)


class TestInvalidVcfDataError:
    def test_error_is_DeepgenocompressError(self):
        assert issubclass(InvalidVcfDataError, DeepgenocompressError)

    @pytest.mark.parametrize(
        "reason, extra, expected_msg",
        [
            pytest.param(reason, extra, msg, id=id)
            for id, reason, extra, msg, _data in invalidVcfDataError_errors_cases
        ],
    )
    def test_single_error_messages_and_extra(self, reason, extra, expected_msg):

        test_error_list = [{"reason": reason, **extra}]
        err = InvalidVcfDataError(test_error_list)
        assert "errors_list" in err.extra
        assert isinstance(err.extra["errors_list"], list)
        assert len(err.extra["errors_list"]) == len(test_error_list)

        for err_error, test_error in zip(
            err.extra["errors_list"], test_error_list, strict=True
        ):
            for k, v in test_error.items():
                assert k in err_error
                assert v == err_error[k]

        assert "Data is not valid VCF data:" in str(err)
        assert expected_msg in str(err)

    def test_all_errors_messages_and_extra(self):

        test_error_list = [
            {"reason": r, **e} for _id, r, e, _m, _d in invalidVcfDataError_errors_cases
        ]
        expected_msgs = [m for _id, _r, _e, m, _d in invalidVcfDataError_errors_cases]

        err = InvalidVcfDataError(test_error_list)
        assert "errors_list" in err.extra
        assert isinstance(err.extra["errors_list"], list)
        assert len(err.extra["errors_list"]) == len(test_error_list)

        for err_error, test_error in zip(
            err.extra["errors_list"], test_error_list, strict=True
        ):
            for k, v in test_error.items():
                assert k in err_error
                assert v == err_error[k]

        assert "Data is not valid VCF data:" in str(err)
        for expected_msg in expected_msgs:
            assert expected_msg in str(err)


class Test_validate_vcf_data:

    @pytest.mark.parametrize(
        "vcf_input",
        [
            lf("valid_vcf_data_with_edge_cases"),
            lf("invalid_vcf_data"),
        ],
    )
    def test_returns_expected_strucutre(self, vcf_input):
        vcf_data = (
            vcf_input if isinstance(vcf_input, pd.DataFrame) else vcf_input["data"]
        )
        validation = validate_vcf_data(vcf_data, as_strings=False)
        assert isinstance(validation, dict)
        assert "errors" in validation
        assert "warnings" in validation
        assert isinstance(validation["errors"], list)
        assert isinstance(validation["warnings"], list)

    def test_no_error_nor_warn_for_valid_inputs(self, valid_vcf_data_with_edge_cases):
        validation = validate_vcf_data(valid_vcf_data_with_edge_cases, as_strings=False)
        assert len(validation["errors"]) == 0
        assert len(validation["warnings"]) == 0

    def test_returns_expected_error_strucutre_as_string_false(self, invalid_vcf_data):
        vcf_data = invalid_vcf_data["data"]
        validation = validate_vcf_data(vcf_data, as_strings=False)
        for err in validation["errors"]:
            assert isinstance(err, dict)

    def test_returns_expected_error_strucutre_as_string_true(self, invalid_vcf_data):
        vcf_data = invalid_vcf_data["data"]
        validation = validate_vcf_data(vcf_data, as_strings=True)
        for err in validation["errors"]:
            assert isinstance(err, str)

    def test_detect_invalid_data(self, invalid_vcf_data):
        vcf_data = invalid_vcf_data["data"]
        validation = validate_vcf_data(vcf_data, as_strings=False)
        assert len(validation["errors"]) == 1

    def test_return_expected_reason(self, invalid_vcf_data):
        vcf_data = invalid_vcf_data["data"]
        validation = validate_vcf_data(vcf_data, as_strings=False)
        assert validation["errors"][0]["reason"] == invalid_vcf_data["reason"]

    def test_return_expected_extra_info(self, invalid_vcf_data):
        vcf_data = invalid_vcf_data["data"]
        validation = validate_vcf_data(vcf_data, as_strings=False)

        for k, v in invalid_vcf_data["extra"].items():
            assert validation["errors"][0][k] == v, f"for key {k}"

    def test_return_expected_error_msgs(self, invalid_vcf_data):
        vcf_data = invalid_vcf_data["data"]
        validation = validate_vcf_data(vcf_data, as_strings=True)

        assert validation["errors"][0] == invalid_vcf_data["expected_message"]

    def test_deep_false(self, invalid_vcf_data):
        vcf_data = invalid_vcf_data["data"]
        validation = validate_vcf_data(vcf_data, deep=False, as_strings=True)

        expected_failures = [
            InvalidVcfDataError.ReasonCode.NOT_A_DATAFRAME,
            InvalidVcfDataError.ReasonCode.MARKERS_INFO_ATTR_MISSING,
            InvalidVcfDataError.ReasonCode.USE_BASES_ATTR_MISSING,
            InvalidVcfDataError.ReasonCode.USE_BASES_INVALID_TYPE,
        ]

        if invalid_vcf_data["reason"] in expected_failures:
            assert len(validation["errors"]) == 1
        else:
            assert len(validation["errors"]) == 0

    def test_validation_pass_with_extra_attrs(self, basic_valid_vcf_data_bases):
        basic_valid_vcf_data_bases.attrs["extra key"] = "something extra"
        validation = validate_vcf_data(basic_valid_vcf_data_bases)
        assert len(validation["errors"]) == 0

    def test_with_multiple_errors(self, basic_valid_vcf_data_bases):
        vcf_data = basic_valid_vcf_data_bases
        vcf_data.attrs["markers_info"]["marker1"].update(ref=1)
        vcf_data.attrs["markers_info"]["marker2"].update(alt="A")

        validation = validate_vcf_data(basic_valid_vcf_data_bases, as_strings=False)
        assert len(validation["errors"]) == 2

        validation_failure_reasons = [e["reason"] for e in validation["errors"]]
        assert (
            InvalidVcfDataError.ReasonCode.MARKER_INFO_INVALID_REF
            in validation_failure_reasons
        )
        assert (
            InvalidVcfDataError.ReasonCode.MARKER_INFO_INVALID_ALT
            in validation_failure_reasons
        )

    def test_marker_info_validity_with_missing_key(self, basic_valid_vcf_data_bases):
        # case where a marker info key is missing and another key is invalid for
        # the same marker
        # here "chrom" is missing and "id" is invalid for "marker1"
        vcf_data = basic_valid_vcf_data_bases
        vcf_data.attrs["markers_info"]["marker1"].pop("chrom", None)
        vcf_data.attrs["markers_info"]["marker1"]["id"] = 1

        validation = validate_vcf_data(vcf_data, as_strings=False)
        assert len(validation["errors"]) == 2

        validation_failure_reasons = [e["reason"] for e in validation["errors"]]
        assert (
            InvalidVcfDataError.ReasonCode.MARKER_INFO_MISSING_KEYS
            in validation_failure_reasons
        )
        assert (
            InvalidVcfDataError.ReasonCode.MARKER_INFO_INVALID_ID
            in validation_failure_reasons
        )

    @pytest.mark.parametrize(
        "as_strings", [True, False], ids=["as_strings=True", "as_strings=False"]
    )
    def test_with_extra_markers_info_entries(
        self, basic_valid_vcf_data_bases, as_strings
    ):
        vcf_data = basic_valid_vcf_data_bases
        vcf_data.attrs["markers_info"]["extra_marker"] = {
            "chrom": "1",
            "pos": 42,
            "id": "extra_marker",
            "ref": "A",
            "alt": ["T"],
        }

        validation = validate_vcf_data(vcf_data, as_strings=as_strings)
        assert len(validation["warnings"]) == 1
        assert validation["warnings"][0] == (
            "attrs['markers_info'] has entries not present in columns: "
            "['extra_marker']."
        )

    @pytest.mark.parametrize(
        "as_strings", [True, False], ids=["as_strings=True", "as_strings=False"]
    )
    def test_with_duplicated_samples(self, basic_valid_vcf_data_bases, as_strings):
        vcf_data = basic_valid_vcf_data_bases
        duplicated_row = vcf_data.loc[["sample2"]]
        vcf_data = pd.concat([vcf_data, duplicated_row])

        validation = validate_vcf_data(vcf_data, as_strings=as_strings)
        assert len(validation["warnings"]) == 1
        assert (
            validation["warnings"][0]
            == "Duplicated sample(s) in row index: ['sample2']."
        )


class Test_build_marker_ids:

    def test_pos_format_is_correct_with_necessary_arguments(self):
        result = build_marker_ids(chrom="1", pos=1_234_567_890, marker_id_format="pos")
        assert result == "1@1234567890"

    def test_pos_format_is_correct_with_all_arguments(self):
        result = build_marker_ids(
            chrom="chr1",
            pos=1_234_567_890,
            id="SNP_01",
            ref="A",
            alt=["T"],
            marker_id_format="pos",
        )
        assert result == "chr1@1234567890"

    def test_pos_format_raise_with_missing_chrom(self):
        with pytest.raises(
            TypeError,
            match=re.escape("build_marker_ids() missing 1 required argument(s): chrom"),
        ):
            build_marker_ids(pos=100, marker_id_format="pos")

    def test_pos_format_raise_with_missing_pos(self):
        with pytest.raises(
            TypeError,
            match=re.escape("build_marker_ids() missing 1 required argument(s): pos"),
        ):
            build_marker_ids(chrom="1", marker_id_format="pos")

    def test_pos_format_raise_with_missing_chrom_pos(self):
        with pytest.raises(
            TypeError,
            match=re.escape(
                "build_marker_ids() missing 2 required argument(s): chrom, pos"
            ),
        ):
            build_marker_ids(id="SNP_01", ref="A", alt=["T"], marker_id_format="pos")

    def test_id_format_is_correct_with_necessary_arguments_str(self):
        result = build_marker_ids(id="SNP_01", marker_id_format="id")
        assert result == "SNP_01"

    def test_id_format_is_correct_with_necessary_arguments_None(self):
        result = build_marker_ids(id=None, marker_id_format="id")
        assert result == "."

    def test_id_format_is_correct_with_all_arguments(self):
        result = build_marker_ids(
            chrom="chr1",
            pos=1_234_567_890,
            id="SNP_01",
            ref="A",
            alt=["T"],
            marker_id_format="id",
        )
        assert result == "SNP_01"

    def test_id_format_raise_with_missing_id(self):
        with pytest.raises(
            TypeError,
            match=re.escape("build_marker_ids() missing 1 required argument(s): id"),
        ):
            build_marker_ids(chrom="1", pos=100, marker_id_format="id")

    def test_ref_alt_format_is_correct_with_necessary_arguments(self):
        result = build_marker_ids(
            chrom="1", pos=1_234_567_890, ref="A", alt=["T"], marker_id_format="ref_alt"
        )
        assert result == "1@1234567890_A_T"

    def test_ref_alt_format_is_correct_with_necessary_arguments_several_alts(self):
        result = build_marker_ids(
            chrom="1",
            pos=1_234_567_890,
            ref="A",
            alt=["T", "G", "C"],
            marker_id_format="ref_alt",
        )
        assert result == "1@1234567890_A_T-G-C"

    def test_ref_alt_format_is_correct_with_all_arguments(self):
        result = build_marker_ids(
            chrom="chr1",
            pos=1_234_567_890,
            id="SNP_01",
            ref="A",
            alt=["T"],
            marker_id_format="ref_alt",
        )
        assert result == "chr1@1234567890_A_T"

    def test_ref_alt_format_raise_with_missing_chrom(self):
        with pytest.raises(
            TypeError,
            match=re.escape("build_marker_ids() missing 1 required argument(s): chrom"),
        ):
            build_marker_ids(pos=100, ref="A", alt=["T"], marker_id_format="ref_alt")

    def test_ref_alt_format_raise_with_missing_pos(self):
        with pytest.raises(
            TypeError,
            match=re.escape("build_marker_ids() missing 1 required argument(s): pos"),
        ):
            build_marker_ids(chrom="1", ref="A", alt=["T"], marker_id_format="ref_alt")

    def test_ref_alt_format_raise_with_missing_ref(self):
        with pytest.raises(
            TypeError,
            match=re.escape("build_marker_ids() missing 1 required argument(s): ref"),
        ):
            build_marker_ids(chrom="1", pos=100, alt=["T"], marker_id_format="ref_alt")

    def test_ref_alt_format_raise_with_missing_alt(self):
        with pytest.raises(
            TypeError,
            match=re.escape("build_marker_ids() missing 1 required argument(s): alt"),
        ):
            build_marker_ids(chrom="1", pos=100, ref="A", marker_id_format="ref_alt")

    def test_alleles_format_is_correct_with_necessary_arguments(self):
        result = build_marker_ids(
            chrom="1", pos=1_234_567_890, ref="T", alt=["A"], marker_id_format="alleles"
        )
        assert result == "1@1234567890_A-T"

    def test_alleles_format_is_correct_with_necessary_arguments_several_alts(self):
        result = build_marker_ids(
            chrom="1",
            pos=1_234_567_890,
            ref="T",
            alt=["A", "G", "C"],
            marker_id_format="alleles",
        )
        assert result == "1@1234567890_A-C-G-T"

    def test_alleles_format_is_correct_with_all_arguments(self):
        result = build_marker_ids(
            chrom="chr1",
            pos=1_234_567_890,
            id="SNP_01",
            ref="C",
            alt=["A", "G", "T"],
            marker_id_format="alleles",
        )
        assert result == "chr1@1234567890_A-C-G-T"

    def test_alleles_format_raise_with_missing_chrom(self):
        with pytest.raises(
            TypeError,
            match=re.escape("build_marker_ids() missing 1 required argument(s): chrom"),
        ):
            build_marker_ids(pos=100, ref="A", alt=["T"], marker_id_format="alleles")

    def test_alleles_format_raise_with_missing_pos(self):
        with pytest.raises(
            TypeError,
            match=re.escape("build_marker_ids() missing 1 required argument(s): pos"),
        ):
            build_marker_ids(chrom="1", ref="A", alt=["T"], marker_id_format="alleles")

    def test_alleles_format_raise_with_missing_ref(self):
        with pytest.raises(
            TypeError,
            match=re.escape("build_marker_ids() missing 1 required argument(s): ref"),
        ):
            build_marker_ids(chrom="1", pos=100, alt=["T"], marker_id_format="alleles")

    def test_alleles_format_raise_with_missing_alt(self):
        with pytest.raises(
            TypeError,
            match=re.escape("build_marker_ids() missing 1 required argument(s): alt"),
        ):
            build_marker_ids(chrom="1", pos=100, ref="A", marker_id_format="alleles")

    def test_invalid_format_raises_error(self):
        with pytest.raises(UnexpectedMarkerIdFormatError):
            build_marker_ids(
                chrom="1",
                pos=100,
                id="rs1",
                ref="A",
                alt=["T"],
                marker_id_format="unexpected fromat",  # pyright: ignore [reportArgumentType]
            )


class Test_reindex_vcf_data:

    def test_correct_column_names(
        self, valid_vcf_data_with_edge_cases, marker_id_format
    ):
        reindexed_data = reindex_vcf_data(
            valid_vcf_data_with_edge_cases, marker_id_format=marker_id_format
        )

        ordered_markers_info = {
            col: valid_vcf_data_with_edge_cases.attrs["markers_info"][col]
            for col in valid_vcf_data_with_edge_cases.columns
        }
        expected_columns_names = [
            build_marker_ids(**m_info, marker_id_format=marker_id_format)
            for m_info in ordered_markers_info.values()
        ]
        assert list(reindexed_data.columns) == expected_columns_names

    def test_update_markers_info_keys(
        self, valid_vcf_data_with_edge_cases, marker_id_format
    ):
        reindexed_data = reindex_vcf_data(
            valid_vcf_data_with_edge_cases, marker_id_format=marker_id_format
        )
        assert set(reindexed_data.attrs["markers_info"]) == set(reindexed_data.columns)

    def test_keep_markers_info_values(
        self, valid_vcf_data_with_edge_cases, marker_id_format
    ):
        reindexed_data = reindex_vcf_data(
            valid_vcf_data_with_edge_cases, marker_id_format=marker_id_format
        )

        old_columns = valid_vcf_data_with_edge_cases.columns
        new_columns = reindexed_data.columns

        for old_col, new_col in zip(old_columns, new_columns, strict=True):
            assert (
                reindexed_data.attrs["markers_info"][new_col]
                == valid_vcf_data_with_edge_cases.attrs["markers_info"][old_col]
            )

    def test_keep_other_attrs_unchanged(
        self, valid_vcf_data_with_edge_cases, marker_id_format
    ):
        reindexed_data = reindex_vcf_data(
            valid_vcf_data_with_edge_cases, marker_id_format=marker_id_format
        )

        other_initial_attrs = {
            k: v
            for k, v in valid_vcf_data_with_edge_cases.attrs.items()
            if k != "markers_info"
        }
        other_reindexed_attrs = {
            k: v for k, v in reindexed_data.attrs.items() if k != "markers_info"
        }

        assert other_reindexed_attrs == other_initial_attrs

    def test_geno_values_unchanged(
        self, valid_vcf_data_with_edge_cases, marker_id_format
    ):
        reindexed_data = reindex_vcf_data(
            valid_vcf_data_with_edge_cases, marker_id_format=marker_id_format
        )
        np.testing.assert_array_equal(
            reindexed_data.to_numpy(), valid_vcf_data_with_edge_cases.to_numpy()
        )

    def test_row_index_unchanged(
        self, valid_vcf_data_with_edge_cases, marker_id_format
    ):
        reindexed_data = reindex_vcf_data(
            valid_vcf_data_with_edge_cases, marker_id_format=marker_id_format
        )
        assert list(reindexed_data.index) == list(valid_vcf_data_with_edge_cases.index)

    def test_initial_vcf_data_not_mutated(
        self, valid_vcf_data_with_edge_cases, marker_id_format
    ):
        initial_data_copy = valid_vcf_data_with_edge_cases.copy(deep=True)
        reindexed_data = reindex_vcf_data(
            valid_vcf_data_with_edge_cases, marker_id_format=marker_id_format
        )
        assert reindexed_data is not valid_vcf_data_with_edge_cases
        pd.testing.assert_frame_equal(valid_vcf_data_with_edge_cases, initial_data_copy)

    def test_raise_DuplicatedMarkerIDsError(self, basic_valid_vcf_data_bases):
        # set all markers_info's ID to None and call with marker_id_format="id"

        for m_info in basic_valid_vcf_data_bases.attrs["markers_info"]:
            basic_valid_vcf_data_bases.attrs["markers_info"][m_info]["id"] = None

        with pytest.raises(DuplicatedMarkerIDsError):
            reindex_vcf_data(basic_valid_vcf_data_bases, marker_id_format="id")

    def test_raise_UnexpectedMarkerIdFormatError(self, basic_valid_vcf_data_bases):
        with pytest.raises(UnexpectedMarkerIdFormatError):
            reindex_vcf_data(
                basic_valid_vcf_data_bases,
                marker_id_format="unexpected format",  # pyright: ignore [reportArgumentType]
            )

    def test_raise_InvalidVcfDataError(
        self, basic_valid_vcf_data_bases, marker_id_format
    ):
        df = basic_valid_vcf_data_bases
        del df.attrs["markers_info"]
        with pytest.raises(InvalidVcfDataError):
            reindex_vcf_data(df, marker_id_format=marker_id_format)

    def test_chaining_and_looping_format_return_to_original(
        self, basic_valid_vcf_data_bases
    ):
        # Note: this needs the vcf_data to not have several marker info that will not
        # lead to duplicated ids with all of formats

        marker_id_formats = get_args(_MARKER_ID_FORMAT)
        vcf_data = reindex_vcf_data(
            basic_valid_vcf_data_bases, marker_id_format=marker_id_formats[0]
        )
        init_vcf_data = vcf_data.copy(deep=True)
        for marker_id_format in marker_id_formats[1:]:
            vcf_data = reindex_vcf_data(vcf_data, marker_id_format=marker_id_format)
        vcf_data = reindex_vcf_data(vcf_data, marker_id_format=marker_id_formats[0])

        pd.testing.assert_frame_equal(vcf_data, init_vcf_data)


class Test_build_vcf_encoding_map:

    def test_use_bases_false(self, basic_valid_vcf_data_no_bases):
        # build_vcf_encoding_map returns a fixed encoding map whatever the dataFrame's
        # values when use_bases is False
        encoding_map = build_vcf_encoding_map(basic_valid_vcf_data_no_bases)
        assert encoding_map == {
            0: [1.0, 0.0, 0.0],
            1: [0.0, 1.0, 0.0],
            2: [0.0, 0.0, 1.0],
            3: [0.0, 0.0, 0.0],
        }

    def test_does_not_deep_validate_custom_vcf_data(self, basic_valid_vcf_data_bases):
        # deep=False is intentional: malformed custom VCF data will not be rejected.
        vcf_data = basic_valid_vcf_data_bases
        vcf_data.attrs["use_bases"] = False
        vcf_data.attrs["markers_info"] = (
            None  # wrong but no effect for build_vcf_encoding_map
        )
        assert build_vcf_encoding_map(vcf_data)

        vcf_data.attrs["use_bases"] = True
        assert build_vcf_encoding_map(vcf_data)

    def test_add_error_note_with_badly_formated_genotypes(
        self, basic_valid_vcf_data_bases
    ):
        # Since malformed custom VCF data will not be rejected.
        # it could be source of error. In such case a note is added to raised errors
        # to suggest to validate the data
        vcf_data = basic_valid_vcf_data_bases
        vcf_data.iloc[0, 0] = "AA"  # can't be parsed

        with pytest.raises(Exception) as e:
            build_vcf_encoding_map(vcf_data)
        assert any("validate_vcf_data(vcf_data)" in note for note in e.value.__notes__)

    def test_use_bases_true_map_have_missing_value(self):
        vcf_data = pd.DataFrame(["A/A"])
        vcf_data.attrs["use_bases"] = True
        vcf_data.attrs["markers_info"] = None

        encoding_map = build_vcf_encoding_map(vcf_data)
        assert "./." in encoding_map
        assert ".|." in encoding_map

        assert all(x == 0 for x in encoding_map["./."])
        assert all(x == 0 for x in encoding_map[".|."])

    def test_use_bases_true_map_have_phased_and_unphased_values(self):
        vcf_data = pd.DataFrame(["A/A", "C|C"])
        vcf_data.attrs["use_bases"] = True
        vcf_data.attrs["markers_info"] = None

        encoding_map = build_vcf_encoding_map(vcf_data)
        _validate_encoding_map(encoding_map)

        assert "A/A" in encoding_map
        assert "A|A" in encoding_map
        assert "C/C" in encoding_map
        assert "C|C" in encoding_map

        assert encoding_map["A/A"] == encoding_map["A|A"]
        assert encoding_map["C/C"] == encoding_map["C|C"]

        assert encoding_map["A/A"] != encoding_map["C|C"]

    def test_use_bases_true_deduplication_accross_phasing(self):
        # even if A/A and A|A appears in different cells,
        # they should have the same encoding
        vcf_data = pd.DataFrame(["A/A", "A|A"])
        vcf_data.attrs["use_bases"] = True
        vcf_data.attrs["markers_info"] = None

        encoding_map = build_vcf_encoding_map(vcf_data)
        _validate_encoding_map(encoding_map)

        assert "A/A" in encoding_map
        assert "A|A" in encoding_map

        assert encoding_map["A/A"] == encoding_map["A|A"]

    def test_use_bases_true_map_have_all_combination_of_alleles(self):
        vcf_data = pd.DataFrame(["A/T", "C|G"])
        vcf_data.attrs["use_bases"] = True
        vcf_data.attrs["markers_info"] = None

        encoding_map = build_vcf_encoding_map(vcf_data)
        _validate_encoding_map(encoding_map)

        assert "A/T" in encoding_map
        assert "T/A" in encoding_map
        assert "A|T" in encoding_map
        assert "T|A" in encoding_map
        assert encoding_map["A/T"] == encoding_map["T/A"]
        assert encoding_map["A/T"] == encoding_map["T|A"]
        assert encoding_map["A/T"] == encoding_map["A|T"]

        assert "C/G" in encoding_map
        assert "G/C" in encoding_map
        assert "C|G" in encoding_map
        assert "G|C" in encoding_map
        assert encoding_map["C/G"] == encoding_map["G/C"]
        assert encoding_map["C/G"] == encoding_map["G|C"]
        assert encoding_map["C/G"] == encoding_map["C|G"]

        assert encoding_map["A/T"] != encoding_map["C/G"]

    def test_use_bases_true_deduplication_accross_mirrored_alleles(self):
        # even if A/T and T/A appears in different cells,
        # they should have the same encoding
        vcf_data = pd.DataFrame(["A/T", "T/A"])
        vcf_data.attrs["use_bases"] = True
        vcf_data.attrs["markers_info"] = None

        encoding_map = build_vcf_encoding_map(vcf_data)
        _validate_encoding_map(encoding_map)

        assert "A/T" in encoding_map
        assert "T/A" in encoding_map

        assert encoding_map["A/T"] == encoding_map["T/A"]

    def test_use_bases_true_map_with_partial_missing_values(self):
        # partial missing genotypes have their own encoding
        vcf_data = pd.DataFrame(["A/T", ".|A"])
        vcf_data.attrs["use_bases"] = True
        vcf_data.attrs["markers_info"] = None

        encoding_map = build_vcf_encoding_map(vcf_data)
        _validate_encoding_map(encoding_map)

        assert "A/." in encoding_map
        assert "./A" in encoding_map
        assert "A|." in encoding_map
        assert ".|A" in encoding_map
        assert encoding_map["A/."] == encoding_map["./A"]
        assert encoding_map["A/."] == encoding_map[".|A"]
        assert encoding_map["A/."] == encoding_map["A|."]

        assert encoding_map["A/."] != encoding_map["A/T"]

    def test_use_bases_true_with_empty_df(self):
        vcf_data = pd.DataFrame()
        vcf_data.attrs["use_bases"] = True
        vcf_data.attrs["markers_info"] = None

        encoding_map = build_vcf_encoding_map(vcf_data)
        _validate_encoding_map(encoding_map)

        assert encoding_map == {"./.": [0], ".|.": [0]}

    def test_use_bases_true_with_df_full_of_missing_values(self):
        vcf_data = pd.DataFrame(["./.", ".|."])
        vcf_data.attrs["use_bases"] = True
        vcf_data.attrs["markers_info"] = None

        encoding_map = build_vcf_encoding_map(vcf_data)
        _validate_encoding_map(encoding_map)

        assert encoding_map == {"./.": [0], ".|.": [0]}


class Test_read_vcf:
    VCF_DIR = Path(__file__).parent / "fixtures" / "vcf"

    BASIC_VCF_MARKERS_INFO = (
        {
            "chrom": "chr1",
            "pos": 1378660,
            "id": "snp_01",
            "ref": "A",
            "alt": ["G"],
        },
        {
            "chrom": "chr1",
            "pos": 7632264,
            "id": None,
            "ref": "C",
            "alt": ["T"],
        },
        {
            "chrom": "chr2",
            "pos": 2466877,
            "id": "snp_03",
            "ref": "T",
            "alt": ["C", "A"],
        },
        {
            "chrom": "chr2",
            "pos": 5627849,
            "id": "snp_04",
            "ref": "G",
            "alt": ["A"],
        },
    )

    @pytest.mark.parametrize(
        "use_bases", [True, False], ids=["use_bases = True", "use_bases = False"]
    )
    def test_returns_valid_vcf_data(self, use_bases, marker_id_format):
        vcf_file = self.VCF_DIR / "basic.vcf"
        vcf_data = read_vcf(
            vcf_file, use_bases=use_bases, marker_id_format=marker_id_format
        )
        vcf_validation = validate_vcf_data(vcf_data, deep=True)
        assert (
            len(vcf_validation["errors"]) == 0
        ), f"expected no error, got {vcf_validation['errors']}"

        assert (
            len(vcf_validation["warnings"]) == 0
        ), f"expected no warnings, got {vcf_validation['warnings']}"

    @pytest.mark.parametrize(
        "use_bases", [True, False], ids=["use_bases = True", "use_bases = False"]
    )
    def test_returns_expected_df_structure(self, use_bases, marker_id_format):
        # Verifies the basic contract: given a small, known VCF file, the returned
        # object is a pd.DataFrame whose index equals the VCF sample names and whose
        # columns equal the expected marker IDs
        vcf_file = self.VCF_DIR / "basic.vcf"
        vcf_data = read_vcf(
            vcf_file, use_bases=use_bases, marker_id_format=marker_id_format
        )

        assert isinstance(vcf_data, pd.DataFrame)
        assert set(vcf_data.index) == {"ind_1", "ind_2", "ind_3"}
        assert set(vcf_data.columns) == {
            build_marker_ids(**m_info, marker_id_format=marker_id_format)
            for m_info in self.BASIC_VCF_MARKERS_INFO
        }

    @pytest.mark.parametrize(
        "use_bases", [True, False], ids=["use_bases = True", "use_bases = False"]
    )
    def test_sets_attrs_use_bases_and_markers_info(self, use_bases, marker_id_format):
        # Confirms vcf_data.attrs contains "use_bases" (matching the input flag) and
        # "markers_info" (a dict keyed by marker ID, with correct
        # chrom, pos, id, ref, alt for each variant).
        vcf_file = self.VCF_DIR / "basic.vcf"
        vcf_data = read_vcf(
            vcf_file, use_bases=use_bases, marker_id_format=marker_id_format
        )

        assert isinstance(vcf_data.attrs, dict)
        assert "use_bases" in vcf_data.attrs
        assert "markers_info" in vcf_data.attrs
        assert vcf_data.attrs["use_bases"] == use_bases

        expected_markers_info = {
            build_marker_ids(**m_info, marker_id_format=marker_id_format): m_info
            for m_info in self.BASIC_VCF_MARKERS_INFO
        }
        assert vcf_data.attrs["markers_info"] == expected_markers_info

    def test_use_bases_true_returns_base_genotypes(self):
        vcf_file = self.VCF_DIR / "basic.vcf"
        vcf_data = read_vcf(vcf_file, use_bases=True, marker_id_format="ref_alt")

        expected = pd.DataFrame(
            {
                "chr1@1378660_A_G": ["A|G", "G|G", "G|A"],
                "chr1@7632264_C_T": ["C/T", "C/C", "T/T"],
                "chr2@2466877_T_C-A": ["C/C", "C/C", "T/C"],
                "chr2@5627849_G_A": ["G/A", "./.", "G/A"],
            },
            index=["ind_1", "ind_2", "ind_3"],
        )
        pd.testing.assert_frame_equal(
            vcf_data,
            expected,
            check_like=True,
            check_dtype=False,
        )

    def test_use_bases_false_returns_numeric_genotypes(self):
        vcf_file = self.VCF_DIR / "basic.vcf"
        vcf_data = read_vcf(vcf_file, use_bases=False, marker_id_format="ref_alt")

        expected = pd.DataFrame(
            {
                "chr1@1378660_A_G": [1, 2, 1],
                "chr1@7632264_C_T": [1, 0, 2],
                "chr2@2466877_T_C-A": [2, 2, 1],
                "chr2@5627849_G_A": [1, 3, 1],
            },
            index=["ind_1", "ind_2", "ind_3"],
        )
        pd.testing.assert_frame_equal(
            vcf_data,
            expected,
            check_like=True,
            check_dtype=False,
        )

    def test_duplicated_marker_ids_raises(self):
        vcf_file = self.VCF_DIR / "basic_no_marker_id.vcf"
        with pytest.raises(
            DuplicatedMarkerIDsError,
        ):
            read_vcf(vcf_file, use_bases=False, marker_id_format="id")

    def test_empty_vcf_returns_empty_dataframe(self):
        vcf_file = self.VCF_DIR / "empty.vcf"
        vcf_data = read_vcf(vcf_file, use_bases=False, marker_id_format="ref_alt")

        assert set(vcf_data.index) == {"ind_1", "ind_2", "ind_3"}
        assert vcf_data.shape == (3, 0)

        vcf_validation = validate_vcf_data(vcf_data, deep=True)
        assert (
            len(vcf_validation["errors"]) == 0
        ), f"expected no error, got {vcf_validation['errors']}"

        assert (
            len(vcf_validation["warnings"]) == 0
        ), f"expected no warnings, got {vcf_validation['warnings']}"

    def test_default_parameters(self):
        vcf_file = self.VCF_DIR / "basic.vcf"
        vcf_data_default = read_vcf(vcf_file)
        vcf_data_explicit = read_vcf(
            vcf_file, use_bases=False, marker_id_format="ref_alt"
        )
        pd.testing.assert_frame_equal(vcf_data_default, vcf_data_explicit)
        assert vcf_data_default.attrs == vcf_data_explicit.attrs

    def test_invalid_marker_id_format_raises(self):
        vcf_file = self.VCF_DIR / "basic.vcf"
        with pytest.raises(
            UnexpectedMarkerIdFormatError,
        ):
            read_vcf(
                vcf_file,
                use_bases=False,
                marker_id_format="Not a marker_id_format",  # pyright: ignore [reportArgumentType]
            )

    def test_missing_file_behavior(self, tmp_path):
        vcf_file = tmp_path / "does_not_exists.vcf"
        with pytest.raises(FileNotFoundError, match="VCF file not found"):
            read_vcf(vcf_file)

    @pytest.mark.upstream_watch
    def test_cyvcf2_bug_missing_genotypes_use_bases_true(self):
        """
        Regression/characterization test for a cyvcf2 gt_bases quirk.

        Upstream issue: https://github.com/brentp/cyvcf2/issues/331

        Pins cyvcf2's current (probably buggy) `gt_bases` separator behavior for
        phased genotypes with a missing allele.

        cyvcf2 renders '.' as unphased '/' when it's the *first* allele in
        a phased pair (e.g. '.|1' -> './A'), even though `genotypes` still
        reports phased=True.

        If this test fails after a cyvcf2 update, the bug may have been fixed.
        then we must adjust the documentation of `read_vcf` to suggest the
        correct version of cyvcf2 to use, add a warning, or adjust its minimum
        version in pyproject.toml.

        Note: what is actually expected is
        ```
                "snp_phased": ["A|.", "G|.", ".|.", ".|A", ".|G"],
        ```
        """
        vcf_file = self.VCF_DIR / "missing_genotypes.vcf"
        vcf_data = read_vcf(vcf_file, use_bases=True, marker_id_format="id")

        expected = pd.DataFrame(
            {
                "snp_phased": ["A|.", "G|.", "./.", "./A", "./G"],
                "snp_unphased": ["A/.", "G/.", "./.", "./A", "./G"],
            },
            index=["ind_1", "ind_2", "ind_3", "ind_4", "ind_5"],
        )

        pd.testing.assert_frame_equal(
            vcf_data,
            expected,
            check_like=True,
            check_dtype=False,
        )

    @pytest.mark.parametrize("strict_gt", [True, False])
    def test_passes_strict_gt_to_VCF(self, strict_gt, mocker: MockerFixture):
        vcf_file = self.VCF_DIR / "basic.vcf"
        mock_vcf_cls = mocker.patch("deepgenocompress._core.vcf.VCF", wraps=VCF)

        read_vcf(vcf_file, use_bases=True, strict_gt=strict_gt)
        mock_vcf_cls.assert_called_once_with(
            fname=str(vcf_file), gts012=True, strict_gt=strict_gt
        )

    def test_strict_gt_default_to_False(self, mocker: MockerFixture):
        vcf_file = self.VCF_DIR / "basic.vcf"
        mock_vcf_cls = mocker.patch("deepgenocompress._core.vcf.VCF", wraps=VCF)

        read_vcf(vcf_file, use_bases=True)
        mock_vcf_cls.assert_called_once_with(
            fname=str(vcf_file), gts012=True, strict_gt=False
        )

    def test_soft_parsing_error_behaviour(self):
        vcf_file = self.VCF_DIR / "bad_empty_file.vcf"

        with pytest.raises(Exception) as e:
            read_vcf(vcf_file)

        expected_note = (
            "Note: Pasring of the VCF file with 'cyvcf2' failed. "
            "Be sure your VCF file is valid."
        )
        assert any(expected_note == note for note in e.value.__notes__)

    @pytest.mark.skip(reason="not implemented")
    def test_fatal_parsing_error_behaviour(self):
        # Currently, such "fatal" error causes the entire Python process to crash,
        # which can provided the execution of other tests.
        # see: the "NOTE" in `read_vcf` implementation.
        vcf_file = self.VCF_DIR / "bad_missing_tab.vcf"

        with pytest.raises(Exception) as e:
            read_vcf(vcf_file)

        expected_note = (
            "Note: Pasring of the VCF file with 'cyvcf2' failed. "
            "Be sure your VCF file is valid."
        )
        assert any(expected_note == note for note in e.value.__notes__)
