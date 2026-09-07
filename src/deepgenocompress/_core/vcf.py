"""Utilities for handling vcf genotype files."""

import re
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, ClassVar, Literal, Required, TypedDict, cast, get_args, overload

import numpy as np
import pandas as pd
from cyvcf2 import VCF

from deepgenocompress._core.exceptions import (
    DeepgenocompressError,
    DuplicatedMarkerIDsError,
    _type_fullname,
)
from deepgenocompress._core.utils import _MISSING, _Missing

_MARKER_ID_FORMAT = Literal["id", "pos", "ref_alt", "alleles"]


class UnexpectedMarkerIdFormatError(DeepgenocompressError, ValueError):
    """Raised when provided marker_id_format is not expected."""

    class _Extra(TypedDict):
        provided: str

    extra: "_Extra | dict[str, Any]"
    """Extra information related to the error.

    One key ``provided`` being the unnexpected marker id format provided.
    """

    def __init__(self, provided):
        super().__init__(
            message=(
                f"Unexpected marker_id_format: {provided!r}. Expected values are: "
                f"{", ".join(get_args(_MARKER_ID_FORMAT))}."
            ),
            extra={"provided": provided},
        )


class InvalidVcfDataError(DeepgenocompressError, ValueError):
    """Raised when a DataFrame is not valid VCF data.

    Instances are constructed with a :class:`ReasonCode` identifying which validation
    failed, plus an ``extra`` mapping of contextual values.
    """

    class ReasonCode(StrEnum):
        """Possible invalid reasons."""

        def __repr__(self) -> str:
            """Return the string representation."""
            return self.name

        NOT_A_DATAFRAME = auto()
        """Provided ``vcf_data`` is not a :class:`pandas.DataFrame`."""
        USE_BASES_ATTR_MISSING = auto()
        """``attrs['use_bases']`` is missing."""
        USE_BASES_INVALID_TYPE = auto()
        """``attrs['use_bases']`` is not a :class:`bool`."""
        VALUES_INCONSISTENT_WITH_USE_BASES = auto()
        """Data values are inconsistent with ``attrs['use_bases']``."""
        MARKERS_INFO_ATTR_MISSING = auto()
        """``attrs['markers_info']`` is missing."""
        MARKERS_INFO_INVALID_TYPE = auto()
        """``attrs['markers_info']`` is not a :class:`dict`."""
        MARKERS_INFO_MISSING_COLUMNS = auto()
        """``attrs['markers_info']`` has no entry for some column(s)."""
        MARKER_INFO_INVALID_TYPE = auto()
        """One or more ``markers_info`` entry is not a :class:`dict`."""
        MARKER_INFO_MISSING_KEYS = auto()
        """One or more ``markers_info`` entry is missing required key(s)."""
        MARKER_INFO_INVALID_CHROM = auto()
        """One or more ``markers_info`` entry's ``chrom`` is invalid."""
        MARKER_INFO_INVALID_POS = auto()
        """One or more ``markers_info`` entry's ``pos`` is invalid."""
        MARKER_INFO_INVALID_ID = auto()
        """One or more ``markers_info`` entry's ``id`` is invalid."""
        MARKER_INFO_INVALID_REF = auto()
        """One or more ``markers_info`` entry's ``ref`` is invalid."""
        MARKER_INFO_INVALID_ALT = auto()
        """One or more ``markers_info`` entry's ``alt`` is invalid."""
        DUPLICATED_MARKERS = auto()
        """Duplicated marker id(s) in columns."""

    _MESSAGES: ClassVar[dict["InvalidVcfDataError.ReasonCode", str]] = {
        ReasonCode.NOT_A_DATAFRAME: (
            "`vcf_data` must be a pandas DataFrame, got {provided_type_str!r}."
        ),
        ReasonCode.USE_BASES_ATTR_MISSING: "attrs['use_bases'] is missing.",
        ReasonCode.USE_BASES_INVALID_TYPE: (
            "attrs['use_bases'] must be a bool, got {provided_type_str!r}."
        ),
        ReasonCode.VALUES_INCONSISTENT_WITH_USE_BASES: (
            "vcf_data contains value(s) inconsistent with "
            "use_bases={use_bases!r}: {sorted_invalid_values}."
        ),
        ReasonCode.MARKERS_INFO_ATTR_MISSING: "attrs['markers_info'] is missing.",
        ReasonCode.MARKERS_INFO_INVALID_TYPE: (
            "attrs['markers_info'] must be a dict, got {provided_type_str!r}."
        ),
        ReasonCode.MARKERS_INFO_MISSING_COLUMNS: (
            "attrs['markers_info'] missing information for column(s): "
            "{missing_columns!r}."
        ),
        ReasonCode.MARKER_INFO_INVALID_TYPE: (
            "markers_info entries must be dict, "
            "invalid for {n_invalid_markers} marker(s)."
        ),
        ReasonCode.MARKER_INFO_MISSING_KEYS: (
            "markers_info entries is missing expected key(s) "
            "for {n_invalid_markers} marker(s)."
        ),
        ReasonCode.MARKER_INFO_INVALID_CHROM: (
            "markers_info entries for 'chrom' must be a non-empty str, "
            "invalid for {n_invalid_markers} marker(s)."
        ),
        ReasonCode.MARKER_INFO_INVALID_POS: (
            "markers_info entries for 'pos' must be a positive int, "
            "invalid for {n_invalid_markers} marker(s)."
        ),
        ReasonCode.MARKER_INFO_INVALID_ID: (
            "markers_info entries for 'id' must be a str or None, "
            "invalid for {n_invalid_markers} marker(s)."
        ),
        ReasonCode.MARKER_INFO_INVALID_REF: (
            "markers_info entries for 'ref' must be a non-empty str, "
            "invalid for {n_invalid_markers} marker(s)."
        ),
        ReasonCode.MARKER_INFO_INVALID_ALT: (
            "markers_info entries for 'alt' must be a list of str, "
            "invalid for {n_invalid_markers} marker(s)."
        ),
        ReasonCode.DUPLICATED_MARKERS: (
            "Duplicated marker id(s) in columns: {duplicated!r}."
        ),
    }

    class _Error(TypedDict, total=False):
        reason: Required["InvalidVcfDataError.ReasonCode"]
        provided_type: type
        use_bases: bool
        invalid_values: set
        missing_columns: list
        marker_ids: list
        duplicated: list

    errors: list["_Error | dict[str , Any]"]
    """The list of extra information related to each detected errors.

    List of :class:`dict` with contextual keys depending on the :attr:`reason`:
        - ``reason``: :class:`ReasonCode`
        - ``provided_type``: :class:`type`
        - ``use_bases``: :class:`bool`
        - ``invalid_values``: :class:`set`
        - ``missing_columns``: :class:`list`
        - ``marker_ids``: :class:`list`
        - ``duplicated``: :class:`list`
    """

    class _Extra(TypedDict):
        errors_list: list["InvalidVcfDataError._Error | dict[str , Any]"]

    extra: "_Extra | dict[str, Any]"
    """Extra information related to the error.

    :class:`dict` with key ``errors_list`` containing the list of errors related
    to VCF data (ie. :attr:`errors`).
    """

    @classmethod
    def _format_err_msg(cls, errors: Sequence["_Error | dict[str , Any]"]):
        err_msgs = []
        for error in errors:
            reason = error["reason"]
            str_extra = {
                "provided_type_str": (
                    _type_fullname(error["provided_type"])
                    if "provided_type" in error
                    else None
                ),
                "n_invalid_markers": (
                    len(error["marker_ids"]) if "marker_ids" in error else None
                ),
                "sorted_invalid_values": (
                    sorted(error["invalid_values"])
                    if "invalid_values" in error
                    else None
                ),
            }
            str_extra = {k: v for k, v in str_extra.items() if v is not None}
            err_msgs.append(cls._MESSAGES[reason].format(**error, **str_extra))
        return err_msgs

    def __init__(
        self,
        errors: Sequence["InvalidVcfDataError._Error | dict[str , Any]"],
    ):
        self.errors = list(errors)
        err_msgs = self._format_err_msg(errors)
        message = "Data is not valid VCF data:\n- " + "\n- ".join(err_msgs)
        super().__init__(message=message, extra={"errors_list": self.errors})


def read_vcf(
    vcf_file: str | Path,
    use_bases: bool = False,
    marker_id_format: _MARKER_ID_FORMAT = "ref_alt",
    strict_gt: bool = False,
) -> pd.DataFrame:
    """Load a VCF file as a :class:`pandas.DataFrame`.

    Parses ``vcf_file`` with :class:`cyvcf2.cyvcf2.VCF` and returns one row per sample
    and one column per marker, with special :attr:`pandas.DataFrame.attrs` set
    on the returned :class:`pandas.DataFrame` so other "VCF data specific" functions
    can later make use of the markers information (eg. :func:`reindex_vcf_data`,
    :func:`build_vcf_encoding_map`).

    **Only diploid genotypes are supported.**

    Parameters
    ----------
    vcf_file :
        Path to the VCF file to read.
    use_bases :
        Whether genotype values should be encoded as base-pair strings
        (``True``) or as integers (``False``, the default).

        - If ``True``, values are strings formated as ``"REF/ALT"`` (unphased) or
          ``"REF|ALT"`` (phased), eg. ``"C/A"``, ``"T|A"``.
        - If ``False``, values are integers:

          - ``0``: homozygous reference allele (ie. ``0/0``, ``0|0``)
          - ``1``: heterozygous (ie. ``1/0``, ``0|1``...) and partial missing
            value if ``strict_gt`` is ``False``
          - ``2``: homozygous alternative allele (ie. ``1/1``, ``1|1``)
          - ``3``: missing values and partial missing value if ``strict_gt``
            is ``True``

    marker_id_format :
        Format used to build each marker's column id. See
        :const:`utils.MARKER_ID_FORMATS` for the available formats.
    strict_gt :
        :class:`cyvcf2.cyvcf2.VCF`'s argument to controls how partially missing
        genotypes are handled:

        - ``False`` (the default): partial missing values are encoded as ``1``
        - ``True``: partial missing values are encoded as ``3``.

        Has no effect when ``use_bases`` is ``True``.


    Returns
    -------
        A :class:`pandas.DataFrame` of genotypes, indexed by sample id, with
        one column per marker (named according to ``marker_id_format``), and
        with :attr:`pandas.DataFrame.attrs` set to:

        - ``use_bases``: the ``use_bases`` value passed in.
        - ``markers_info``: a :class:`dict` mapping each column to a
          :class:`dict` with keys ``chrom``, ``pos``, ``id``, ``ref`` and
          ``alt``.

    Raises
    ------
    FileNotFoundError
        If ``vcf_file`` does not exist.
    DuplicatedMarkerIDsError
        If the marker ids built from ``marker_id_format`` are not unique.
    Exception
        If parsing the VCF file with :class:`cyvcf2.cyvcf2.VCF` fails.

    Warning
    -------
        - In addition to regular Python exceptions (caught and re-raised
          here with an added note), a badly formatted VCF file can trigger a
          fatal parsing error in ``cyvcf2`` that aborts the whole Python
          process at the lower level. Such fatal errors **cannot** be caught
          with a ``try``/``except`` block.

        - Depending on the installed ``cyvcf2`` version, phased missing
          values may be loaded as unphased: ``.|.`` becomes ``./.``, and
          ``.|1`` becomes eg. ``./A`` (see
          `cyvcf2#331 <https://github.com/brentp/cyvcf2/issues/331>`_).

    See Also
    --------
        :const:`utils.MARKER_ID_FORMATS` available `marker_id_format`
        :func:`utils.build_marker_ids` : Builds each individual marker id.
        :func:`reindex_vcf_data` : Rebuilds the columns index with a new
        ``marker_id_format``.
        :func:`validate_vcf_data` : Validates a VCF DataFrame such as the one returned
        here.
        :func:`build_vcf_encoding_map` : Builds a one-hot encoding map from a VCF
        DataFrame.
    """
    # Limitation, on badly formatted VCF parsing can fail with "Fatal error"
    # which make handling of such error impossible.

    if not Path(vcf_file).exists():
        raise FileNotFoundError(f"VCF file not found: {vcf_file}")

    try:
        vcf = VCF(fname=str(vcf_file), gts012=True, strict_gt=strict_gt)
    except Exception as e:
        # NOTE: in addition of raising regular Python exceptions (caught here),
        # cyvcf2 can also have fatal parsing error and abort which crashes the
        # whole Python process (and cannot be caught with try/except, since it
        # happens at the C level below the interpreter).
        # If this is a problem in practice, the fix would be to run this VCF()
        # call (and maybe this whole function) in a subprocess to keep the main
        # one safe (but this may not be easy to do properly for all Linux, Windows, Mac)
        e.add_note(
            "Note: Pasring of the VCF file with 'cyvcf2' failed. "
            "Be sure your VCF file is valid."
        )
        raise
    samples = vcf.samples

    genotypes = []
    markers_ids = []
    markers_info = {}

    if use_bases:
        gt_to_use = "gt_bases"
        dtype_to_use = str
    else:
        gt_to_use = "gt_types"
        dtype_to_use = np.int8

    for variant in vcf:
        marker_info = {
            "chrom": variant.CHROM,
            "pos": variant.POS,
            "id": variant.ID,
            "ref": variant.REF,
            "alt": variant.ALT,
        }
        genotypes.append(getattr(variant, gt_to_use).copy())
        markers_id = build_marker_ids(**marker_info, marker_id_format=marker_id_format)
        markers_info[markers_id] = marker_info
        markers_ids.append(markers_id)

    if len(markers_ids) != len(set(markers_ids)):
        duplicated_ids = [
            m_id for m_id, count in Counter(markers_ids).items() if count > 1
        ]
        raise DuplicatedMarkerIDsError(duplicated_ids)

    genotypes = (
        np.array(genotypes, dtype=dtype_to_use)
        if genotypes
        else np.empty((0, len(samples)), dtype=dtype_to_use)
    )

    vcf_data = pd.DataFrame(genotypes.T, index=samples, columns=markers_ids)
    vcf_data.attrs = {
        "use_bases": use_bases,
        "markers_info": markers_info,
    }
    return vcf_data


class _ValidationResults(TypedDict):
    errors: list[InvalidVcfDataError._Error]
    warnings: list[str]


class _ValidationResultsStrings(TypedDict):
    errors: list[str]
    warnings: list[str]


@overload
def validate_vcf_data(
    vcf_data: object, deep: bool = ..., as_strings: Literal[True] = ...
) -> _ValidationResultsStrings: ...
@overload
def validate_vcf_data(
    vcf_data: object, deep: bool = ..., as_strings: Literal[False] = ...
) -> _ValidationResults: ...


def validate_vcf_data(
    vcf_data: object,
    deep=True,
    as_strings: bool = True,
) -> _ValidationResults | _ValidationResultsStrings:
    """Validate the provided data are correct VCF data.

    Checks ``vcf_data`` against the structure expected by other "VCF data
    specific" functions (eg. :func:`reindex_vcf_data`, :func:`build_vcf_encoding_map`):

    Detected issues are not raised as exceptions but collected and returned, so that
    all problems with the data can be reported at once.

    With ``deep=False``, only a limited number of check are performed:

    - ``vcf_data`` is a :class:`pandas.DataFrame`.
    - :attr:`pandas.DataFrame.attrs` ``'use_bases'`` is present.
    - :attr:`pandas.DataFrame.attrs` ``'use_bases'`` is a :class:`bool`.
    - :attr:`pandas.DataFrame.attrs` ``'markers_info'`` is present.

    This minimal validation is used internally (eg. by :func:`reindex_vcf_data`,
    :func:`build_vcf_encoding_map`) to allow custom-made "VCF data" to be used
    more easily (at the user's own risk) with those functions, and for speed.

    With ``deep=True`` (the default), additional checks are performed:

    - Data values are consistent with ``attrs['use_bases']`` (eg. all integers in
      ``{0, 1, 2, 3}`` if ``False``, or well-formated strings if ``True``).
    - ``attrs['markers_info']`` is a :class:`dict`.
    - ``attrs['markers_info']`` has an entry for every df's column. Entries with no
      matching column are reported as a warning rather than an error.
    - Each ``markers_info`` entry is itself a :class:`dict` with the expected
      keys (``chrom``, ``pos``, ``id``, ``ref``, ``alt``), and each key's value
      has the expected type (eg. ``pos`` is a positive :class:`int`, ``alt`` is a
      :class:`list` of :class:`str`).
    - Duplicated sample ids in the row index are reported as a warning.
    - Duplicated marker ids in the columns are reported as an error.

    This "deep" validation can be used directly by users to check the validity of
    their own custom-made VCF data.

    Parameters
    ----------
    vcf_data :
        The object to validate, expected to be a VCF genotype DataFrame such as
        returned by :func:`read_vcf`.
    deep :
        Whether to perform the additional checks described above. Defaults to ``True``.
    as_strings :
        Whether to format the detected errors as human-readable strings
        (``True``, the default) or leave them as structured
        :class:`InvalidVcfDataError._Error` dicts (``False``), for programmatic
        use (eg. to build an :class:`InvalidVcfDataError`).

    Returns
    -------
        list of erros either as :class:`str` or as :class:`InvalidVcfDataError._Error`
        dicts.
    """
    invalid_reason = InvalidVcfDataError.ReasonCode
    errors: list[InvalidVcfDataError._Error] = []
    warnings = []

    if not isinstance(vcf_data, pd.DataFrame):
        errors.append(
            {
                "reason": invalid_reason.NOT_A_DATAFRAME,
                "provided_type": type(vcf_data),
            }
        )
        if as_strings:
            errors = InvalidVcfDataError._format_err_msg(errors)  # noqa: SLF001
        return {"errors": errors, "warnings": warnings}

    attrs = vcf_data.attrs

    if "use_bases" not in attrs:
        errors.append({"reason": invalid_reason.USE_BASES_ATTR_MISSING})
    if "markers_info" not in attrs:
        errors.append({"reason": invalid_reason.MARKERS_INFO_ATTR_MISSING})

    # ---- use_bases ----
    use_bases = attrs.get("use_bases")
    if "use_bases" in attrs and not isinstance(use_bases, bool):
        errors.append(
            {
                "reason": invalid_reason.USE_BASES_INVALID_TYPE,
                "provided_type": type(use_bases),
            }
        )
        use_bases = None

    if not deep:
        if as_strings:
            errors = InvalidVcfDataError._format_err_msg(errors)  # noqa: SLF001
        return {"errors": errors, "warnings": warnings}

    if use_bases is not None and not vcf_data.empty:
        unique_values = set(np.asarray(vcf_data).ravel())
        if use_bases:
            valid_base_pattern = re.compile(r"^[ACGTN.]+([/|][ACGTN.]+)*$")
            invalid_values = {
                v
                for v in unique_values
                if not isinstance(v, str) or not valid_base_pattern.match(v)
            }
            if invalid_values:
                errors.append(
                    {
                        "reason": invalid_reason.VALUES_INCONSISTENT_WITH_USE_BASES,
                        "use_bases": use_bases,
                        "invalid_values": invalid_values,
                    }
                )
        else:
            expected_non_base_values = {0, 1, 2, 3}
            invalid_values = {
                v for v in unique_values if v not in expected_non_base_values
            }
            if invalid_values:
                errors.append(
                    {
                        "reason": invalid_reason.VALUES_INCONSISTENT_WITH_USE_BASES,
                        "use_bases": use_bases,
                        "invalid_values": invalid_values,
                    }
                )

    # ---- marker_info ----
    markers_info = attrs.get("markers_info")
    if "markers_info" in attrs and not isinstance(markers_info, dict):
        errors.append(
            {
                "reason": invalid_reason.MARKERS_INFO_INVALID_TYPE,
                "provided_type": type(markers_info),
            }
        )

        markers_info = None

    if markers_info is not None:
        columns_set = set(vcf_data.columns)
        markers_info_keys_set = set(markers_info.keys())

        missing_in_markers_info = columns_set - markers_info_keys_set
        if missing_in_markers_info:
            errors.append(
                {
                    "reason": invalid_reason.MARKERS_INFO_MISSING_COLUMNS,
                    "missing_columns": list(missing_in_markers_info),
                }
            )

        extra_in_markers_info = markers_info_keys_set - columns_set
        if extra_in_markers_info:
            warnings.append(
                "attrs['markers_info'] has entries not present in columns: "
                f"{sorted(extra_in_markers_info)}."
            )

        invalid_key_to_reason_code = {
            "type": invalid_reason.MARKER_INFO_INVALID_TYPE,
            "missing_keys": invalid_reason.MARKER_INFO_MISSING_KEYS,
            "chrom": invalid_reason.MARKER_INFO_INVALID_CHROM,
            "pos": invalid_reason.MARKER_INFO_INVALID_POS,
            "id": invalid_reason.MARKER_INFO_INVALID_ID,
            "ref": invalid_reason.MARKER_INFO_INVALID_REF,
            "alt": invalid_reason.MARKER_INFO_INVALID_ALT,
        }
        invalid_merker_ids = {key: [] for key in invalid_key_to_reason_code}
        required_keys = {"chrom", "pos", "id", "ref", "alt"}
        for marker_id, marker_info in markers_info.items():
            if not isinstance(marker_info, dict):
                invalid_merker_ids["type"].append(marker_id)
                continue

            missing_keys = required_keys - marker_info.keys()
            if missing_keys:
                invalid_merker_ids["missing_keys"].append(marker_id)

            chrom, pos, m_id, ref, alt = (
                marker_info.get("chrom", _MISSING),
                marker_info.get("pos", _MISSING),
                marker_info.get("id", _MISSING),
                marker_info.get("ref", _MISSING),
                marker_info.get("alt", _MISSING),
            )
            if chrom is not _MISSING and (not isinstance(chrom, str) or not chrom):
                invalid_merker_ids["chrom"].append(marker_id)
            if pos is not _MISSING and (
                isinstance(pos, bool) or not isinstance(pos, int) or pos <= 0
            ):
                invalid_merker_ids["pos"].append(marker_id)
            if m_id is not _MISSING and (
                m_id is not None and not isinstance(m_id, str)
            ):
                invalid_merker_ids["id"].append(marker_id)
            if ref is not _MISSING and (not isinstance(ref, str) or not ref):
                invalid_merker_ids["ref"].append(marker_id)
            if alt is not _MISSING and (
                not isinstance(alt, list) or not all(isinstance(a, str) for a in alt)
            ):
                invalid_merker_ids["alt"].append(marker_id)

        for key, reason in invalid_key_to_reason_code.items():
            if invalid_merker_ids[key]:
                errors.append({"reason": reason, "marker_ids": invalid_merker_ids[key]})

    if vcf_data.index.has_duplicates:
        dup_samples = vcf_data.index[vcf_data.index.duplicated()].unique().tolist()
        warnings.append(f"Duplicated sample(s) in row index: {dup_samples}.")

    if vcf_data.columns.has_duplicates:
        dup_markers = vcf_data.columns[vcf_data.columns.duplicated()].unique().tolist()
        errors.append(
            {
                "reason": invalid_reason.DUPLICATED_MARKERS,
                "duplicated": dup_markers,
            }
        )

    if as_strings:
        errors = InvalidVcfDataError._format_err_msg(errors)  # noqa: SLF001

    return {"errors": errors, "warnings": warnings}


def reindex_vcf_data(
    vcf_data: pd.DataFrame,
    marker_id_format: _MARKER_ID_FORMAT,
) -> pd.DataFrame:
    """Rebuilds the column index of a VCF DataFrame for the given ``marker_id_format``.

    Return **a copy** of the given ``vcf_data`` with the column index (and the
    corresponding ``attrs['markers_info']``) recomputed for the specified
    ``marker_id_format`` using the marker information already stored in
    ``attrs['markers_info']``, formatted according to ``marker_id_format`` (see
    :func:`utils.build_marker_ids`, :const:`utils.MARKER_ID_FORMATS`).

    ``vcf_data`` **is not mutated by this function.**

    Parameters
    ----------
    vcf_data :
        VCF DataFrame as returned by :func:`read_vcf`. Must have ``attrs['use_bases']``
        and ``attrs['markers_info']`` set correctly for desired ``marker_id_format``.

        .. note::
            Only a minimal validation is performed using :func:`validate_vcf_data`
            with ``deep=False``.

            This let users use this function with custom made DataFrame without having
            to build ``attrs['markers_info']`` entirly but only with information needed
            for the disired ``marker_id_format``.

            Be aware that, in such case, if the custom data are malformed this function
            can fail unnexpectedly or return wrong results. Therfore, it is recommended
            to first validate the data with :func:`validate_vcf_data` and ``deep=True``
            (the default) to prevent such unnexpected errors.

    marker_id_format :
        Marker id format to rebuild the columns with. See
        :const:`utils.MARKER_ID_FORMATS` for the available formats.

    Returns
    -------
        A new VCF DataFrame with the same data as ``vcf_data``, but with columns
        (and ``attrs['markers_info']``) rebuilt using ``marker_id_format``.

    Raises
    ------
    InvalidVcfDataError
        If ``vcf_data`` do not pass basic validation (see :func:`validate_vcf_data`).
    DuplicatedMarkerIDsError
        If the new marker ids built from ``marker_id_format`` are not unique.
    Exception
        If ``vcf_data`` is malformed. In such case run :func:`validate_vcf_data` (with
        ``deep = True``, the default) for more details about data validity.

    See Also
    --------
        :const:`utils.MARKER_ID_FORMATS` available `marker_id_format`
        :func:`utils.build_marker_ids` : Builds each individual marker id.
        :func:`read_vcf` : Builds the original VCF genotype DataFrame.
        :func:`validate_vcf_data` : Performs the validation used here.
    """
    validation = validate_vcf_data(vcf_data, deep=False, as_strings=False)
    if validation["errors"]:
        raise InvalidVcfDataError(validation["errors"])

    markers_info = vcf_data.attrs["markers_info"]

    new_columns = []
    new_markers_info = {}
    for old_column_name in vcf_data.columns:
        marker_info = markers_info[old_column_name]
        new_column_name = build_marker_ids(
            **marker_info, marker_id_format=marker_id_format
        )
        new_columns.append(new_column_name)
        new_markers_info[new_column_name] = marker_info

    if len(new_columns) != len(set(new_columns)):
        duplicated_ids = [
            m_id for m_id, count in Counter(new_columns).items() if count > 1
        ]
        # explicit error because "markers_info" would loose information
        raise DuplicatedMarkerIDsError(duplicated_ids)

    new_vcf_data = vcf_data.copy()
    new_vcf_data.columns = new_columns
    new_vcf_data.attrs = {
        **vcf_data.attrs,
        "markers_info": new_markers_info,
    }
    return new_vcf_data


_REQUIRED_ARGS_build_marker_ids: dict[_MARKER_ID_FORMAT, tuple[str, ...]] = {
    "id": ("id",),
    "pos": ("chrom", "pos"),
    "ref_alt": ("chrom", "pos", "ref", "alt"),
    "alleles": ("chrom", "pos", "ref", "alt"),
}


def build_marker_ids(
    chrom: str | _Missing = _MISSING,
    pos: int | _Missing = _MISSING,
    id: str | None | _Missing = _MISSING,
    ref: str | _Missing = _MISSING,
    alt: list[str] | _Missing = _MISSING,
    marker_id_format: _MARKER_ID_FORMAT = "ref_alt",
) -> str:
    """Build a marker id string from marker information, according to a given format.

    Combines the provided marker attributes into a single string identifier,
    according to ``marker_id_format``. Which arguments are required depends on
    the chosen format:

    - **"id"**: returns ``id`` itself, or ``"."`` if ``id`` is falsy (eg. ``None``).
    - **"pos"**: returns ``"{chrom}@{pos}"``.
    - **"ref_alt"**: returns ``"{chrom}@{pos}_{ref}_{alt}"``, with multiple ``alt``
      values joined by ``"-"``.
    - **"alleles"**: returns ``"{chrom}@{pos}_{alleles}"``, with ``ref`` and ``alt``
      values combined, sorted, and joined by ``"-"``.

    Parameters
    ----------
    chrom :
        Chromosome name. Required for ``marker_id_format`` ``"pos"``,
        ``"ref_alt"`` and ``"alleles"``.
    pos :
        Position on the chromosome. Required for ``marker_id_format`` ``"pos"``,
        ``"ref_alt"`` and ``"alleles"``.
    id :
        Marker id (eg. from a VCF ``ID`` field), or ``None`` if not available.
        Required for ``marker_id_format`` ``"id"``.
    ref :
        Reference allele. Required for ``marker_id_format`` ``"ref_alt"`` and
        ``"alleles"``.
    alt :
        List of alternative alleles. Required for ``marker_id_format``
        ``"ref_alt"`` and ``"alleles"``.
    marker_id_format :
        Format of the id to build. One of ``"pos"``, ``"ref_alt"``, ``"id"`` or
        ``"alleles"``. Those are included in :const:`utils.MARKER_ID_FORMATS`.


    Returns
    -------
        The built marker id string.

    Raises
    ------
    UnexpectedMarkerIdFormatError
        If ``marker_id_format`` is not one of the expected values.
    TypeError
        If one or more required argument(s) for the given ``marker_id_format``
        is missing (ie. left as its default sentinel value).

    See Also
    --------
    :const:`utils.MARKER_ID_FORMATS` available `marker_id_format`
    :func:`read_vcf`: Uses this function to build the columns index.
    :func:`reindex_vcf_data`: Rebuilds columns index with a new ``marker_id_format``.

    Examples
    --------
    .. jupyter-kernel::
       :id: build_marker_ids-example

    .. jupyter-execute::

        from deepgenocompress.utils import build_marker_ids

        # get list of accepted marker_id_format
        from deepgenocompress.utils import MARKER_ID_FORMATS

        marker_info = {
          "chrom": "chr1",
          "pos": 1234,
          "id": "marker_1",
          "ref": "T",
          "alt": ["C", "A"]
        }
        for format in MARKER_ID_FORMATS:
            id = build_marker_ids(**marker_info, marker_id_format=format)
            print(f"{format}: {id}")
    """
    if marker_id_format not in _REQUIRED_ARGS_build_marker_ids:
        raise UnexpectedMarkerIdFormatError(marker_id_format)

    values = {"chrom": chrom, "pos": pos, "id": id, "ref": ref, "alt": alt}
    missing = [
        arg
        for arg in _REQUIRED_ARGS_build_marker_ids[marker_id_format]
        if values[arg] is _MISSING
    ]
    if missing:
        raise TypeError(
            f"{build_marker_ids.__name__}() missing {len(missing)} required "
            f"argument(s): {", ".join(missing)}"
        )

    chrom_pos = f"{chrom}@{pos}"

    match marker_id_format:
        case "id":
            id = cast(str | None, id)
            return id if id else "."  # cyvcf2.variant.ID is None if ID is . in vcf file
        case "pos":
            return chrom_pos
        case "ref_alt":
            alt = cast(list[str], alt)
            return f"{chrom_pos}_{ref}_{'-'.join(alt)}"
        case "alleles":
            ref = cast(str, ref)
            alt = cast(list[str], alt)
            return f"{chrom_pos}_{'-'.join(sorted([ref, *alt]))}"
        case _:
            raise UnexpectedMarkerIdFormatError(marker_id_format)


def build_vcf_encoding_map(vcf_data: pd.DataFrame) -> dict[Any, list[float]]:
    """Build an encoding map for the genotype data comming from a VCF file.

    Returns a mapping from genotype values found in ``vcf_data`` to their
    one-hot encoding vectors.

    If ``use_bases`` is ``False``, a fixed encoding map is returned regardless of
    the actual values in ``vcf_data``:

    - ``0``: ``[1.0, 0.0, 0.0]``
    - ``1``: ``[0.0, 1.0, 0.0]``
    - ``2``: ``[0.0, 0.0, 1.0]``
    - ``3``: ``[0.0, 0.0, 0.0]``

    If ``vcf_data.attrs['use_bases']`` is ``True``, the map is built dynamically
    from the unique bases present in ``vcf_data`` (eg. ``"A/T"``).
    Strings representing the same pair of alleles regardless of allele
    order or phasing (eg. ``"A/T"``, ``"T/A"``, ``"A|T"``, ``"T|A"``) are grouped
    and assigned the same one-hot vector. Missing genotypes (``"./."`` and
    ``".|."``) are always mapped to a zero vector.

    Parameters
    ----------
    vcf_data :
        VCF DataFrame as returned by :func:`read_vcf`. Must have ``attrs['use_bases']``
        set to a :type:`bool` and ``attrs['markers_info']`` to any value.

        .. note::
            Only a minimal validation is performed on ``vcf_data`` using
            :func:`validate_vcf_data` with ``deep=False``.

            This let users use this function with custom made DataFrame by simply
            setting DataFrame's ``attrs['use_bases']`` to its correct value and
            ``attrs['markers_info']`` any value (eg. to ``NULL``, as this information
            is not used by this function).

            Be aware that, in such case, if the custom data are malformed this function
            can fail unnexpectedly or return wrong results. Therfore, it is recommended
            to first validate the data with :func:`validate_vcf_data` and ``deep=True``
            (the default) to prevent such unnexpected errors.

    Returns
    -------
        A dictionary mapping each unique allele to its one-hot encoded vector.

    Raises
    ------
    InvalidVcfDataError
        If ``vcf_data`` do not pass basic validation (see :func:`validate_vcf_data`).
    Exception
        If ``use_bases`` is ``True`` and ``vcf_data`` is malformed (eg. not splittable
        into two alleles). In such case run :func:`validate_vcf_data` (with
        ``deep = True``, the default) for more details about data validity.

    See Also
    --------
    :func:`read_vcf`: Builds a VCF DataFrame from a VCF file.
    :func:`validate_vcf_data`: Performs validation on VCF DataFrame.

    Examples
    --------
    .. jupyter-kernel::
       :id: build_vcf_encoding_map-example

    .. jupyter-execute::

        import pandas as pd
        from pprint import pprint
        from deepgenocompress import build_vcf_encoding_map

        vcf_data = pd.DataFrame(["A/T", "C|G"])
        vcf_data.attrs["use_bases"] = True
        vcf_data.attrs["markers_info"] = None

        pprint(build_vcf_encoding_map(vcf_data))
    """
    validation = validate_vcf_data(vcf_data, deep=False, as_strings=False)
    if validation["errors"]:
        raise InvalidVcfDataError(validation["errors"])

    if vcf_data.attrs["use_bases"]:
        unique_values = set(np.asarray(vcf_data).ravel())

        try:
            unique_bases_combinations = sorted(
                {
                    tuple(sorted([a + "/" + b, b + "/" + a, a + "|" + b, b + "|" + a]))
                    for v in unique_values
                    for a, b in [v.replace("|", "/").split("/")]
                    if (a, b) != (".", ".")
                }
            )
        except Exception as e:
            e.add_note(
                "Note: some genotype values may be malformed. Run "
                "`validate_vcf_data(vcf_data)` to get more details about data validity."
            )
            raise

        n = len(unique_bases_combinations)
        encoding_map = {}
        for i, alleles in enumerate(unique_bases_combinations):
            encoding = [1.0 if i == j else 0.0 for j in range(n)]
            for allele in alleles:
                encoding_map[allele] = encoding

        encoding_map["./."] = [0] * max(n, 1)
        encoding_map[".|."] = [0] * max(n, 1)
        return encoding_map

    return {
        0: [1.0, 0.0, 0.0],
        1: [0.0, 1.0, 0.0],
        2: [0.0, 0.0, 1.0],
        3: [0.0, 0.0, 0.0],
    }
