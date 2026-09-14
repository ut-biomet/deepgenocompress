from collections.abc import Mapping
from enum import Enum
from importlib import resources
from pathlib import Path
from pprint import pformat
from typing import Any


class _Missing(Enum):
    missing = object()


_MISSING = _Missing.missing


def encoding_size(encoding_map: Mapping[Any, list[int] | list[float]]):
    """Return the length of the encoding vectors in an encoding map.

    Minimal function to determined from the first value in ``encoding_map`` from
    the first value. For simplicity this function does not validate the provided
    ``encoding_map`` is correct; use :func:`_validate_encoding_map` first
    if necessary.

    Parameters
    ----------
    encoding_map :
        Mapping from allele values to their encoding vectors.

    Returns
    -------
        Length of the encoding vector associated with the first key in
        ``encoding_map``.

    Raises
    ------
    ValueError
        If ``encoding_map`` is empty.
    """
    try:
        return len(next(iter(encoding_map.values())))
    except StopIteration as e:
        raise ValueError("encoding_map must not be empty") from e


class ExampleFiles:
    """Namespace of example data file paths.

    Examples
    --------
    .. jupyter-kernel::
       :id: ExampleFiles-example

    .. jupyter-execute::

        import pandas as pd

        from deepgenocompress import read_vcf
        from deepgenocompress.utils import ExampleFiles

        with pd.option_context('display.max_rows', 10, 'display.max_columns', 5):
            example_csv = pd.read_csv(ExampleFiles().csv)
            print(example_csv)

            example_vcf = read_vcf(ExampleFiles().vcf)
            print(example_vcf)
    """

    _DATA_DIR = resources.files("deepgenocompress._core.data")
    """Base directory for example data files."""

    @property
    def csv(self) -> Path:
        """CSV example file path.

        SNP genotype table. Each row is a named accession/variety (e.g. ``PURPLE``,
        ``Jefferson``, ``Basmati`` ...), and columns ``V1`` to``V14`` are 14 biallelic
        SNP markers with values being single-nucleotide genotype calls
        (``A``, ``C``, ``G``, ``T``), with ``N`` denoting missing values.
        """
        return Path(str(self._DATA_DIR / "example_1.csv"))

    @property
    def vcf(self) -> Path:
        """VCF example file path.

        Small and simple example VCF (v4.3) file containing 14 biallelic and
        multiallelic SNP variants across 3 synthetic chromosomes and 50 individuals.
        """
        return Path(str(self._DATA_DIR / "example_2.vcf"))

    def __repr__(self):
        files = {
            name: getattr(self, name)
            for name in dir(self)
            if not name.startswith("_") and not callable(getattr(self, name))
        }
        return pformat(files)
