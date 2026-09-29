# Changelog

All notable changes to this project will be documented in this file.

the format is based on [keep a changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `CHANGELOG.md`: All notable changes to this project will be documented in this file.
- `utils.infer_marker_id_format` to infer the `marker_id_format` of a marker id string.

### Changed

- `CompressionModel.compress_vcf_file`: `marker_id_format` now auto-detects
  `marker_id_format` from `training_markers_index` when available, falling back to
  `"ref_alt"` if `training_markers_index` are not available.

## [0.1.0] - 2026-09-14

First release of `deepgenocompress`, a continuation of the
[ConvCGP](https://github.com/tanzilamohita/ConvCGP) work by @tanzilamohita, restructured
as an installable Python package (requires Python >= 3.11).

### Added

- `CompressionModel`: class for compressing genome-wide polymorphism data with
  autoencoders. It handles data chunking, autoencoder construction, training and
  compression.
  - Can be initialised from a NumPy array, a `pandas.DataFrame` (`from_dataframe`) or a
    VCF file (`from_vcf_file`), and can compress `DataFrame` inputs directly.
  - Supports layer sizes that are not compatible with the number of markers by padding the
    data (a warning is raised in that case).
  - Exposes `training_markers_index` and validates a custom `encoding_map` against the
    training data and layer sizes.
- `AutoencoderModels`: container class holding the autoencoder and encoder models. Both
  can be built with an arbitrary number of layers and layer sizes.
- SNP encoding:
  - `build_one_hot_encoding_map` and `encode_snp_array`, with customisable encoding maps
    (one-hot by default).
  - Missing values (including NA-like values such as `np.nan` and `None`) are explicitly
    encoded, by default as vectors of zeros.
  - Non-string arrays are supported.
- VCF support: `read_vcf`, `build_vcf_encoding_map`, `reindex_vcf_data` and
  `validate_vcf_data` (plain and `.vcf.gz` files).
- `utils` module:
  - `possible_first_layer_sizes`: lists first-layer sizes compatible with the encoded
    training data.
  - `encoding_size`: returns the length of the encoded vectors of an encoding map.
  - `build_marker_ids` and the `MARKER_ID_FORMATS` constant.
  - Example data files shipped with the package.
- Custom exceptions (`deepgenocompress.exceptions`), such as `InvalidEncodingMapError`,
  `LayerSizesConfigurationError`, `ModelStateError`, `IncompatibleDataError` and
  `CompressionModelConfigurationError`. Long collections are truncated in error messages.
- Custom warnings (`deepgenocompress.warnings`), such as `UnmappedValuesWarning` and
  `LessThanOneAlleleChunksWarning`.
- Optional Keras backend extras: `tensorflow`, `torch` and `jax` (e.g.
  `pip install "deepgenocompress[torch] @ git+..."`).
- Online documentation (Sphinx and Quarto) deployed on GitHub Pages, including a usage
  example covering DataFrames, VCF files and saving the compressed array.
- Nix package and flake for Nix users.

[Unreleased]: https://github.com/ut-biomet/deepgenocompress/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ut-biomet/deepgenocompress/releases/tag/v0.1.0
