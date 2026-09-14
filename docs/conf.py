"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Project information -----------------------------------------------------
https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault(
    "TF_ENABLE_ONEDNN_OPTS", "0"
)  # disable oneDNN to silences its warning

import sys
from importlib.metadata import version as pkg_version

# required by jupyter-sphinx:
package_path = os.path.abspath("..")
os.environ["PYTHONPATH"] = ":".join((package_path, os.environ.get("PYTHONPATH", "")))

# make docs/ importable (for using custom functions if necessary)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


project = "deepgenocompress"
copyright = (
    " 2026, Laboratory of Biometry and Bioinformatics,"
    "Department of Agricultural and Environmental Biology,"
    "Graduate School of Agricultural and Life Science, The University of Tokyo"
)
author = "Tanzila Islam"
release = pkg_version("deepgenocompress")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # generates docs from python docstrings
    "sphinx_autodoc_typehints",
    "sphinx.ext.napoleon",  # enables Google/NumPy style docstrings
    "sphinx.ext.intersphinx",  # enables links to external documentation
    "jupyter_sphinx",  # embeds and executes jupyter cells in docs (for usage examples)
    # "myst_nb",  # Integrates Jupyter Notebooks into Sphinx documentation
    "myst_parser",  # parse .md files
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

# -- Extentions configuration ------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "cyvcf2": ("https://brentp.github.io/cyvcf2/", None),
}
nitpicky = True
nitpick_ignore_regex = {
    # (r"py:class", r".*\._[A-Za-z_]*$"),  # any dotted path ending in ._Something
    (r"py:class", r"^(?:[^.]*\.)*_\w*$"),  # any dotted path ending in ._Something
}

autodoc_default_options = {
    # "members": True,
    "member-order": "groupwise",
    # # "special-members": "__init__",
    # "undoc-members": False,
    # "exclude-members": "__weakref__",
}
autodoc_typehints = "both"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "show_toc_level": 3,
    "use_sidenotes": True,
    "repository_url": "https://github.com/ut-biomet/deepgenocompress",
    "use_repository_button": True,
    "use_download_button": True,
    "use_source_button": True,
    "use_issues_button": True,
}
