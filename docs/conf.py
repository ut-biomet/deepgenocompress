"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Project information -----------------------------------------------------
https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
"""

import os
import sys
from importlib.metadata import version as pkg_version

# required by jupyter-sphinx:
package_path = os.path.abspath("..")
os.environ["PYTHONPATH"] = ":".join((package_path, os.environ.get("PYTHONPATH", "")))

# make docs/ importable (for using custom functions if necessary)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


project = "deepcgp"
copyright = (
    " 2026, Laboratory of Biometry and Bioinformatics,"
    "Department of Agricultural and Environmental Biology,"
    "Graduate School of Agricultural and Life Science, The University of Tokyo"
)
author = "Tanzila Islam"
release = pkg_version("deepcgp")

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
    "show_toc_level": 2,
    "use_sidenotes": True,
    "repository_url": "https://github.com/ut-biomet/DeepCGP",
    "use_repository_button": True,
    "use_download_button": True,
    "use_source_button": True,
    "use_issues_button": True,
}
