# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Add project root, training_scripts, and UQ to path
project_root = Path("../..").resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "training_scripts"))
sys.path.insert(0, str(project_root / "UQ"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mphys-surrogate-model"
copyright = "2025, Emily de Jong, Nipun Gunawardena, Jonas Katona"
author = "Emily de Jong, Nipun Gunawardena, Jonas Katona"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",  # Adds source code links
    "sphinx_autodoc_typehints",  # Better type hint rendering
    "sphinx.ext.mathjax",  # Math formatting
    "myst_parser",  # Markdown support
]
autodoc_preserve_defaults = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Support both .rst and .md files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
