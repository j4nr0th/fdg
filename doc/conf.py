"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "exts"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "fdg"
copyright = "2024, Jan Roth"
author = "Jan Roth"
release = "0.0.1a"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "jupyter_sphinx",
    "pydata_sphinx_theme",
    "hawkmoth",
    "hawkmoth.ext.javadoc",
    "hawkmoth_compat",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# -- Options for Intersphinx -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for Napoleon ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True
napoleon_type_aliases = None


# -- Options for Autodoc -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritence": True,
}
autodoc_member_order = "groupwise"
autodoc_type_aliases = {
    "npt.NDArray": "array",
    "npt.ArrayLike": "array_like",
}

# -- Options for Sphinx Gallery ----------------------------------------------
# https://sphinx-gallery.github.io/stable/index.html
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "reference_url": {
        # The module you locally document uses None
        "fdg": None,
    },
    "image_scrapers": ("matplotlib"),
}

# -- Options for C hawkmoth --------------------------------------------------
# https://hawkmoth.readthedocs.io/en/stable/extension.html#configuration
hawkmoth_root = str((Path(__file__).parent.parent / "src").absolute())
hawkmoth_transform_default = "javadoc"
# The repository root and the cutl submodule header directory are on the
# include path so that headers referencing the cutl submodule
# (`#include <cutl/...>`) parse correctly.
hawkmoth_clang = [
    "--std=c17",
    "-I",
    str((Path(__file__).parent.parent).absolute()),
    "-I",
    str((Path(__file__).parent.parent / "cutl" / "include").absolute()),
]
