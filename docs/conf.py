"""Sphinx configuration for supervised_phenotype_scoring."""

import sys
from pathlib import Path

# Make the package root importable so autodoc can find the modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Project info ──────────────────────────────────────────────────────────────
project   = "supervised_phenotype_scoring"
copyright = "2026, Annie"
author    = "Annie"
release   = "0.1.0"

# ── Extensions ────────────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",      # pull docstrings from source
    "sphinx.ext.napoleon",     # NumPy / Google docstring styles
    "sphinx.ext.viewcode",     # [source] links in the HTML
    "sphinx.ext.intersphinx",  # cross-link to numpy, scipy, etc.
    "numpydoc",                # richer NumPy docstring rendering
]

# ── autodoc settings ──────────────────────────────────────────────────────────
autodoc_default_options = {
    "members":          True,
    "undoc-members":    False,
    "show-inheritance": True,
    "member-order":     "bysource",
}
autodoc_typehints = "description"
add_module_names  = False

# ── Napoleon (NumPy docstring style) ─────────────────────────────────────────
napoleon_numpy_docstring        = True
napoleon_google_docstring       = False
napoleon_use_param              = True
napoleon_use_rtype              = True
napoleon_preprocess_types       = True

# ── numpydoc ─────────────────────────────────────────────────────────────────
numpydoc_show_class_members = False

# ── intersphinx ───────────────────────────────────────────────────────────────
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
    "scipy":  ("https://docs.scipy.org/doc/scipy", None),
}

# ── Theme ─────────────────────────────────────────────────────────────────────
html_theme       = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "titles_only":      False,
}

# ── Misc ──────────────────────────────────────────────────────────────────────
templates_path   = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
