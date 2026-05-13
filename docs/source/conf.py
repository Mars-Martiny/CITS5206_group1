"""Sphinx configuration for the incident classifier project."""

import os
import sys

# Make the project root importable so autodoc can find all modules.
sys.path.insert(0, os.path.abspath("../.."))

project = "Incident Classifier"
copyright = "2026, CITS5206 Group 1"
author = "CITS5206 Group 1"
version = "1.0"
release = "1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]

html_theme_options = {
    "description": "NLP incident classification pipeline",
    "github_user": "CITS5206-group1",
    "fixed_sidebar": True,
}
