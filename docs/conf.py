# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

from importlib.metadata import version as package_version

project = "opytimizer"
copyright = "2020, Gustavo de Rosa"
author = "Gustavo de Rosa"
release = package_version("opytimizer")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
]
autosummary_generate = True
exclude_patterns = ["_build"]
html_theme = "alabaster"
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_member_order = "bysource"
autodoc_typehints_format = "short"
autoclass_content = "both"
