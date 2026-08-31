from importlib.metadata import version as package_version

project = "opytimizer"
copyright = "2020, Gustavo de Rosa"
author = "Gustavo de Rosa"
release = package_version("opytimizer")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]
autosummary_generate = True
exclude_patterns = ["_build"]
html_theme = "alabaster"
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_member_order = "bysource"
