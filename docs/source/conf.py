import datetime
import importlib

# Building the API reference needs the compiled extension. Without a GPU build
# (e.g. on a docs-only CI runner) it is mocked, so only the signatures of the
# pybind11 classes are missing
try:
    importlib.import_module("phipict")
    autodoc_mock_imports = []
except ImportError:
    autodoc_mock_imports = ["phipict._C", "phipict._noise"]

    # Annotations such as ``str | _C.BoundarySampling`` are evaluated at import;
    # let the mocked classes take part in ``X | Y`` unions
    from sphinx.ext.autodoc.mock import _MockObject

    _MockObject.__or__ = lambda self, other: self
    _MockObject.__ror__ = lambda self, other: self

project = "phiPICT"
author = "Jannis Becktepe, Safe Autonomous Systems (SAS), TU Dortmund University"
copyright = (
    f"{datetime.date.today().strftime('%Y')}, "
    "Safe Autonomous Systems (SAS), TU Dortmund University"
)
release = "0.1.0"
version = "0.1.0"

templates_path = ["_templates"]
html_static_path = ["_static"]

html_theme = "sphinx_rtd_theme"
html_logo = "_static/img/logo_lm.png"
html_context = {
    "display_github": True,
    "github_user": "safe-autonomous-systems",
    "github_repo": "phiPICT",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "titles_only": True,
}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

# Inventories are fetched at build time; keep an offline build from stalling.
intersphinx_timeout = 10

autosectionlabel_prefix_document = True
autosummary_generate = True
autosummary_generate_overwrite = True
