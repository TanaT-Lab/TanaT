#!/usr/bin/env python3
"""
Sphinx configuration file.
"""

# pylint: disable=invalid-name,redefined-builtin

import importlib
import pathlib
import sys

source_code = "../../src"
git_url = "https://github.com/TanaT-Lab/TanaT"

this_path = pathlib.Path(__file__).resolve()
sys.path.insert(0, str((this_path.parent / source_code).resolve()))

author = "Arnaud Duvermy, Thomas Guyet"
copyright = "2024-2026, Inria"
project = "TanaT"
html_theme = "pydata_sphinx_theme"
html_logo = "static/logo.png"

html_theme_options = {
    "logo": {
        "image_light": "static/logo.png",
        "image_dark": "static/logo.png",
    },
    "use_edit_page_button": False,
    "show_toc_level": 2,
    "show_nav_level": 2,
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
    "navbar_persistent": ["search-button"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/TanaT-Lab/TanaT",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "LLMs context (llms.txt)",
            "url": "https://tanat-lab.github.io/TanaT/llms.txt",
            "icon": "fa-solid fa-file-lines",
            "type": "fontawesome",
        },
    ],
}

html_static_path = ["static"]
html_css_files = [
    "css/custom.css",
]

autodoc_mock_imports = [
    "numba",
]

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinxcontrib.mermaid",
    "sphinx_gallery.gen_gallery",
]

exclude_patterns = [
    "**.ipynb_checkpoints",
    "user-guide/examples/**",
    "user-guide/tutorials/**",
]


class _ExampleOrder:
    """Sort gallery examples in an explicit pedagogical order.

    Files not listed fall back to alphabetical order at the end.
    To reorder examples, edit the ``_ORDER`` dict below.
    """

    _ORDER = {
        # container: simple → composite
        "sequence.py": 0,
        "trajectory.py": 3,
        # visualization: overview → specific
        "timeline.py": 0,
        "barplot.py": 1,
        "spanplot.py": 2,
        "distribution.py": 3,
        # zeroing: sequence-level → trajectory-level
        "sequence_t0.py": 0,
        "trajectory_t0.py": 1,
        # criterion
        "entity.py": 0,
        "static.py": 1,
        "time.py": 2,
        "pattern.py": 3,
        "length.py": 4,
        "rank.py": 5,
        # metric: entity → sequence → trajectory
        "hamming.py": 0,
        "linear_pairwise.py": 0,
        "edit.py": 1,
        "lcp.py": 2,
        "lcs.py": 3,
        "dtw.py": 4,
        "softdtw.py": 5,
        "chi2.py": 6,
        "aggregation.py": 0,
        "custom.py": 99,  # always in last position
        # clustering: one per algorithm
        "hierarchical.py": 0,
        "pam.py": 1,
        "clara.py": 2,
        # Tutorials:
        #    - Build pool from multiple sources
        "building_pools.py": 0,
        #    - Mimic: end-to-end pipeline
        "explore_a_cohort.py": 1,
        "filter_and_prepare.py": 2,
        "analyse_and_cluster.py": 3,
        "survival_analysis.py": 4,
        #    - MOOC: learning session analysis
        "explore_sessions.py": 0,
        "cluster_sessions.py": 1,
        #    - AEON
        "discretize.py": 0,
        #    - Deep learning
        "phenotype_with_swotted.py": 0,
        "federated_learning.py": 1,
    }

    def __init__(self, src_dir):
        self.src_dir = src_dir

    def __call__(self, filename):
        return (self._ORDER.get(filename, 999), filename)


sphinx_gallery_conf = {
    "examples_dirs": ["user-guide/examples", "user-guide/tutorials"],
    "gallery_dirs": ["user-guide/auto_examples", "user-guide/auto_tutorials"],
    "filename_pattern": r".*\.py$",
    "ignore_pattern": r"__init__\.py",
    "within_subsection_order": _ExampleOrder,
    "download_all_examples": False,
    "plot_gallery": True,
    "show_memory": False,
    "remove_config_comments": True,
    "first_notebook_cell": None,
}

# Suppress "multiple targets found" warnings from autodoc for re-exported symbols
suppress_warnings = ["ref.python", "toc.not_included"]

# Show .. todo:: directives in the rendered output
todo_include_todos = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


def skip(
    _app, _what, name, _obj, would_skip, _options
):  # pylint: disable=too-many-arguments
    """Customize autodoc member skipping."""
    if name == "__init__":
        return False
    return would_skip


def _replace_gallery_index(_app, docname, source):
    """Substitute our hand-written index for sphinx-gallery's auto-generated one.

    sphinx-gallery writes auto_examples/index.rst and auto_tutorials/index.rst
    to disk before Sphinx reads them.  Those files contain an :orphan: tag and a
    thumbnail-grid toctree that breaks sidebar navigation.

    The ``source-read`` event fires after sphinx-gallery has written the files
    but before Sphinx processes them.  We replace the in-memory content with our
    manually maintained versions (examples/index.rst, tutorials/index.rst) which
    carry a clean toctree and no :orphan:.
    """
    _GALLERY_INDEX_MAP = {
        "user-guide/auto_examples/index": pathlib.Path(__file__).parent
        / "user-guide"
        / "examples"
        / "index.rst",
        "user-guide/auto_tutorials/index": pathlib.Path(__file__).parent
        / "user-guide"
        / "tutorials"
        / "index.rst",
    }
    manual = _GALLERY_INDEX_MAP.get(docname)
    if manual is not None and manual.exists():
        source[0] = manual.read_text(encoding="utf-8")


def setup(app):
    """Connect custom event handlers."""
    app.connect("autodoc-skip-member", skip)
    app.connect("source-read", _replace_gallery_index)


def linkcode_resolve(domain, info):
    """Get source links for the linkcode extension."""
    module = info["module"]
    if domain != "py" or not module:
        return None
    top_mod = importlib.import_module(module.split(".")[0])
    mod = importlib.import_module(module)
    top_mod_path = pathlib.Path(top_mod.__file__)
    mod_path = pathlib.Path(mod.__file__)
    subpath = str(mod_path.relative_to(top_mod_path.parent.parent))
    return f"{git_url}/blob/main/src/{subpath}"
