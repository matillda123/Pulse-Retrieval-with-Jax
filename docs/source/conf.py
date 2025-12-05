project = 'PulseRetrievalwithJAX'
copyright = '2025, Till Stehling'
author = 'Till Stehling'
release = '2025'

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../src'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',   # for Google/NumPy style docstrings
    'sphinx.ext.autosummary',
    'nbsphinx',
    'sphinx.ext.mathjax'
#    'myst_nb',
#    'sphinx.ext.mathjax',
]

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []

autodoc_member_order = 'bysource'

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']



nbsphinx_section_numbers = False
nbsphinx_allow_errors = True
nbsphinx_prolog = r"""
.. raw:: html

    <style>
        .nbsphinx-nav { display: none; }
    </style>
"""

