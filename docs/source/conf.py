import os
import sys
# works both locally (docs/source/) and on RTD (repo root)
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../src')
))

project = 'CETRA'
copyright = '2025, Leigh Smith'
author = 'Leigh Smith'
release = 'v1.04'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# pycuda and CUDA are not available on ReadTheDocs build servers
autodoc_mock_imports = ['pycuda', 'pycuda.compiler', 'pycuda.driver',
                        'pycuda.tools', 'cuda']

autoclass_content = 'both'

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

templates_path = ['_templates']
exclude_patterns = ['_build']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
