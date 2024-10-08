# -*- coding: utf-8 -*-
"""Test suite for the docstrings of the pygsp2 package."""

import doctest
import os
import unittest


def gen_recursive_file(root, ext):
    for root, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(ext):
                yield os.path.join(root, name)


def test_docstrings(root, ext, setup=None):
    files = list(gen_recursive_file(root, ext))
    return doctest.DocFileSuite(*files, setUp=setup, tearDown=teardown, module_relative=False)


def setup(doctest):
    import numpy

    import pygsp2
    doctest.globs = {
        'graphs': pygsp2.graphs,
        'filters': pygsp2.filters,
        'utils': pygsp2.utils,
        'np': numpy,
    }


def teardown(doctest):
    """Close matplotlib figures to avoid warning and save memory."""
    import pygsp2
    pygsp2.plotting.close_all()


# Docstrings from API reference.
suite_reference = test_docstrings('pygsp2', '.py', setup)

# Docstrings from tutorials. No setup to not forget imports.
suite_tutorials = test_docstrings('.', '.rst')

suite = unittest.TestSuite([suite_reference, suite_tutorials])
