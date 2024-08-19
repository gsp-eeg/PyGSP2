# -*- coding: utf-8 -*-
"""Test suite for the docstrings of the pygsp2 package.

This module tests the docstrings of the pygsp2 package, ensuring that they
are correctly formatted and that the examples in the docstrings execute
without errors.

The test suite uses the doctest module to automatically run the code examples
found in the docstrings of Python files and reStructuredText files.
"""

import doctest
import os
import unittest


def gen_recursive_file(root, ext):
    """
    Generate a list of files with a specific extension within a directory tree.

    Parameters
    ----------
    root : str
        The root directory to start the search.
    ext : str
        The file extension to search for (e.g., '.py' for Python files).

    Yields
    ------
    str
        The full path to each file found with the specified extension.
    """
    for root, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(ext):
                yield os.path.join(root, name)


def test_docstrings(root, ext, setup=None):
    """
    Create a DocFileSuite to test the docstrings in files with a given extension.

    Parameters
    ----------
    root : str
        The root directory to start the search.
    ext : str
        The file extension to search for (e.g., '.py' for Python files).
    setup : function, optional
        A setup function to initialize the environment before running the tests.

    Returns
    -------
    doctest.DocFileSuite
        A test suite for running doctests on the specified files.
    """
    files = list(gen_recursive_file(root, ext))
    return doctest.DocFileSuite(*files, setUp=setup, tearDown=teardown,
                                module_relative=False)


def setup(doctest):
    """
    Set up the testing environment for doctests.

    This function imports necessary modules and assigns them to the doctest
    global namespace to be used in the docstring examples.

    Parameters
    ----------
    doctest : doctest.DocTest
        The doctest object that will be executed.
    """
    import numpy

    import pygsp2

    doctest.globs = {
        'graphs': pygsp2.graphs,
        'filters': pygsp2.filters,
        'utils': pygsp2.utils,
        'np': numpy,
    }


def teardown(doctest):
    """
    Clean up the testing environment after doctests.

    This function closes all matplotlib figures to avoid warnings and save memory.

    Parameters
    ----------
    doctest : doctest.DocTest
        The doctest object that was executed.
    """
    import pygsp2

    pygsp2.plotting.close_all()


# Docstrings from API reference.
suite_reference = test_docstrings('pygsp2', '.py', setup)

# Docstrings from tutorials. No setup to not forget imports.
suite_tutorials = test_docstrings('.', '.rst')

# Combine the test suites into a single test suite.
suite = unittest.TestSuite([suite_reference, suite_tutorials])
