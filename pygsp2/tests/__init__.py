# -*- coding: utf-8 -*-
"""Test suite of the PyGSP2 package, broken by modules."""

import unittest

from . import (test_docstrings, test_filters, test_graphs, test_learning, test_plotting,
               test_utils, test_utils_examples)

suite = unittest.TestSuite([
    test_graphs.suite,
    test_filters.suite,
    test_utils.suite,
    test_learning.suite,
    test_docstrings.suite,
    test_plotting.suite,  # TODO: can SIGSEGV if not last
    test_utils_examples.suite,  # TODO: test function
])
