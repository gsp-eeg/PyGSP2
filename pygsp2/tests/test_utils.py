# -*- coding: utf-8 -*-
"""Test suite for the utils module of the pygsp2 package."""

import unittest

import numpy as np
from scipy import sparse

from pygsp2 import graphs, utils


class TestCase(unittest.TestCase):
    """
    TestCase for the utils module of the pygsp2 package.

    This test case checks the functionality of the symmetrize function
    within the utils module. The symmetrize function is used to make
    a matrix symmetric using various methods.

    Methods
    -------
    setUpClass(cls):
        Set up any state specific to the test case.

    tearDownClass(cls):
        Clean up any resources set up during the test case.

    test_symmetrize():
        Tests the symmetrize function with different methods and
        compares the output of sparse and dense matrices.
    """

    @classmethod
    def setUpClass(cls):
        """Set up any state specific to the test case."""
        pass

    @classmethod
    def tearDownClass(cls):
        """Clean up any resources set up during the test case."""
        pass

    def test_symmetrize(self):
        """
        Test the symmetrize function with various methods.

        This method tests the symmetrize function using the following methods:
        - 'average'
        - 'maximum'
        - 'fill'
        - 'tril'
        - 'triu'

        The test ensures that the symmetrize function produces the same result
        when applied to both sparse and dense matrices.

        Raises
        ------
        ValueError
            If an invalid method is passed to the symmetrize function.
        """
        W = sparse.random(100, 100, random_state=42)
        for method in ['average', 'maximum', 'fill', 'tril', 'triu']:
            # Test that the regular and sparse versions give the same result.
            W1 = utils.symmetrize(W, method=method)
            W2 = utils.symmetrize(W.toarray(), method=method)
            np.testing.assert_equal(W1.toarray(), W2)
        self.assertRaises(ValueError, utils.symmetrize, W, 'sum')


# Load the test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
