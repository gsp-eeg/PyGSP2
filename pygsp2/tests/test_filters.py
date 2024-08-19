# -*- coding: utf-8 -*-
"""Test suite for the filters module of the pygsp2 package."""

import sys
import unittest

import numpy as np

from pygsp2 import filters, graphs


class TestCase(unittest.TestCase):
    """Tests for the filter functionalities in the pygsp2 package."""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment.

        Initializes the graph and signal used in the tests.
        """
        cls._G = graphs.Sensor(123, seed=42)
        cls._G.compute_fourier_basis()
        cls._rng = np.random.default_rng(42)
        cls._signal = cls._rng.uniform(size=cls._G.N)

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests.

        Currently does nothing.
        """
        pass

    def _generate_coefficients(self, N, Nf, vertex_delta=83):
        """Generate filter coefficients for testing.

        Parameters
        ----------
        N : int
            The number of vertices in the graph.
        Nf : int
            The number of filters.
        vertex_delta : int, optional
            The vertex index to start the coefficients, by default 83

        Returns
        -------
        np.ndarray
            The generated coefficients.
        """
        S = np.zeros((N * Nf, Nf))
        S[vertex_delta] = 1
        for i in range(Nf):
            S[vertex_delta + i * self._G.N, i] = 1
        return S

    def _test_methods(self, f, tight, check=True):
        """Test various methods of a filter.

        Parameters
        ----------
        f : filters.Filter
            The filter object to test.
        tight : bool
            Whether the filter is expected to be tight.
        check : bool, optional
            Whether to check the filter methods, by default True
        """
        self.assertIs(f.G, self._G)

        f.evaluate(self._G.e)
        f.evaluate(np.random.default_rng().uniform(0, 1, size=(4, 6, 3)))

        A, B = f.estimate_frame_bounds(self._G.e)
        if tight:
            np.testing.assert_allclose(A, B)
        else:
            assert B - A > 0.01

        # Analysis.
        s2 = f.filter(self._signal, method='exact')
        s3 = f.filter(self._signal, method='chebyshev', order=100)

        # Synthesis.
        s4 = f.filter(s2, method='exact')
        s5 = f.filter(s3, method='chebyshev', order=100)

        if check:
            # Chebyshev should be close to exact.
            # Does not pass for Gabor, Modulation and Rectangular (not smooth).
            np.testing.assert_allclose(s2, s3, rtol=0.1, atol=0.01)
            np.testing.assert_allclose(s4, s5, rtol=0.1, atol=0.01)

        if tight:
            # Tight frames should not loose information.
            np.testing.assert_allclose(s4, A * self._signal)
            assert np.linalg.norm(s5 - A * self._signal) < 0.1

        # Computing the frame is an alternative way to filter.
        if f.n_filters != self._G.n_vertices:
            # It doesn't work (yet) if the number of filters is the same as the
            # number of nodes as f.filter() infers that we want synthesis where
            # we actually want analysis.
            F = f.compute_frame(method='exact')
            s = F.dot(self._signal).reshape(-1, self._G.N).T.squeeze()
            np.testing.assert_allclose(s, s2)

            F = f.compute_frame(method='chebyshev', order=100)
            s = F.dot(self._signal).reshape(-1, self._G.N).T.squeeze()
            np.testing.assert_allclose(s, s3)

    def test_filter(self, n_filters=5):
        """Test the filter method of MexicanHat filter.

        Parameters
        ----------
        n_filters : int, optional
            Number of filters to test, by default 5
        """
        g = filters.MexicanHat(self._G, n_filters)

        s1 = self._rng.uniform(size=self._G.N)
        s2 = s1.reshape((self._G.N, 1))
        s3 = g.filter(s1)
        s4 = g.filter(s2)
        s5 = g.analyze(s1)
        assert s3.shape == (self._G.N, n_filters)
        np.testing.assert_allclose(s3, s4)
        np.testing.assert_allclose(s3, s5)

        s1 = self._rng.uniform(size=(self._G.N, 4))
        s2 = s1.reshape((self._G.N, 4, 1))
        s3 = g.filter(s1)
        s4 = g.filter(s2)
        s5 = g.analyze(s1)
        assert s3.shape == (self._G.N, 4, n_filters)
        np.testing.assert_allclose(s3, s4)
        np.testing.assert_allclose(s3, s5)

        s1 = self._rng.uniform(size=(self._G.N, n_filters))
        s2 = s1.reshape((self._G.N, 1, n_filters))
        s3 = g.filter(s1)
        s4 = g.filter(s2)
        s5 = g.synthesize(s1)
        assert s3.shape == (self._G.N, )
        np.testing.assert_allclose(s3, s4)
        np.testing.assert_allclose(s3, s5)

        s1 = self._rng.uniform(size=(self._G.N, 10, n_filters))
        s3 = g.filter(s1)
        s5 = g.synthesize(s1)
        assert s3.shape == (self._G.N, 10)
        np.testing.assert_allclose(s3, s5)

    def test_localize(self):
        """Test localization of signals at nodes.

        Verifies that localization of signal at a node is correct.
        """
        g = filters.Heat(self._G, 100)

        # Localize signal at node by filtering Kronecker delta.
        NODE = 10
        s1 = g.localize(NODE, method='exact')

        # Should be equal to a row / column of the filtering operator.
        gL = self._G.U.dot(np.diag(g.evaluate(self._G.e)[0]).dot(self._G.U.T))
        s2 = np.sqrt(self._G.N) * gL[NODE, :]
        np.testing.assert_allclose(s1, s2)

        # That is actually a row / column of the analysis operator.
        F = g.compute_frame(method='exact')
        np.testing.assert_allclose(F, gL)

    def test_frame_bounds(self):
        """Test estimation of frame bounds.

        Checks estimation for different filter types.
        """
        # Not a frame, it as a null-space.
        g = filters.Rectangular(self._G)
        A, B = g.estimate_frame_bounds()
        self.assertEqual(A, 0)
        self.assertEqual(B, 1)
        # Identity is tight.
        g = filters.Filter(self._G, lambda x: np.full_like(x, 2))
        A, B = g.estimate_frame_bounds()
        self.assertEqual(A, 4)
        self.assertEqual(B, 4)

    def test_frame(self):
        """Test the frame of a filter.

        Verifies that the frame is a stack of functions of the Laplacian.
        """
        g = filters.Heat(self._G, scale=[8, 9])
        gL1 = g.compute_frame(method='exact')
        gL2 = g.compute_frame(method='chebyshev', order=30)

        def get_frame(freq_response):
            return self._G.U.dot(np.diag(freq_response).dot(self._G.U.T))

        gL = np.concatenate([get_frame(gl) for gl in g.evaluate(self._G.e)])
        np.testing.assert_allclose(gL1, gL)
        np.testing.assert_allclose(gL2, gL, atol=1e-10)

    def test_complement(self, frame_bound=2.5):
        """Test if filter banks become tight with the addition of their complement.

        Parameters
        ----------
        frame_bound : float, optional
            Frame bound for the filter, by default 2.5
        """
        g = filters.MexicanHat(self._G)
        g += g.complement(frame_bound)
        A, B = g.estimate_frame_bounds()
        np.testing.assert_allclose(A, frame_bound)
        np.testing.assert_allclose(B, frame_bound)

    def test_inverse(self, frame_bound=3):
        """Test the pseudo-inverse of a filter's frame.

        Parameters
        ----------
        frame_bound : float, optional
            Frame bound for the filter, by default 3
        """
        g = filters.Heat(self._G, scale=[2, 3, 4])
        h = g.inverse()
        Ag, Bg = g.estimate_frame_bounds()
        Ah, Bh = h.estimate_frame_bounds()
        np.testing.assert_allclose(Ag * Bh, 1)
        np.testing.assert_allclose(Bg * Ah, 1)
        gL = g.compute_frame(method='exact')
        hL = h.compute_frame(method='exact')
        I = np.identity(self._G.N)
        np.testing.assert_allclose(hL.T.dot(gL), I, atol=1e-10)
        pinv = np.linalg.inv(gL.T.dot(gL)).dot(gL.T)
        np.testing.assert_allclose(pinv, hL.T, atol=1e-10)
        # The reconstruction is exact for any frame (lower bound A > 0).
        y = g.filter(self._signal, method='exact')
        z = h.filter(y, method='exact')
        np.testing.assert_allclose(z, self._signal)
        # Not invertible if not a frame.
        g = filters.Expwin(self._G)
        with self.assertLogs(level='WARNING'):
            h = g.inverse()
            h.evaluate(self._G.e)
        # If the frame is tight, inverse is h=g/A.
        g += g.complement(frame_bound)
        h = g.inverse()
        he = g(self._G.e) / frame_bound
        np.testing.assert_allclose(h(self._G.e), he, atol=1e-10)

    def test_custom_filter(self):
        """
        Test the custom filter with a specific kernel function.

        This test verifies the initialization of the `Filter` class with a
        custom kernel function. It ensures that the number of filters is set
        correctly and that the kernel function is assigned as expected.
        The `_test_methods` method is used to further validate the filter's
        functionality.

        Raises
        ------
        AssertionError
            If the number of filters or kernel assignment is incorrect.
        """

        def kernel(x):
            return x / (1.0 + x)

        f = filters.Filter(self._G, kernels=kernel)
        self.assertEqual(f.Nf, 1)
        self.assertIs(f._kernels[0], kernel)
        self._test_methods(f, tight=False)

    def test_abspline(self):
        """
        Test the Abspline filter.

        This test checks the functionality of the `Abspline` filter with a
        specific number of filters. The `_test_methods` method is used to
        validate the filter's methods.

        Raises
        ------
        AssertionError
            If the filter's methods do not pass the validation checks.
        """
        f = filters.Abspline(self._G, Nf=4)
        self._test_methods(f, tight=False)

    def test_gabor(self):
        """
        Test the Gabor filter.

        This test verifies the initialization of the `Gabor` filter using
        the `Rectangular` filter and checks for proper error handling when
        using invalid inputs. It compares the output of the `Gabor` filter
        with the `Modulation` filter to ensure equivalency.

        Raises
        ------
        AssertionError
            If the Gabor filter does not work as expected with valid inputs or
            if it fails to raise errors with invalid inputs.
        """
        f = filters.Rectangular(self._G, None, 0.1)
        f = filters.Gabor(self._G, f)
        self._test_methods(f, tight=False, check=False)
        self.assertRaises(ValueError, filters.Gabor, graphs.Sensor(), f)
        f = filters.Regular(self._G)
        self.assertRaises(ValueError, filters.Gabor, self._G, f)

    def test_modulation(self):
        """
        Test the Modulation filter.

        This test checks the `Modulation` filter with various configurations,
        including different `modulation_first` parameters. It also verifies
        that errors are raised when invalid inputs are used.

        Raises
        ------
        AssertionError
            If the Modulation filter does not work as expected with valid inputs
            or if it fails to raise errors with invalid inputs.
        """
        f = filters.Rectangular(self._G, None, 0.1)
        # TODO: synthesis doesn't work yet.
        # f = filters.Modulation(self._G, f, modulation_first=False)
        # self._test_methods(f, tight=False, check=False)
        f = filters.Modulation(self._G, f, modulation_first=True)
        self._test_methods(f, tight=False, check=False)
        self.assertRaises(ValueError, filters.Modulation, graphs.Sensor(), f)
        f = filters.Regular(self._G)
        self.assertRaises(ValueError, filters.Modulation, self._G, f)

    def test_modulation_gabor(self):
        """
        Test the equivalency of Modulation and Gabor filters.

        This test verifies that the `Modulation` and `Gabor` filters produce
        equivalent results when applied to a signal, using deltas centered at
        the eigenvalues.

        Raises
        ------
        AssertionError
            If the outputs of Modulation and Gabor filters are not close to each other.
        """
        f = filters.Rectangular(self._G, 0, 0)
        f1 = filters.Modulation(self._G, f, modulation_first=True)
        f2 = filters.Gabor(self._G, f)
        s1 = f1.filter(self._signal)
        s2 = f2.filter(self._signal)
        np.testing.assert_allclose(abs(s1), abs(s2), atol=1e-5)

    def test_halfcosine(self):
        """
        Test the HalfCosine filter.

        This test checks the functionality of the `HalfCosine` filter with a
        specified number of filters. The `_test_methods` method is used to
        validate the filter's methods.

        Raises
        ------
        AssertionError
            If the filter's methods do not pass the validation checks.
        """
        f = filters.HalfCosine(self._G, Nf=4)
        self._test_methods(f, tight=True)

    def test_itersine(self):
        """
        Test the Itersine filter.

        This test checks the functionality of the `Itersine` filter with a
        specified number of filters. The `_test_methods` method is used to
        validate the filter's methods.

        Raises
        ------
        AssertionError
            If the filter's methods do not pass the validation checks.
        """
        f = filters.Itersine(self._G, Nf=4)
        self._test_methods(f, tight=True)

    def test_mexicanhat(self):
        """
        Test the MexicanHat filter.

        This test verifies the functionality of the `MexicanHat` filter with
        different configurations for normalization and number of filters.
        The `_test_methods` method is used to validate the filter's methods.

        Raises
        ------
        AssertionError
            If the filter's methods do not pass the validation checks.
        """
        f = filters.MexicanHat(self._G, Nf=5, normalize=False)
        self._test_methods(f, tight=False)
        f = filters.MexicanHat(self._G, Nf=4, normalize=True)
        self._test_methods(f, tight=False)

    def test_meyer(self):
        """
        Test the Meyer filter.

        This test checks the functionality of the `Meyer` filter with a
        specified number of filters. The `_test_methods` method is used to
        validate the filter's methods.

        Raises
        ------
        AssertionError
            If the filter's methods do not pass the validation checks.
        """
        f = filters.Meyer(self._G, Nf=4)
        self._test_methods(f, tight=True)

    def test_simpletf(self):
        """
        Test the SimpleTight filter.

        This test verifies the functionality of the `SimpleTight` filter with
        a specified number of filters. The `_test_methods` method is used to
        validate the filter's methods.

        Raises
        ------
        AssertionError
            If the filter's methods do not pass the validation checks.
        """
        f = filters.SimpleTight(self._G, Nf=4)
        self._test_methods(f, tight=True)

    def test_regular(self):
        """
        Test the Regular filter.

        This test checks the functionality of the `Regular` filter with
        different configurations for degree. The `_test_methods` method
        is used to validate the filter's methods.

        Raises
        ------
        AssertionError
            If the filter's methods do not pass the validation checks.
        """
        f = filters.Regular(self._G)
        self._test_methods(f, tight=True)
        f = filters.Regular(self._G, degree=5)
        self._test_methods(f, tight=True)
        f = filters.Regular(self._G, degree=0)
        self._test_methods(f, tight=True)

    def test_held(self):
        """
        Test the Held filter.

        This test verifies the functionality of the `Held` filter with and
        without specific parameter `a`. The `_test_methods` method is used
        to validate the filter's methods.

        Raises
        ------
        AssertionError
            If the filter's methods do not pass the validation checks.
        """
        f = filters.Held(self._G)
        self._test_methods(f, tight=True)
        f = filters.Held(self._G, a=0.25)
        self._test_methods(f, tight=True)

    def test_simoncelli(self):
        """
        Test the Simoncelli filter.

        This test checks the functionality of the `Simoncelli` filter with
        and without specific parameter `a`. The `_test_methods` method is
        used to validate the filter's methods.

        Raises
        ------
        AssertionError
            If the filter's methods do not pass the validation checks.
        """
        f = filters.Simoncelli(self._G)
        self._test_methods(f, tight=True)
        f = filters.Simoncelli(self._G, a=0.25)
        self._test_methods(f, tight=True)

    def test_papadakis(self):
        """
        Test the Papadakis filter.

        This test verifies the functionality of the `Papadakis` filter with
        and without specific parameter `a`. The `_test_methods` method is used
        to validate the filter's methods.

        Raises
        ------
        AssertionError
            If the filter's methods do not pass the validation checks.
        """
        f = filters.Papadakis(self._G)
        self._test_methods(f, tight=True)
        f = filters.Papadakis(self._G, a=0.25)
        self._test_methods(f, tight=True)

    def test_heat(self):
        """
        Test the Heat filter.

        This test checks the functionality of the `Heat` filter with various
        configurations for normalization and scale. It verifies that the
        normalization is correctly applied and that the filter's methods
        work as expected.

        Raises
        ------
        AssertionError
            If the filter's output does not match the expected results for
            different configurations.
        """
        f = filters.Heat(self._G, normalize=False, scale=10)
        self._test_methods(f, tight=False)
        f = filters.Heat(self._G, normalize=False, scale=np.array([5, 10]))
        self._test_methods(f, tight=False)
        f = filters.Heat(self._G, normalize=True, scale=10)
        np.testing.assert_allclose(np.linalg.norm(f.evaluate(self._G.e)), 1)
        self._test_methods(f, tight=False)
        f = filters.Heat(self._G, normalize=True, scale=[5, 10])
        np.testing.assert_allclose(np.linalg.norm(f.evaluate(self._G.e)[0]), 1)
        np.testing.assert_allclose(np.linalg.norm(f.evaluate(self._G.e)[1]), 1)
        self._test_methods(f, tight=False)

    def test_wave(self):
        """
        Test the Wave filter.

        This test verifies the functionality of the `Wave` filter with various
        configurations for time and speed parameters. It also checks for error
        handling with invalid inputs.

        Raises
        ------
        AssertionError
            If the Wave filter's output does not match expected results or
            if it does not raise errors with invalid inputs.
        """
        f = filters.Wave(self._G)
        self._test_methods(f, tight=False)
        f = filters.Wave(self._G, time=1)
        self._test_methods(f, tight=False)
        f = filters.Wave(self._G, time=[1, 2, 3])
        self._test_methods(f, tight=False)
        f = filters.Wave(self._G, speed=[1])
        self._test_methods(f, tight=False)
        f = filters.Wave(self._G, speed=[0.5, 1, 1.5])
        self._test_methods(f, tight=False)
        f = filters.Wave(self._G, time=[1, 2], speed=[1, 1.5])
        self._test_methods(f, tight=False)
        # Sequences of differing lengths.
        self.assertRaises(ValueError, filters.Wave, self._G, time=[1, 2, 3],
                          speed=[0, 1])
        # Invalid speed.
        self.assertRaises(ValueError, filters.Wave, self._G, speed=2)

    def test_expwin(self):
        """
        Test the Expwin filter.

        This test checks the functionality of the `Expwin` filter with different
        configurations for `band_min` and `band_max`. The `_test_methods` method
        is used to validate the filter's methods.

        Raises
        ------
        AssertionError
            If the Expwin filter's methods do not pass the validation checks.
        """
        f = filters.Expwin(self._G)
        self._test_methods(f, tight=False)
        f = filters.Expwin(self._G, band_min=None, band_max=0.8)
        self._test_methods(f, tight=False)
        f = filters.Expwin(self._G, band_min=0.1, band_max=None)
        self._test_methods(f, tight=False)
        f = filters.Expwin(self._G, band_min=0.1, band_max=0.7)
        self._test_methods(f, tight=False)
        f = filters.Expwin(self._G, band_min=None, band_max=None)
        self._test_methods(f, tight=True)

    def test_rectangular(self):
        """
        Test the Rectangular filter.

        This test verifies the functionality of the `Rectangular` filter with
        different configurations for `band_min` and `band_max`. It also checks
        the validation of the filter's methods.

        Raises
        ------
        AssertionError
            If the Rectangular filter's methods do not pass the validation checks.
        """
        f = filters.Rectangular(self._G)
        self._test_methods(f, tight=False, check=False)
        f = filters.Rectangular(self._G, band_min=None, band_max=0.8)
        self._test_methods(f, tight=False, check=False)
        f = filters.Rectangular(self._G, band_min=0.1, band_max=None)
        self._test_methods(f, tight=False, check=False)
        f = filters.Rectangular(self._G, band_min=0.1, band_max=0.7)
        self._test_methods(f, tight=False, check=False)
        f = filters.Rectangular(self._G, band_min=None, band_max=None)
        self._test_methods(f, tight=True, check=True)

    def test_approximations(self):
        """
        Test the consistency of filter analysis methods.

        This test verifies that different methods for filter analysis
        (`exact`, `cheby`, and `lanczos`) produce the same output. The
        method `exact` and `chebyshev` results are compared, while `lanczos`
        is expected to raise an error.

        Raises
        ------
        AssertionError
            If the `exact` and `chebyshev` methods do not produce the same
            results, or if `lanczos` does not raise a ValueError.
        """
        # TODO: done in _test_methods.

        f = filters.Heat(self._G)
        c_exact = f.filter(self._signal, method='exact')
        c_cheby = f.filter(self._signal, method='chebyshev')

        np.testing.assert_allclose(c_exact, c_cheby)
        self.assertRaises(ValueError, f.filter, self._signal, method='lanczos')


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
