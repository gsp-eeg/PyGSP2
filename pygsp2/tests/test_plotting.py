# -*- coding: utf-8 -*-
"""Test suite for the plotting module of the pygsp2 package."""

import os
import unittest

import numpy as np
from matplotlib import pyplot as plt
from skimage import data, img_as_float

from pygsp2 import filters, graphs, plotting


class TestGraphs(unittest.TestCase):
    """Tests for the graph plotting functionalities in the pygsp2 package."""

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment.

        Load and preprocess a sample image to be used in tests.
        """
        cls._img = img_as_float(data.camera()[::16, ::16])

    def tearDown(self):
        """
        Clean up after each test.

        Close all open plotting windows.
        """
        plotting.close_all()

    def test_all_graphs(self):
        """
        Test plotting for all graph types that have coordinates.

        This includes plotting with and without signals and using both backends: 'pyqtgraph' and 'matplotlib'.
        """
        # Define graph classes without coordinates.
        COORDS_NO = {
            'Graph',
            'BarabasiAlbert',
            'ErdosRenyi',
            'FullConnected',
            'RandomRegular',
            'StochasticBlockModel',
        }

        Gs = []
        for classname in dir(graphs):
            if not classname[0].isupper():
                # Skip non-graph classes.
                continue
            elif classname in COORDS_NO:
                continue
            elif classname == 'ImgPatches':
                # Skip classes with coordinates not in 2D or 3D.
                continue

            Graph = getattr(graphs, classname)

            # Instantiate graph classes with required parameters.
            if classname == 'NNGraph':
                Xin = np.arange(90).reshape(30, 3)
                Gs.append(Graph(Xin))
            elif classname == 'Grid2dImgPatches':
                Gs.append(Graph(img=self._img, patch_shape=(3, 3)))
            elif classname == 'LineGraph':
                Gs.append(Graph(graphs.Sensor(20, seed=42)))
            else:
                Gs.append(Graph())

            # Add additional test cases.
            if classname == 'TwoMoons':
                Gs.append(Graph(moontype='standard'))
                Gs.append(Graph(moontype='synthesized'))
            elif classname == 'Cube':
                Gs.append(Graph(nb_dim=2))
                Gs.append(Graph(nb_dim=3))
            elif classname == 'DavidSensorNet':
                Gs.append(Graph(N=64))
                Gs.append(Graph(N=500))
                Gs.append(Graph(N=128))

        for G in Gs:
            self.assertTrue(hasattr(G, 'coords'))
            self.assertEqual(G.N, G.coords.shape[0])

            signal = np.arange(G.N) + 0.3

            G.plot(backend='pyqtgraph')
            G.plot(backend='matplotlib')
            G.plot(signal, backend='pyqtgraph')
            G.plot(signal, backend='matplotlib')
            plotting.close_all()

    def test_highlight(self):
        """
        Test highlighting functionality for graph plots.

        Verify correct highlighting for 1D, 2D, and 3D graphs.
        """

        def test(G):
            s = np.arange(G.N)
            G.plot(s, backend='matplotlib', highlight=0)
            G.plot(s, backend='matplotlib', highlight=[0])
            G.plot(s, backend='matplotlib', highlight=[0, 1])

        # Test for various graph types.
        G = graphs.Ring(10)
        test(G)
        G.set_coordinates('line1D')
        test(G)
        G = graphs.Torus(Nv=5)
        test(G)

    def test_indices(self):
        """
        Test plotting with and without indices on graph plots.

        Verify correct plotting for 2D and 3D graphs with index labels.
        """

        def test(G):
            G.plot(backend='matplotlib', indices=False)
            G.plot(backend='matplotlib', indices=True)

        # Test for various graph types.
        G = graphs.Ring(10)
        test(G)
        G = graphs.Torus(Nv=5)
        test(G)

    def test_signals(self):
        """
        Test various types of signals that can be plotted on graphs.

        Includes testing different color and size parameters for vertices and edges.
        """
        G = graphs.Sensor()
        G.plot()
        rng = np.random.default_rng(42)

        def test_color(param, length):
            for value in [
                    'r',
                    4 * (0.5, ),
                    length * (2, ),
                    np.ones([1, length]),
                    rng.random(length),
                    np.ones([length, 3]),
                ['red'] * length,
                    rng.random([length, 4]),
            ]:
                params = {param: value}
                G.plot(**params)
            for value in [
                    10,
                (0.5, 0.5),
                    np.ones([length, 2]),
                    np.ones([2, length, 3]),
                    np.ones([length, 3]) * 1.1,
            ]:
                params = {param: value}
                self.assertRaises(ValueError, G.plot, **params)
            for value in ['r', 4 * (0.5)]:
                params = {param: value, 'backend': 'pyqtgraph'}
                self.assertRaises(ValueError, G.plot, **params)

        test_color('vertex_color', G.n_vertices)
        test_color('edge_color', G.n_edges)

        def test_size(param, length):
            for value in [15, length * (2, ), np.ones([1, length]), rng.random(length)]:
                params = {param: value}
                G.plot(**params)
            for value in [(2, 3, 4, 5), np.ones([2, length]), np.ones([2, length, 3])]:
                params = {param: value}
                self.assertRaises(ValueError, G.plot, **params)

        test_size('vertex_size', G.n_vertices)
        test_size('edge_width', G.n_edges)

    def test_show_close(self):
        """
        Test the show and close functionality of plots.

        Verify that plots can be shown without blocking the test execution and closed correctly.
        """
        G = graphs.Sensor()
        G.plot()
        plotting.show(block=False)  # Non-blocking to avoid test halt.
        plotting.close()
        plotting.close_all()

    def test_coords(self):
        """
        Test handling of graph coordinates.

        Verify that plotting raises an AttributeError for invalid coordinate configurations.
        """
        G = graphs.Sensor()
        del G.coords
        self.assertRaises(AttributeError, G.plot)
        G.coords = None
        self.assertRaises(AttributeError, G.plot)
        G.coords = np.ones((G.N, 4))
        self.assertRaises(AttributeError, G.plot)
        G.coords = np.ones((G.N, 3, 1))
        self.assertRaises(AttributeError, G.plot)
        G.coords = np.ones((G.N // 2, 3))
        self.assertRaises(AttributeError, G.plot)

    def test_unknown_backend(self):
        """
        Test handling of unknown backend names.

        Verify that plotting raises a ValueError for unrecognized backend names.
        """
        G = graphs.Sensor()
        self.assertRaises(ValueError, G.plot, backend='abc')


class TestFilters(unittest.TestCase):
    """Tests for the filter plotting functionalities in the pygsp2 package."""

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment.

        Create a sample graph and compute its Fourier basis.
        """
        cls._graph = graphs.Sensor(20, seed=42)
        cls._graph.compute_fourier_basis()

    def tearDown(self):
        """
        Clean up after each test.

        Close all open plotting windows.
        """
        plotting.close_all()

    def test_all_filters(self):
        """
        Test plotting for all filter types.

        Verify that all filters can be plotted.
        """
        for classname in dir(filters):
            if not classname[0].isupper():
                # Skip non-filter classes.
                continue
            Filter = getattr(filters, classname)
            if classname in ['Filter', 'Modulation', 'Gabor']:
                g = Filter(self._graph, filters.Heat(self._graph))
            else:
                g = Filter(self._graph)
            g.plot()
            plotting.close_all()

    def test_evaluation_points(self):
        """
        Test the effect of changing the number of evaluation points on filter plots.

        Verify that the number of lines and data points match expectations.
        """

        def check(ax, n_lines, n_points):
            self.assertEqual(len(ax.lines), n_lines)  # n_filters + sum
            x, y = ax.lines[0].get_data()
            self.assertEqual(len(x), n_points)
            self.assertEqual(len(y), n_points)

        g = filters.Abspline(self._graph, 5)
        fig, ax = g.plot(eigenvalues=False)
        check(ax, 6, 500)
        fig, ax = g.plot(40, eigenvalues=False)
        check(ax, 6, 40)
        fig, ax = g.plot(n=20, eigenvalues=False)
        check(ax, 6, 20)

    def test_eigenvalues(self):
        """
        Test plotting with and without showing eigenvalues.

        Verify correct plotting with and without eigenvalues for the Heat filter.
        """
        graph = graphs.Sensor(20, seed=42)
        graph.estimate_lmax()
        filters.Heat(graph).plot()
        filters.Heat(graph).plot(eigenvalues=False)
        graph.compute_fourier_basis()
        filters.Heat(graph).plot()
        filters.Heat(graph).plot(eigenvalues=True)
        filters.Heat(graph).plot(eigenvalues=False)

    def test_sum_and_labels(self):
        """
        Test plotting with and without sum or labels.

        Verify that the plot behaves correctly with different combinations of sum and label parameters.
        """

        def test(g):
            for sum in [None, True, False]:
                for labels in [None, True, False]:
                    g.plot(sum=sum, labels=labels)

        test(filters.Heat(self._graph, 10))  # one filter
        test(filters.Heat(self._graph, [10, 100]))  # multiple filters

    def test_title(self):
        """
        Test the plot title functionality.

        Verify that the title of the plot matches expected values.
        """
        fig, ax = filters.Wave(self._graph, 2, 1).plot()
        assert ax.get_title() == 'Wave(in=1, out=1, time=[2.00], speed=[1.00])'
        fig, ax = filters.Wave(self._graph).plot(title='test')
        assert ax.get_title() == 'test'

    def test_ax(self):
        """
        Test the return of axes from the plot function.

        Verify that axes are returned correctly and can be manually set.
        """
        fig, ax = plt.subplots()
        fig2, ax2 = filters.Heat(self._graph).plot(ax=ax)
        self.assertIs(fig2, fig)
        self.assertIs(ax2, ax)

    def test_kwargs(self):
        """
        Test passing additional parameters to the matplotlib plotting functions.

        Verify that extra parameters like alpha, linewidth, linestyle, and label are correctly handled.
        """
        g = filters.Heat(self._graph)
        g.plot(alpha=1)
        g.plot(linewidth=2)
        g.plot(linestyle='-')
        g.plot(label='myfilter')


suite = unittest.TestSuite([
    unittest.TestLoader().loadTestsFromTestCase(TestGraphs),
    unittest.TestLoader().loadTestsFromTestCase(TestFilters),
])
