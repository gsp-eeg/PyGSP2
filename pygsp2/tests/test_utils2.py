# -*- coding: utf-8 -*-
"""Test suite for the utils module of the pygsp2 package."""

import unittest

import pandas as pd
from unidecode import unidecode

from pygsp2 import utils_examples


class TestCase(unittest.TestCase):
    """
    TestCase for the utils module of the pygsp2 package.

    This test case checks the functionality of the utils_examples module,
    particularly the functions related to metro graph generation and
    signal processing.

    Methods
    -------
    setUpClass(cls):
        Set up any state specific to the test case.

    tearDownClass(cls):
        Clean up any resources set up during the test case.

    test_graphs():
        Tests the graph-related functions from utils_examples.
    """

    @classmethod
    def setUpClass(cls):
        """Set up any state specific to the test case."""
        pass

    @classmethod
    def tearDownClass(cls):
        """Clean up any resources set up during the test case."""
        pass

    def test_graphs(self):
        """
        Test the graph-related functions from utils_examples.

        This method tests the following:
        - Generation of a metro graph using make_metro_graph.
        - Handling of invalid parameters in make_metro_graph.
        - Preprocessing of metro commute data with metro_database_preprocessing.
        - Plotting of signal data on the metro graph using plot_signal_in_graph.

        Raises
        ------
        ValueError
            If an invalid aggregation function is passed to make_metro_graph.
        """
        edgesfile = ('../data/santiago_metro_stations_connections.txt', )
        coordsfile = '../data/santiago_metro_stations_coords.geojson'
        G, pos = utils_examples.make_metro_graph(edgesfile, coordsfile)

        self.assertRaises(ValueError, utils_examples.make_metro_graph,
                          (edgesfile, coordsfile), 'sum')

        commutes = pd.read_excel(
            'data/2023.11 Matriz_baj_SS_MH.xlsb',
            header=1,
            sheet_name='bajadas_prom_laboral',
        )
        stations = [name.upper() for name in list(G.nodes)]
        stations = [unidecode(station) for station in stations]
        mc, s = utils_examples.metro_database_preprocessing(commutes, stations)
        fig, ax = utils_examples.plot_signal_in_graph(G, s, 'utils_examples test')


# Load the test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
