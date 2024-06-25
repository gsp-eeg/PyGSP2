# -*- coding: utf-8 -*-

"""
Test suite for the utils module of the pygsp2 package.

"""

import unittest
import pandas as pd
from unidecode import unidecode

from pygsp2 import utils2


class TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_graphs(self):
        print("hola")
        edgesfile='../data/santiago_metro_stations_connections.txt',
        coordsfile='../data/santiago_metro_stations_coords.geojson'
        G, pos = utils2.make_metro_graph(edgesfile, coordsfile)
        self.assertRaises(ValueError, utils2.make_metro_graph, (edgesfile, coordsfile), 'sum')

        commutes = pd.read_excel('data/2023.11 Matriz_baj_SS_MH.xlsb', header=1,
                             sheet_name='bajadas_prom_laboral')
        stations = [name.upper() for name in list(G.nodes)]
        stations = [unidecode(station) for station in stations]
        mc, s = utils2.metro_database_preprocessing(commutes, stations)
        fig, ax = utils2.plot_signal_in_graph(G, s, 'Utils2 test')


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
