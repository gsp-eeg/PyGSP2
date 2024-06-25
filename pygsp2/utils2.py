r"""
Utils 2
=====

Utils functions used in the examples.

This module contains functions to run examples of graph signal processing.

Functions are made to avoid excesive repetition of loading, plotting and
database management. Bleow there is a list of the examples that use these
functions:

* metro_graph_signal.py
* metro_graph_regression.py
* metro_simulation.py
* metro_simulation2.py
"""
import os
import json
import utm
import networkx as nx
from geopy.distance import distance
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def make_metro_graph(edgesfile='../pygsp2/data/santiago_metro_stations_connections.txt',
                     coordsfile='../pygsp2/data/santiago_metro_stations_coords.geojson'):
    """Create a NetworkX graph corresponding to Santiago Metro network.

    Parameters
    ----------
    edgesfile : str, optional.
        File name that contains the edges of the connections.  See notes for
        the link to download the file.

    coordsfile : str, optional.
        Name of the file with the coordenates of the nodes for the graph.  See
        notes for the link to download the file.

    Returns
    -------
    G : Network Graph.
        Graph data structure containing the graph and its attributes.
    pos : dict.
        Contains the coordenates of each station. Keys are the station names.
        Names can be obtained with `list(G)`.

    Notes
    -----
    * Download the file `santiago_metro_stations_coords.geojson` from this link:

    https://zenodo.org/records/11637462/files/santiago_metro_stations_coords.geojson

    * Download the file `santiago_metro_stations_connections.txt` from this link:

    https://zenodo.org/records/11637462/files/santiago_metro_stations_connections.txt
    """

    with open(coordsfile) as f:
        data = json.load(f)

    d = {x['properties']['name']: x['geometry']['coordinates'] for
         x in data['features']}

    # Manually add missing stations
    d['Irarrázaval'] = (-70.6283197, -33.4550371)
    d['Vicente Valdés'] = (-70.5967924, -33.5264186)

    G = nx.Graph()
    ref_station = 'San Pablo'
    for station, latlong in d.items():
        x, y, _, _ = utm.from_latlon(*latlong)
        G.add_node(station, latlong=latlong,
                   distance=distance(latlong, d[ref_station]).meters,
                   x=x, y=y)

    with open(edgesfile) as f:
        for e in f.readlines():
            if e[0] == '#' or len(e) < 2:
                continue
            u, v = e.split(',')
            G.add_edge(u.strip(), v.strip())

    pos = {node: (G.nodes[node]['y'], G.nodes[node]['x']) for node in G.nodes}

    return G, pos


def metro_database_preprocessing(commutes, stations):
    """Preprocess commute database.

    Preprocess commute dataframe to match with the metro network names in
    `santiago_metro_stations_connections.txt`.

    Parameters
    ----------
    commutes : pandas DataFrame.
        The `DataFrame` is loaded from https://www.dtpm.cl/index.php/documentos/matrices-de-viaje
        the files in "tablas de Subidas y Bajadas" have to be downloaded.
    stations : list of strings.
        `stations` should contain the name of metro stations
        in the graph ordered by the node index. In a networkx
        graph this can be obtained through `list(G)`.

    Returns
    -------
    metro_commutes : pandas DataFrame.
        Rows in the given database that correspond to metro commutes.
    stations : numpy array.
        One dimensional array with the values in the `TOTAL` column. The length
        of the array is the number of stations.
    """
    # %% Use only metro commutes in database
    idx = ['L' in servicio for servicio in commutes['servicio Sonda']]
    metro_commutes = commutes[idx]

    # Change some names to coincide with node names
    metro_commutes.loc[metro_commutes['paradero'] == 'CAL Y CANTO',
                       'paradero'] = 'PUENTE CAL Y CANTO'
    metro_commutes.loc[metro_commutes['paradero'] == 'PARQUE OHIGGINS',
                       'paradero'] = 'PARQUE O\'HIGGINS'
    metro_commutes.loc[metro_commutes['paradero'] == 'PDTE PEDRO AGUIRRE CERDA',
                       'paradero'] = 'PRESIDENTE PEDRO AGUIRRE CERDA'
    metro_commutes.loc[metro_commutes['paradero'] == 'PLAZA MAIPU',
                       'paradero'] = 'PLAZA DE MAIPU'
    metro_commutes.loc[metro_commutes['paradero'] == 'RONDIZONNI',
                       'paradero'] = 'RONDIZZONI'
    metro_commutes.loc[metro_commutes['paradero'] == 'UNION LATINO AMERICANA',
                       'paradero'] = 'UNION LATINOAMERICANA'

    # %% Build signal bassed on the stations
    signal = np.zeros_like(stations, dtype=float)

    for value, station in zip(metro_commutes['TOTAL'], metro_commutes['paradero']):
        graph_idx = [station == station_graph for station_graph in stations]
        signal[graph_idx] = float(value)

    return metro_commutes, signal


def plot_signal_in_graph(G, signal, title='Graph Signal', label=''):
    """Function to plot signal in graph using networkx.

    Parameters
    ----------
    G : networkx Graph.
    signal: numpy array.
        Vector with signal values for each node. Must have the same length as
        number of nodes in `G`.
    title : str, optional.
        Title in the graph.
    label : str, optional.
        Lables to be displayed in colorbar.

    Returns
    -------
    fig : matplotlib figure.
    ax : matplotlib.pyplot axes.
    """
    pos = {node: (G.nodes[node]['y'], G.nodes[node]['x']) for node in G.nodes}
    # Map signal to a color
    cmap = matplotlib.colormaps.get_cmap('viridis')

    normalized_signal = signal / np.max(signal)
    colors = cmap(normalized_signal)

    # Initialize figure
    fig, ax = plt.subplots(figsize=(10, 7))
    # Draw edges and nodes
    nx.draw_networkx_edges(G, pos, node_size=20, ax=ax)
    pc = nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=20, ax=ax)
    cbar = plt.colorbar(pc, ticks=[0, 0.5, 1], label=label)
    cbar.set_ticklabels([f'{label:.0f}' for label in [
                        0, np.amax(signal)/2, np.amax(signal)]])
    plt.title(title)
    plt.tight_layout()

    return fig, ax
