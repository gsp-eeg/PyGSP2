"""Example of a graph signal for the Santiago Metro.

This example shows a graph signal defined over a graph induced by the Santiago
Metro (https://en.wikipedia.org/wiki/Santiago_Metro). Each station is a
node. Two nodes are connected if the corresponding stations are connected. The
data for each node is the daily average of people leaving each station.

To run this example, you need to download three files and place them in the
same directory as this script.

1. Download the file `Tablas de subidas y bajadas nov23.zip` from this link:

https://www.dtpm.cl/descargas/modelos_y_matrices/Tablas%20de%20subidas%20y%20bajadas%20nov23.zip

Then, uncompress the zip file and copy `2023.11 Matriz_baj_SS_MH.xlsb` to the
same location as this script.

2. Download the file `santiago_metro_stations_coords.geojson` from this link:

https://zenodo.org/records/11637462/files/santiago_metro_stations_coords.geojson

3. Download the file `santiago_metro_stations_connections.txt` from this link:

https://zenodo.org/records/11637462/files/santiago_metro_stations_connections.txt
"""
# %%
import os
import sys
import pandas as pd
from unidecode import unidecode
import matplotlib.pyplot as plt
from utils import make_metro_graph, metro_database_preprocessing, plot_signal_in_graph
os.chdir(os.path.dirname(__file__))

try:
    commutes = pd.read_excel('2023.11 Matriz_baj_SS_MH.xlsb', header=1,
                             sheet_name='bajadas_prom_laboral')
except FileNotFoundError:
    print(f'Data file was not found in:\n {os.getcwd()}')
    print('Download it from:\n' +
          r'https://www.dtpm.cl/descargas/modelos_y_matrices/Tablas%20de%20subidas%20y%20bajadas%20nov23.zip')
    sys.exit(-1)

# %% Load graph

G = make_metro_graph()
pos = {node: (G.nodes[node]['y'], G.nodes[node]['x']) for node in G.nodes}

stations = [name.upper() for name in list(G.nodes)]
stations = [unidecode(station) for station in stations]

metro_commutes, signal = metro_database_preprocessing(commutes, stations)

# %% Let me know if there are stations not in the database

available_stations = list(metro_commutes['paradero'])

stations_missing = []
for i, station in enumerate(stations):
    if not station in available_stations:
        stations_missing.append(station)

print(f'{len(stations_missing)} Stations Not Meassured:')
_ = [print(f'\t{i+1}. {station}')
     for i, station in enumerate(stations_missing)]

# %% Plot signal in graph

fig, ax = plot_signal_in_graph(G, signal, 'Promedio Diario\nBajadas de Metro')
plt.show()
