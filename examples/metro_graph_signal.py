"""Example of a graph signal for the Santiago Metro

This example shows a graph signal defined over a graph induced by the Santiago
Metro (https://en.wikipedia.org/wiki/Santiago_Metro). Each station is a
node. Two nodes are connected if the corresponding stations are connected. The
data for each node is the daily average of people leaving each station.

Download the file `Tablas de subidas y bajadas nov23.zip` from this link:

https://www.dtpm.cl/descargas/modelos_y_matrices/Tablas%20de%20subidas%20y%20bajadas%20nov23.zip

Then, uncompress the zip file and copy `2023.11 Matriz_baj_SS_MH.xlsb` to the
same location as this script.

An additional requirement is to have the file `metroCoords.geojson`
which was obtained from OpenStreetMap data, using the
service https://overpass-turbo.eu/, with the query

node
[public_transport=station]
[station=subway]
({{bbox}});
out;

Not that you need to have the city of Santiago in the map to use it as the
bounding box.

We can use this to get the lines, but is not clear at the moment how to
programatically get the stations that are connected:

[out:json][timeout:25];
// gather results
way["railway"="subway"]
["name"="Línea 5"]
({{bbox}});
// print results
out geom;

"""
# %%
import os
import sys
import pandas as pd
from unidecode import unidecode
import matplotlib.pyplot as plt
from utils import make_metro_graph, metro_database_preprocessing, plot_signal_in_graph
os.chdir(os.path.dirname(__file__))

# Change the name to the file you downloaded
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
