"""
Using the data available in
https://www.dtpm.cl/index.php/documentos/matrices-de-viaje. The specific file is
able to be downloaded in the following link:

https://www.dtpm.cl/descargas/modelos_y_matrices/Tablas%20de%20subidas%20y%20bajadas%20nov23.zip

This example also uses the edges defined in edges.txt
Plot the number of average exits in each metro station in
a working day.
"""
# %%
import pandas as pd
import numpy as np
from unidecode import unidecode
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from examples.utils import make_metro_graph

# Change the name to the file you downloaded
commutes = pd.read_excel('2023.11 Matriz_baj_SS_MH.xlsb',
                         sheet_name='bajadas_prom_laboral')

# %% Load graph

G = make_metro_graph()
pos = {node: (G.nodes[node]['y'], G.nodes[node]['x']) for node in G.nodes}

stations = [name.upper() for name in list(G.nodes)]
stations = [unidecode(station) for station in stations]

# %% Use only metro commutes in database
idx = ['L' in servicio for servicio in commutes['servicio Sonda']]
metro = commutes[idx]

# Change some names to coincide with node names
metro.loc[metro['paradero'] == 'CAL Y CANTO',
          'paradero'] = 'PUENTE CAL Y CANTO'
metro.loc[metro['paradero'] == 'PARQUE OHIGGINS',
          'paradero'] = 'PARQUE O\'HIGGINS'
metro.loc[metro['paradero'] == 'PDTE PEDRO AGUIRRE CERDA',
          'paradero'] = 'PRESIDENTE PEDRO AGUIRRE CERDA'
metro.loc[metro['paradero'] == 'PLAZA MAIPU',
          'paradero'] = 'PLAZA DE MAIPU'
metro.loc[metro['paradero'] == 'RONDIZONNI',
          'paradero'] = 'RONDIZZONI'
metro.loc[metro['paradero'] == 'UNION LATINO AMERICANA',
          'paradero'] = 'UNION LATINOAMERICANA'

# %% Let me know if there are stations not in the database

available_stations = list(metro['paradero'])

stations_missing = []
for i, station in enumerate(stations):
    if not station in available_stations:
        stations_missing.append(station)

print(f'{len(stations_missing)} Stations Not Meassured:')
_ = [print(f'\t{i+1}. {station}')
     for i, station in enumerate(stations_missing)]

# %% Define te graph signal as a vector with length nodes

signal = np.zeros_like(stations, dtype=float)

for value, station in zip(metro['TOTAL'], metro['paradero']):
    graph_idx = [station == station_graph for station_graph in stations]

    signal[graph_idx] = float(value)

# Map signal to a color
cmap = matplotlib.colormaps.get_cmap('viridis')
normalized_signal = signal / np.max(signal)
colors = cmap(normalized_signal)

# %% Plot signal in the graph
fig, ax = plt.subplots(figsize=(10, 7))

nx.draw_networkx_edges(G, pos, node_size=20, ax=ax)
im = nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=20, ax=ax)

cbar = fig.add_axes([0.92, 0.12, 0.05, 0.75])
cbar = plt.colorbar(im, cax=cbar, ax=ax,
                    label='Promedio Diario\nBajadas de Metro', ticks=[0, 0.5, 1])
cbar.set_ticklabels([f'{label:.0f}' for label in [
                    0, np.amax(signal)/2, np.amax(signal)]])
