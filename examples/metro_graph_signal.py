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
import os
import sys
import pandas as pd
from unidecode import unidecode
import matplotlib.pyplot as plt
from examples.utils import make_metro_graph, metro_database_preprocessing, plot_signal_in_graph

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
