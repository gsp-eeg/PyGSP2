"""
Using the data available in
https://www.dtpm.cl/index.php/documentos/matrices-de-viaje.
This example also uses the edges defined in edges.txt

Compute the number of exits in a random metro station. Then,
compute the regression over the whole network and plot the
error of the regression. Here, Tikhonov regession is used.
Lastly, the average of neighboring nodes is also used to
compare the error of the regression.
"""
# %%
import sys
import os
import pandas as pd
import numpy as np
from unidecode import unidecode
import networkx as nx
import matplotlib.pyplot as plt
from examples.utils import make_metro_graph, plot_signal_in_graph, metro_database_preprocessing
from pygsp import graphs, learning

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
pos_list = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in G.nodes]

# Extract adjacency matrix
W = nx.adjacency_matrix(G).toarray()
G_pygsp = graphs.Graph(W)

# Node degree matrix
D = np.diag(G_pygsp.d)
# Compute inverse for later
D_inv = np.linalg.inv(D)

# Convert to uppercase
stations = [name.upper() for name in list(G.nodes)]
stations = [unidecode(station) for station in stations]

# %% Use only metro commutes in database
metro_commutes, signal = metro_database_preprocessing(commutes, stations)

# %% Choose a point in the network and delete its value to regress it

signal2 = signal.copy()
station_idx = np.random.randint(0, len(signal2))
# station_idx = 24

print(f'Deleted Station: {stations[station_idx]}')
signal2[station_idx] = np.nan
mask = np.ones(len(signal)).astype(bool)
mask[station_idx] = False

# Use tikhonov regression to recover the signal
recovered_signal = learning.regression_tikhonov(
    G_pygsp, signal2, mask, tau=0.5)

# Compute the average of the nodes around the missing value
average = (W@D_inv@signal)[station_idx]

print(f'Estimated: {recovered_signal[station_idx]:.2f}')
print(f'One-hop Average: {average:.2f}')
print(f'Real: {signal[station_idx]:.2f}')

# %% Compute error with each station
tikhonov_estimation = np.zeros_like(signal)
average_estimation = W@D_inv@signal

for i, s in enumerate(signal):
    # Allocate new signal
    signal2 = signal.copy()
    print(f'Deleted Station: {stations[i]}')

    # Delete value in the signal
    signal2[i] = np.nan
    mask = np.ones(len(signal)).astype(bool)
    mask[i] = False

    # Use tikhonov regression to recover the signal
    recovered_signal = learning.regression_tikhonov(
        G_pygsp, signal2, mask, tau=0.5)
    tikhonov_estimation[i] = recovered_signal[i]

abs_err = np.abs(tikhonov_estimation - signal)

# %% Plot Tikhonov error in graph
fig1, ax = plot_signal_in_graph(G, abs_err, label='Error absoluto')
ax.set_title('Error of Tikhonov Regression')

# %% Plot Average errorr in graph
# Change variable to error with average estimation
abs_err = np.abs(average_estimation - signal)

fig2, ax = plot_signal_in_graph(G, abs_err, label='Error absoluto')
ax.set_title(r'Error of $y = AD^{-1}x$')


plt.show()
