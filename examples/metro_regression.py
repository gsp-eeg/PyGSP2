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
import pandas as pd
import numpy as np
from unidecode import unidecode
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from examples.utils import make_metro_graph
from pygsp import graphs, learning


def plot_signal_in_graph(G, signal, label='Signal'):
    """Function to plot signal in graph using networkx.
    Parameters:
    G: Networkx Graph.
    signal: 1d array. Should have the same length as number of nodes 
    in G.
    label: String. Lables to be displayed in colorbar.
    """

    # Map signal to a color
    cmap = matplotlib.colormaps.get_cmap('viridis')

    normalized_signal = signal / np.max(signal)
    colors = cmap(normalized_signal)

    # Initialize figure
    fig, ax = plt.subplots(figsize=(10, 7))
    # Draw edges and nodes
    nx.draw_networkx_edges(G, pos, node_size=20, ax=ax)
    im = nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=20, ax=ax)

    # Add Colorbar
    cbar = fig.add_axes([0.92, 0.12, 0.05, 0.75])
    cbar = plt.colorbar(im, cax=cbar, ax=ax,
                        label=label, ticks=[0, 0.5, 1])
    # Set labels to adjust original signal
    cbar.set_ticklabels([f'{label:.0f}' for label in [
                        0, np.amax(signal)/2, np.amax(signal)]])

    return fig, ax


commutes = pd.read_excel('viajes_bajada.xlsb',
                         sheet_name='bajadas_prom_laboral')

# %% Load graph
G = make_metro_graph()
pos = {node: (G.nodes[node]['y'], G.nodes[node]['x']) for node in G.nodes}

# Extract adjacency matrix
W = nx.adjacency_matrix(G).toarray()
G_pygsp = graphs.Graph(W)
# Node degree matrix
D = np.diag(W@np.ones(len(W)))
# Compute inverse for later
D_inv = np.linalg.inv(D)


# Convert to uppercase
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

# %% Define te graph signal as a vector with length nodes

signal = np.zeros_like(stations, dtype=float)

for value, station in zip(metro['TOTAL'], metro['paradero']):
    graph_idx = [station == station_graph for station_graph in stations]

    signal[graph_idx] = float(value)

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
