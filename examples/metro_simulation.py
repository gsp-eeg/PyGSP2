""" Simulation of the distribution of a signal over the
metro network. The starting conditions is a graph signal
with only one positive integer value bigger.The plots 
or animation show how this signal distributes over the
network using:
.. math:: y = AD^-1x

"""
# %%
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
from utils import make_metro_graph

# If set to true make animation,
# otherwise store each frame as png
MAKE_ANIMATION = True

# %% Load graph and compute adjacency and node degree
G = make_metro_graph()
pos = {node: (G.nodes[node]['y'], G.nodes[node]['x']) for node in G.nodes}

W = nx.adjacency_matrix(G).toarray()
D = np.diag(W.sum(1))
D_inv = np.linalg.inv(D)

# %% Initialize parameters
NSTEPS = 30  # Arbitrary units, steps
INIT_VALUE = 5000  # Initial conditions
signal = np.zeros([NSTEPS, len(W)])
signal[0, np.random.randint(0, len(W), 1)] = INIT_VALUE

# %% Simulate advancing through the graph
# As a physical distribution network

print('Prerforming simulation...')

for i in np.arange(1, NSTEPS):
    signal[i, :] = W@D_inv@signal[i-1, :]

print('Finished.')

# %% Make animation
if MAKE_ANIMATION:
    # Initiate figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    cmap = plt.get_cmap('viridis')

    # Draw edges and nodes
    im = nx.draw_networkx_edges(G, pos, node_size=20, ax=ax)
    im = nx.draw_networkx_nodes(G, pos, node_color='gray', node_size=20, ax=ax)
    ax.set_title('T = 0')
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 0.5, 1], label='Signal')
    cbar.set_ticklabels([0, np.amax(signal)/2, np.amax(signal)])

    # Define function for animation

    def update(frame):
        """Function that defines what happens in each frame."""

        idxs = ((np.where(signal[frame, :] > 0)[0]).astype(int))
        nodelist = list(np.array(G)[idxs])
        colors = cmap(signal[frame, idxs] / INIT_VALUE)

        nx.draw_networkx_nodes(
            G, pos, node_color='gray', node_size=20, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color=colors, nodelist=nodelist,
                               node_size=20, ax=ax)
        ax.set_title(f'T = {frame}')

    anim = animation.FuncAnimation(fig, update,
                                   frames=np.arange(0, len(signal)),
                                   interval=50)

    # saving to m4 using ffmpeg writer
    writervideo = animation.FFMpegWriter(fps=5)

    anim.save('metro_simulation.gif', writer=writervideo)

else:
    try:
        os.mkdir('metro_simulation/')
    except FileExistsError:
        print(
            'Warning: It seems like this folder already exists. Overwritting...')

    # Initiate figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    cmap = plt.get_cmap('viridis')

    # Draw edges and nodes
    im = nx.draw_networkx_edges(G, pos, node_size=20, ax=ax)
    im = nx.draw_networkx_nodes(G, pos, node_color='gray', node_size=20, ax=ax)
    ax.set_title('T = 0')
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 0.5, 1], label='Signal')
    cbar.set_ticklabels([0, np.amax(signal)/2, np.amax(signal)])

    # Iterate through each graph signal
    for i, s in enumerate(signal):

        idxs = (np.where(s > 0)[0]).astype(int)
        nodelist = list(np.array(G)[idxs])
        colors = cmap(s[idxs] / INIT_VALUE)

        im = nx.draw_networkx_nodes(
            G, pos, node_color='gray', node_size=20, ax=ax)
        im = nx.draw_networkx_nodes(G, pos, node_color=colors, nodelist=nodelist,
                                    node_size=20, ax=ax)
        ax.set_title(f'T = {i}')

        fig.savefig(f'metro_simulation/{i}.png')
