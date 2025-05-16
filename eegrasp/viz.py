r"""Viz.
========

Define default values and functions used for plotting function in eegrasp main module.
"""

import dataclasses

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.channels.layout import _auto_topomap_coords
from pygsp2 import graphs


@dataclasses.dataclass
class PlottingDefaults:
    """Class containing default values and functions for eegrasp plotting functions.

    Attributes
    ----------
    DEFAULT_CMAP : str.
        Default colormap for plotting.
    DEFAULT_VERTEX_COLOR : str.
        Default color for the vertices.
    DEFAULT_SPHERE : str.
        Default sphere for plotting.
    DEFAULT_ALPHAN : float.
        Default alpha value for the vertices.
    DEFAULT_VERTEX_SIZE : float.
        Default size for the vertices.
    DEFAULT_POINTSIZE : float.
        Default point size for plotting.
    DEFAULT_LINEWIDTH : float.
        Default line width for plotting.
    DEFAULT_EDGE_WIDTH : float.
        Default edge width for plotting.
    DEFAULT_EDGE_COLOR : str.
        Default edge color for plotting.
    DEFAULT_ALPHAV : float.
        Default alpha value for the vertices.
    """

    DEFAULT_CMAP: str = 'Spectral_r'
    DEFAULT_VERTEX_COLOR: str = 'teal'
    DEFAULT_SPHERE: str = 'eeglab'
    DEFAULT_ALPHAN: float = 0.5
    DEFAULT_VERTEX_SIZE: float = 50.
    DEFAULT_POINTSIZE: float = 0.5
    DEFAULT_LINEWIDTH: float = 1.
    DEFAULT_EDGE_WIDTH: float = 2.
    DEFAULT_EDGE_COLOR: str = 'black'
    DEFAULT_ALPHAV: float = 1.

    def load_defaults(self, kwargs):
        """Return dictionary with added default values for plotting functions if parameters have not
        been set.

        Parameters
        ----------
        kwargs : dict.
            Dictionary containing the variables to be updated.

        Returns
        -------
        kwargs : dict.
            Dictionary with default values added.
        """
        for key, value in self.__dict__.items():
            true_key = key.lower().rsplit('default_', maxsplit=1)[-1]
            if true_key not in kwargs:
                kwargs[true_key] = value
        return kwargs


def _update_locals(kwargs, local_vars):
    """Update local variables with the kwargs dict.

    Parameters.
    ----------
    kwargs : dict.
        Dictionary containing the variables to be updated.
    local_vars : list.
        List of local variables to be updated.
    """
    new_kwargs = kwargs.copy()
    for key in kwargs.keys():
        if key in local_vars:
            local_vars[key] = kwargs[key]
            del new_kwargs[key]
    return new_kwargs


def _separate_kwargs(kwargs, names):
    """Separate kwargs into two dictionaries based on names on vars."""
    var1 = {}
    var2 = {}
    for key in kwargs.keys():
        if key in names:
            var1[key] = kwargs[key]
        else:
            var2[key] = kwargs[key]
    return var1, var2


def plot_graph(eegrasp=None, graph: graphs.Graph | None = None, signal=None,
               coordinates=None, labels=None, montage=None, colorbar=True, axis=None,
               clabel='Edge Weights', kind='topoplot', show_names=True, edge_percent=100, **kwargs):
    
    """Plot the graph over the eeg montage.

    Parameters
    ----------
    eegrasp : EEGrasp object.
        Instance of the EEGrasp class. If `None` (default), the other parameters mus be specified. 
    graph : PyGSP2 Graph object | None.
        If `None` (default) the instance's graph will be used. If a `PyGSP2 Graph` object is passed, it will
        be used to plot the graph.
    signal : ndarray | list | None.
        If `None` (default), vertices will have different size depending on their weighted degree and the edges
        will have a different color depending on the connection strength between vertices. If a list or ndarray
        is passed, the vertices will have a different color depending on the signal passed.
    coordinates : ndarray | list | None.
        If `None`, the instance's coordinates will be used.
    labels : list | ndarray | None.
        Labels to be plotted with vertices. If `None`, the instance's labels will be used.
    montage : str | mne RawBase | mne EpochsBase | mne EvokedBase | None.
        If `None`, the instance's coordinates will be used to build a custom montage. Since it
        will only use the coordinateds tu build the custom montage, the sphere outline will
        not be adjusted to contain the electrodes. If a string is used, it will try to build
        a montage from the standard built-in mne library. If a ` mne DigiMontage` Class is
        used it will plot the sensors using the given montage and set sphere parameter to
        `None` when using `mne.viz.plot_sensors`.
    colorbar : bool.
        If True (default), a colorbar will be plotted.
    clabel : str.
        Label for the colorbar. Default is 'Edge Weights'.
    axis : matplotlib axis object | None.
        If `None` (default), a new figure will be created.
    kind : str.
        Kind of plot to use. Options are 'topoplot' and '3d'. Default is topoplot
    edge_percent : int, optional
        Percentage of strongest edges (by weight) to display. Values must be between 0 and 100. Default is 100 (no filtering).
        
    %(pygsp2.plot)s
    %(mne.viz.plot_sensors)s

    Returns
    -------
    figure : matplotlib figure.
        Figure object.
    axis : matplotlib axis.
        Axis object.

    Notes
    -----
    Any argument from `mne.viz.plot_sensors` and `pygsp2.plot` can be passed to the function.
    Default parameters can be found in `PlottingDefaults` class in `eegrasp.viz`.

    See Also
    --------
    * pygsp2 function `pygsp2.plotting.plot`
    * mne function `mne.viz.plot_sensors`
    """

    edge_percent = kwargs.pop('edge_percent', edge_percent)
    if not (0 < edge_percent <= 100):
        raise ValueError(f"edge_percent must be in (0, 100], got {edge_percent}")
    
    # Import EEGrasp class to verify instance
    from .eegrasp import EEGrasp

    # Check eegrasp instance
    if not isinstance(eegrasp, (type(None), EEGrasp)):
        raise TypeError('eegrasp must be an instance of EEGrasp class or None')

    # Load default values
    default_values = PlottingDefaults()
    original_kwargs = kwargs.copy()
    # Add default values for plotting
    kwargs = default_values.load_defaults(kwargs)
    cmap = kwargs['cmap']

    # Separate kwargs for pygsp2 and mne
    pygsp_arg_list = graphs.Graph.plot.__code__.co_varnames
    mne_arg_list = mne.viz.plot_sensors.__code__.co_varnames

    kwargs_pygsp_plot, kwargs = _separate_kwargs(kwargs, pygsp_arg_list)
    kwargs_mne_plot, kwargs = _separate_kwargs(kwargs, mne_arg_list)

    # Raise exemption if kwargs is not empty
    
    if len(kwargs) > 0:
        raise ValueError(f'Invalid arguments: {list(kwargs.keys())}')
    # 🩺 Parche quirúrgico: solo permitimos kwargs válidos para matplotlib
    ####################################################################################
    #ALLOWED_EXTRA_KWARGS = {'vmin', 'vmax', 'cmap', 'alpha', 'linewidths', 'edgecolors'}
    #for key in list(kwargs):
    #    if key not in ALLOWED_EXTRA_KWARGS:
    #        raise ValueError(f"Invalid argument passed to graph.plot(): {key}")
    ####################################################################################

    # Handle variables if not passed
    if graph is None:
        graph = eegrasp.graph
    ##########################################################
    if signal is None:
        W_full = (eegrasp.graph if eegrasp is not None else graph).W.toarray()
        avg_in = np.sum(W_full, axis=0)
        avg_in /= np.max(avg_in)
        vertex_color_base = plt.get_cmap(cmap)(avg_in)
        kwargs_pygsp_plot['vertex_color'] = vertex_color_base
    ##########################################################

    if coordinates is None:
        coordinates = eegrasp.coordinates

    ########################################################################
    # Optional: Edge filtering by relative intensity threshold
    #
    # This block applies an edge filter based on a user-defined percentile.
    # If edge_percent < 100, only connections in the top edge_percent%
    # of weight values will be shown. Default is 100, meaning no filtering.
    #
    # This does not modify the original graph object. It creates a temporary
    # visualization-only graph with the same node positions and labels.
    ########################################################################    
    if 0 < edge_percent < 100:
        try:
            W = graph.W.toarray()
            threshold = np.percentile(W[W > 0], 100 - edge_percent)
            W_filtered = np.where(W >= threshold, W, 0)
            from pygsp2.graphs import Graph  # Local import for clarity
            filtered_graph = Graph(W_filtered)
            filtered_graph.set_coordinates(coordinates)
            graph = filtered_graph
    
        except Exception as e:
            print(f"[EEGrasp] Warning: Failed to apply edge_percent filtering: {e}")
    ########################################################################

    if axis is None:
        fig = plt.figure()
        if kind == 'topoplot':
            axis = fig.add_subplot(111)
        elif kind == '3d':
            axis = fig.add_subplot(111, projection='3d')
    else:
        fig = axis.get_figure()

    if labels is None:
        labels = eegrasp.labels

    if montage is None:
        ch_pos = dict(zip(labels, coordinates))
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    elif isinstance(montage, str):
        try:
            montage = mne.channels.make_standard_montage(montage)
            labels = montage.ch_names
            kwargs_mne_plot['sphere'] = None
        except ValueError:
            print(
                f'{montage} Montage not found. Creating custom montage based on eegrasp.coordinates...'
            )
            fig, axis = plot_graph(eegrasp, graph=graph, signal=signal,
                                   coordinates=coordinates, labels=labels,
                                   montage=montage, colorbar=colorbar, axis=axis,
                                   clabel=clabel, kind=kind, show_names=show_names,
                                   **original_kwargs)
            return fig, axis
    else:
        kwargs_mne_plot['sphere'] = None

    if signal is None and 'edge_color' not in original_kwargs:
        
        # Plot edge color and width depending on edge weights
        # This is the first internal call to plot(), used only to suppress PyGSP2's default colorbar
        edge_weights = graph.get_edge_list()[2]
        normalized_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())
        kwargs_pygsp_plot['edge_color'] = plt.get_cmap(cmap)(normalized_weights)
        kwargs_pygsp_plot['edge_width'] = normalized_weights  # Proporcional, sin escala absoluta
        ####################################################################################################

    # if vertex size was not given, use weighted degree
    if 'vertex_size' not in original_kwargs:
        degree = np.array(graph.dw, dtype=float)
        degree /= np.max(degree)
        original_kwargs['vertex_size'] = degree
        fig, axis = plot_graph(eegrasp, graph, signal, coordinates, labels, montage,
                               colorbar, axis, clabel, kind, show_names,
                               **original_kwargs)
        return fig, axis

    if isinstance(signal, (list, np.ndarray)):

        norm_signal = np.array(signal, dtype=float)
        norm_signal -= np.min(norm_signal)
        norm_signal /= np.max(norm_signal)
        kwargs_pygsp_plot['edge_color'] = plt.get_cmap(cmap)(norm_signal)
    #################################################################################################################
    #if colorbar and signal is not None:
     #   cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axis, label=clabel)
      #  cbar.set_ticks([0, 0.5, 1])
       # cbar.ax.set_yticklabels(np.round([0, np.max(signal) / 2, np.max(signal)], 2))
    #################################################################################################################
    # Only add a single colorbar manually at top-level call
    if colorbar and not original_kwargs.get('_disable_colorbar', False):
        edge_c = kwargs_pygsp_plot.get('edge_color')
        vertex_c = kwargs_pygsp_plot.get('vertex_color')
    
        if isinstance(edge_c, np.ndarray):
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array(edge_c)
            sm.set_clim(vmin=0, vmax=1)
            cbar = fig.colorbar(sm, ax=axis, label='Edge weights')
        elif isinstance(vertex_c, np.ndarray):
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array(vertex_c)
            sm.set_clim(vmin=0, vmax=1)
            cbar = fig.colorbar(sm, ax=axis, label='Node strength')
        elif signal is not None:
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array(signal)
            sm.set_clim(vmin=np.min(signal), vmax=np.max(signal))
            cbar = fig.colorbar(sm, ax=axis, label=clabel)
    #################################################################################################################
    # Plot the montage
    if kind == 'topoplot':

        info = mne.create_info(labels, sfreq=250, ch_types='eeg')
        info.set_montage(montage)

        xy = _auto_topomap_coords(info, None, True, to_sphere=True,
                                  sphere=kwargs_mne_plot['sphere'])
        graph.set_coordinates(xy)
        figure = mne.viz.plot_sensors(info, kind='topomap', show_names=show_names,
                                      ch_type='eeg', axes=axis, show=False,
                                      **kwargs_mne_plot)
        figure, axis = graph.plot(ax=axis, colorbar=colorbar, **kwargs_pygsp_plot)

    elif kind == '3d':

        info = mne.create_info(labels, sfreq=250, ch_types='eeg')
        info.set_montage(montage)
        eeg_pos = montage.get_positions()['ch_pos']
        eeg_pos = np.array([pos for _, pos in eeg_pos.items()])

        dev_head_t = info['dev_head_t']
        eeg_pos = mne.transforms.apply_trans(dev_head_t, eeg_pos)
        graph.set_coordinates(eeg_pos)

        figure = mne.viz.plot_sensors(info, kind='3d', show_names=True, axes=axis,
                                      show=False, **kwargs_mne_plot)
        figure, axis = graph.plot(ax=axis, colorbar=colorbar, **kwargs_pygsp_plot)

    return (figure, axis)


if __name__ == '__main__':
    defaults = PlottingDefaults()
    print(defaults.__dict__)