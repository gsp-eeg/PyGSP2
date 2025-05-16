r"""EEGRasP module.
==============

Contains the class EEGrasp which is used to analyze EEG signals
based graph signal processing.
"""

import mne
import numpy as np


class EEGrasp():
    """Class containing functionality to analyze EEG signals.

    Parameters
    ----------
    data : ndarray
        2D or 3D array. Where the first dim are channels and the second is
        samples. If 3D, the first dimension is trials.
    eeg_pos : ndarray
        Position of the electrodes.
    ch_names : ndarray | list
        Channel names.

    Notes
    -----
    Gaussian Kernel functionality overlapping with PyGSP2 toolbox. This has
    been purposefully added.
    """

    def __init__(self, data=None, coordinates=None, labels=None):
        """Parameters
        ----------
        data : ndarray | mne.Evoked | mne.BaseRaw | mne.BaseEpochs | None
            2D array. Where the first dim are channels and the second is
            samples. If 3D, the first dimension is trials. If an mne object is
            passed, the data will be extracted from it along with the
            coordinates and labels of the channels. If `None`, the class will
            be initialized without data. Default is `None`.
        coordinates : ndarray | list | None
            N-dim array or list with position of the electrodes. Dimensions mus
            coincide with the number of channels in `data`. If not provided the
            class instance will not have coordinates associated with the
            nodes. Some functions will not work without this information but
            can be provided later. Default is `None`.
        labels : ndarray | list | None
            Channel names. If not provided the class instance will not have
            labels associated with the nodes. Some functions will not work
            without this information but can be provided later. If `None` then
            the labels will be set to a range of numbers from 0 to the number
            of channels in the data. Default is `None`.
        """
        # Detect if data is a mne object
        if self._validate_MNE(data):
            self._init_from_mne(data)
        else:
            self.data = data
            self.coordinates = coordinates
            self.labels = labels
        self.distances = None
        self.graph_weights = None
        self.graph = None

    def _init_from_mne(self, data):
        """Initialize EEGrasp attributes from the MNE object.

        Parameters
        ----------
        data : any
            Object to be checked if it is an instance of the valid MNE objects
            allowed by the EEGrasp toolbox.
        """
        info = data.info
        self.data = data.get_data()
        self.coordinates = np.array(
            [pos for _, pos in info.get_montage().get_positions()['ch_pos'].items()])
        self.labels = info.ch_names

    def _validate_MNE(self, data):
        """Check if the data passed is a MNE object and extract the data and
        coordinates.

        Parameters
        ----------
        data : any

            Object to be checked if it is an instance of the valid MNE objects
            allowed by the EEGrasp toolbox.
        """
        is_mne = False
        if isinstance(data, (mne.Epochs, mne.Evoked, mne.io.Raw)):
            is_mne = True

        return is_mne

    def euc_dist(self, pos):
        """Calculate euclidean distance.
        %(eegrasp.utils.euc_dist).
        """
        from .utils import euc_dist
        return euc_dist(pos)

    def compute_distance(self, coordinates=None, method='Euclidean', normalize=True):
        """Compute distance.
        %(eegrasp.utils.compute_distance).
        """
        # If passed, use the coordinates argument
        if coordinates is None:
            coordinates = self.coordinates

        from .utils import compute_distance
        self.distances = compute_distance(coordinates=coordinates, method=method,
                                          normalize=normalize)
        return self.distances

    def gaussian_kernel(self, x, sigma=0.1):
        """Calculate Gaussian Kernel.
        %(eegrasp.graph_creation.gaussian_kernel).
        """
        from .graph import gaussian_kernel
        return gaussian_kernel(x, sigma)

    def compute_graph(self, W=None, epsilon=.5, sigma=.1, distances=None, graph=None,
                      coordinates=None):
        """Compute graph.
        %(eegrasp.graph_creation.compute_graph).
        """
        if W is None:
            distances = self.distances

        from .graph import compute_graph
        self.graph, self.graph_weights = compute_graph(W=W, epsilon=epsilon,
                                                       sigma=sigma, distances=distances,
                                                       graph=graph,
                                                       coordinates=coordinates)
        return self.graph

    def interpolate_channel(self, missing_idx: int | list[int] | tuple[int], graph=None,
                            data=None):
        """Interpolates channel.
        %(eegrasp.interpolate.interpolate_channel).
        """
        # Check if values are passed or use the instance's
        if data is None:
            data = self.data
        if graph is None:
            graph = self.graph

        from .interpolate import interpolate_channel
        return interpolate_channel(missing_idx, graph=graph, data=data)

    def fit_epsilon(self, missing_idx: int | list[int] | tuple[int], data=None,
                    distances=None, sigma=0.1):
        """Find the best distance to use as threshold.
        %(eegrasp.graph.fit_epsilon).
        """
        # Check if values are passed or use the instance's
        if data is None:
            data = self.data
        if distances is None:
            distances = self.distances

        if data is None or distances is None:
            raise TypeError('Check data or W arguments.')

        from .graph import fit_epsilon
        fit_epsilon(missing_idx=missing_idx, data=data, distances=distances,
                    sigma=sigma)

    def fit_sigma(self, missing_idx: int | list[int] | tuple[int], data=None,
                  distances=None, epsilon=0.5, min_sigma=0.1, max_sigma=1., step=0.1):
        """Find the best parameter for the gaussian kernel.
        %(eegrasp.graph.fit_sigma).
        """
        # Check if values are passed or use the instance's
        if data is None:
            data = self.data
        if distances is None:
            distances = self.distances

        if data is None or distances is None:
            raise TypeError('Check data or W arguments.')

        from .graph import fit_sigma
        return fit_sigma(missing_idx=missing_idx, data=data, distances=distances,
                         epsilon=epsilon, min_sigma=min_sigma, max_sigma=max_sigma,
                         step=step)

    def learn_graph(self, Z=None, a=0.1, b=0.1, gamma=0.04, maxiter=1000, w_max=np.inf,
                    mode='Average', data=None, **kwargs):
        """Learns graph using PyGSP2.
        %(eegrasp.graph.learn_graph).
        """
        if Z is None:
            data = self.data

        from .graph import learn_graph
        return learn_graph(Z=Z, a=a, b=b, gamma=gamma, maxiter=maxiter, w_max=w_max,
                           mode=mode, data=data, **kwargs)

    def plot(self, graph=None, signal=None, coordinates=None, labels=None, montage=None,
         colorbar=True, axis=None, clabel='Edge Weights', kind='topoplot', show_names=True,
         edge_percent=100, **kwargs):
    #def plot(self, graph=None, signal=None, coordinates=None, labels=None, montage=None,
    #         colorbar=True, axis=None, clabel='Edge Weights', kind='topoplot',
    #         show_names=True, **kwargs):
        """Plot graph over the eeg montage.
        %(eegrasp.viz.plot_graph)s.
        """
        from .viz import plot_graph
        return plot_graph(eegrasp=self, graph=graph, signal=signal,
                          coordinates=coordinates, labels=labels, montage=montage,
                          colorbar=colorbar, axis=axis, clabel=clabel, kind=kind,
                          show_names=show_names,edge_percent=edge_percent, **kwargs)