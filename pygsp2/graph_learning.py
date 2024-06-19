r"""
The :mod:`pygsp2.graph_learning` module provides functions
to estimate a graph based on a distance matrix Z.

.. autosummary::

    graph_log_degree

"""
import numpy as np
from scipy.spatial.distance import squareform
from pygsp2.utils import sum_squareform


def graph_log_degree(Z, a=1.0, b=1.0, w_0='zeros', w_max=np.inf, tol=1e-5,
                     maxiter=1000, gamma=0.04, verbose=True):
    r"""Learn graph from pairwise distances using negative log prior on nodes
    degrees.

    The minimization problem solved is:

    ..math::
    minimize_W sum(sum(W * Z)) - a * sum(log(sum(W))) + b * ||W||_F^2/2 + c * ||W-W_0||_F^2/2

    Parameters
    ----------
    Z : array
        Matrix with squared pairwise distances of nodes.
    a : float
        Log prior constant (larger a -> bigger weiights in W).
    b : float
        W||_F^2 prior constant (larger b -> W denser).
    w_0 : string or matrix
        String 'zeros' will set the w priors to be a 0 vector.  Otherwise a
        matrix with the priors can be passed.
    w_max : int or float
        Maximum value for the estimated W matrix.
    tol : float
        Tolerance to end the iteration.
    maxiter : int
        Maximum number of iterations.
    gamma : float
        Step size. Number between (0,1).
    verbose : bool
        If `True` print the number of iterations.

    Returns
    -------
    W: array
        Weighted adjacency matrix.

    References
    ----------
    Vassilis Kalofolias Proceedings of the 19th International Conference on
    Artificial Intelligence and Statistics, PMLR 51:920-929, 2016.

    Examples
    -------

    Create a graph

    >>> G = graphs.Graph([[0, 0, 0, 0],
                  [0, 0, 1, 1],
                  [0, 1, 0, 1],
                  [0, 1, 1, 0]])
    >>> signal = np.array([0, 1, 1, 1])

    Compute distance

    >>> kdt = spatial.KDTree(signal[:, None])
    >>> D, NN = kdt.query(signal[:, None], k=len(signal))

    Allocate distance array
    >>> Z = np.zeros((G.N, G.N))
    >>> for i, n in enumerate(NN):
    >>>    Z[i, n] = D[i]

    Learn graph
    >>> A = 0.2
    >>> B = 0.01

    >>> W = graph_learning.graph_log_degree(Z, A, B, w_max=1)

    Threshold adjacency matrix
    >>> W[W < 1e-1] = 0

    >>> plt.subplot(121)
    >>> plt.imshow(G.W.toarray())
    >>> plt.colorbar()

    >>> plt.subplot(122)
    >>> plt.imshow(W)
    >>> plt.colorbar()

    """

    # Transform to vector space
    z = squareform(Z)
    z = z / np.amax(z)

    N = len(Z)
    S, St = sum_squareform(N)

    if w_0 == 'zeros':
        w_0 = squareform(np.zeros_like(Z))
    elif isinstance(w_0, np.ndarray) and len(w_0.shape) == 2:
        w_0 = squareform(w_0)
    else:
        raise TypeError(
            "w_0 is not 'zeros' or a 2d numpy array. Check parameter type.")

    w = w_0.copy()
    v_n = S @ w

    for i in np.arange(maxiter):

        Y = w - gamma * (2 * b * w + St @ v_n)
        y = v_n + gamma * (S @ w)

        P = np.minimum(w_max, np.maximum(0, Y - 2 * gamma * z))
        p = y - gamma * a * \
            ((y/(gamma * a)) + np.sqrt(((y/(gamma*a))**2) + (4/(gamma*a))))/2

        Q = P - gamma * (2 * b * P + St @ p)
        q = p + gamma * (S @ P)

        w_new = w - Y + Q
        v_new = v_n - y + q

        w = w_new.copy()
        v_n = v_new.copy()

        if (np.linalg.norm(- Y + Q) / np.linalg.norm(w) < tol) and \
                (np.linalg.norm(- y + q) / np.linalg.norm(v_n) < tol):
            break

    if verbose:
        print(f'Found solution after {i+1} iterations')

    return squareform(w)
