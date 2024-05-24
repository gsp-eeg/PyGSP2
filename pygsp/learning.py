# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.learning` module provides functions to solve learning problems.

Semi-supervized learning
========================

Those functions help to solve a semi-supervized learning problem, i.e., a
problem where only some values of a graph signal are known and the others shall
be inferred.

.. autosummary::

    regression_tikhonov
    classification_tikhonov
    classification_tikhonov_simplex

"""

import numpy as np
from scipy import sparse
from scipy.spatial.distance import squareform
from pygsp.utils import sum_squareform


def _import_pyunlocbox():
    try:
        from pyunlocbox import functions, solvers
    except Exception as e:
        raise ImportError('Cannot import pyunlocbox, which is needed to solve '
                          'this optimization problem. Try to install it with '
                          'pip (or conda) install pyunlocbox. '
                          'Original exception: {}'.format(e))
    return functions, solvers


def _to_logits(x):
    logits = np.zeros([len(x), np.max(x)+1])
    logits[range(len(x)), x] = 1
    return logits


def classification_tikhonov_simplex(G, y, M, tau=0.1, **kwargs):
    r"""Solve a classification problem on graph via Tikhonov minimization
    with simple constraints.

    The function first transforms :math:`y` in logits :math:`Y`, then solves

    .. math:: \operatorname*{arg min}_X \| M X - Y \|_2^2 + \tau \ tr(X^T L X)
              \text{ s.t. } sum(X) = 1 \text{ and } X >= 0,

    where :math:`X` and :math:`Y` are logits.

    Parameters
    ----------
    G : :class:`pygsp.graphs.Graph`
    y : array, length G.n_vertices
        Measurements.
    M : array of boolean, length G.n_vertices
        Masking vector.
    tau : float
        Regularization parameter.
    kwargs : dict
        Parameters for :func:`pyunlocbox.solvers.solve`.

    Returns
    -------
    logits : array, length G.n_vertices
        The logits :math:`X`.

    Examples
    --------
    >>> from pygsp import graphs, learning
    >>> import matplotlib.pyplot as plt
    >>>
    >>> G = graphs.Logo()
    >>> G.estimate_lmax()

    Create a ground truth signal:

    >>> signal = np.zeros(G.n_vertices)
    >>> signal[G.info['idx_s']] = 1
    >>> signal[G.info['idx_p']] = 2

    Construct a measurement signal from a binary mask:

    >>> rng = np.random.default_rng(42)
    >>> mask = rng.uniform(0, 1, G.n_vertices) > 0.5
    >>> measures = signal.copy()
    >>> measures[~mask] = np.nan

    Solve the classification problem by reconstructing the signal:

    >>> recovery = learning.classification_tikhonov_simplex(
    ...     G, measures, mask, tau=0.1, verbosity='NONE')

    Plot the results.
    Note that we recover the class with ``np.argmax(recovery, axis=1)``.

    >>> prediction = np.argmax(recovery, axis=1)
    >>> fig, ax = plt.subplots(2, 3, sharey=True, figsize=(10, 6))
    >>> _ = G.plot(signal, ax=ax[0, 0], title='Ground truth')
    >>> _ = G.plot(measures, ax=ax[0, 1], title='Measurements')
    >>> _ = G.plot(prediction, ax=ax[0, 2], title='Recovered class')
    >>> _ = G.plot(recovery[:, 0], ax=ax[1, 0], title='Logit 0')
    >>> _ = G.plot(recovery[:, 1], ax=ax[1, 1], title='Logit 1')
    >>> _ = G.plot(recovery[:, 2], ax=ax[1, 2], title='Logit 2')
    >>> _ = fig.tight_layout()

    """

    functions, solvers = _import_pyunlocbox()

    if tau <= 0:
        raise ValueError('Tau should be greater than 0.')

    y = y.copy()
    y[M == False] = 0
    Y = _to_logits(y.astype(int))
    Y[M == False, :] = 0

    def proj_simplex(y):
        d = y.shape[1]
        a = np.ones(d)
        idx = np.argsort(y)

        def evalpL(y, k, idx):
            return np.sum(y[idx[k:]] - y[idx[k]]) - 1

        def bisectsearch(idx, y):
            idxL, idxH = 0, d-1
            L = evalpL(y, idxL, idx)
            H = evalpL(y, idxH, idx)

            if L < 0:
                return idxL

            while (idxH-idxL) > 1:
                iMid = int((idxL + idxH) / 2)
                M = evalpL(y, iMid, idx)

                if M > 0:
                    idxL, L = iMid, M
                else:
                    idxH, H = iMid, M

            return idxH

        def proj(idx, y):
            k = bisectsearch(idx, y)
            lam = (np.sum(y[idx[k:]]) - 1) / (d - k)
            return np.maximum(0, y - lam)

        x = np.empty_like(y)
        for i in range(len(y)):
            x[i] = proj(idx[i], y[i])
        # x = np.stack(map(proj, idx, y))

        return x

    def smooth_eval(x):
        xTLx = np.sum(x * (G.L.dot(x)))
        e = M * ((M * x.T) - Y.T)
        l2 = np.sum(e * e)
        return tau * xTLx + l2

    def smooth_grad(x):
        return 2 * ((M * (M * x.T - Y.T)).T + tau * G.L * x)

    f1 = functions.func()
    f1._eval = smooth_eval
    f1._grad = smooth_grad

    f2 = functions.func()
    f2._eval = lambda x: 0  # Indicator functions evaluate to zero.
    f2._prox = lambda x, step: proj_simplex(x)

    step = 0.5 / (1 + tau * G.lmax)
    solver = solvers.forward_backward(step=step)
    ret = solvers.solve([f1, f2], Y.copy(), solver, **kwargs)
    return ret['sol']


def classification_tikhonov(G, y, M, tau=0):
    r"""Solve a classification problem on graph via Tikhonov minimization.

    The function first transforms :math:`y` in logits :math:`Y`, then solves

    .. math:: \operatorname*{arg min}_X \| M X - Y \|_2^2 + \tau \ tr(X^T L X)

    if :math:`\tau > 0`, and

    .. math:: \operatorname*{arg min}_X tr(X^T L X) \ \text{ s. t. } \ Y = M X

    otherwise, where :math:`X` and :math:`Y` are logits.
    The function returns the maximum of the logits.

    Parameters
    ----------
    G : :class:`pygsp.graphs.Graph`
    y : array, length G.n_vertices
        Measurements.
    M : array of boolean, length G.n_vertices
        Masking vector.
    tau : float
        Regularization parameter.

    Returns
    -------
    logits : array, length G.n_vertices
        The logits :math:`X`.

    Examples
    --------
    >>> from pygsp import graphs, learning
    >>> import matplotlib.pyplot as plt
    >>>
    >>> G = graphs.Logo()

    Create a ground truth signal:

    >>> signal = np.zeros(G.n_vertices)
    >>> signal[G.info['idx_s']] = 1
    >>> signal[G.info['idx_p']] = 2

    Construct a measurement signal from a binary mask:

    >>> rng = np.random.default_rng(42)
    >>> mask = rng.uniform(0, 1, G.n_vertices) > 0.5
    >>> measures = signal.copy()
    >>> measures[~mask] = np.nan

    Solve the classification problem by reconstructing the signal:

    >>> recovery = learning.classification_tikhonov(G, measures, mask, tau=0)

    Plot the results.
    Note that we recover the class with ``np.argmax(recovery, axis=1)``.

    >>> prediction = np.argmax(recovery, axis=1)
    >>> fig, ax = plt.subplots(2, 3, sharey=True, figsize=(10, 6))
    >>> _ = G.plot(signal, ax=ax[0, 0], title='Ground truth')
    >>> _ = G.plot(measures, ax=ax[0, 1], title='Measurements')
    >>> _ = G.plot(prediction, ax=ax[0, 2], title='Recovered class')
    >>> _ = G.plot(recovery[:, 0], ax=ax[1, 0], title='Logit 0')
    >>> _ = G.plot(recovery[:, 1], ax=ax[1, 1], title='Logit 1')
    >>> _ = G.plot(recovery[:, 2], ax=ax[1, 2], title='Logit 2')
    >>> _ = fig.tight_layout()

    """
    y = y.copy()
    y[M == False] = 0
    Y = _to_logits(y.astype(int))
    return regression_tikhonov(G, Y, M, tau)


def regression_tikhonov(G, y, M, tau=0):
    r"""Solve a regression problem on graph via Tikhonov minimization.

    The function solves

    .. math:: \operatorname*{arg min}_x \| M x - y \|_2^2 + \tau \ x^T L x

    if :math:`\tau > 0`, and

    .. math:: \operatorname*{arg min}_x x^T L x \ \text{ s. t. } \ y = M x

    otherwise.

    Parameters
    ----------
    G : :class:`pygsp.graphs.Graph`
    y : array, length G.n_vertices
        Measurements.
    M : array of boolean, length G.n_vertices
        Masking vector.
    tau : float
        Regularization parameter.

    Returns
    -------
    x : array, length G.n_vertices
        Recovered values :math:`x`.

    Examples
    --------
    >>> from pygsp import graphs, filters, learning
    >>> import matplotlib.pyplot as plt
    >>>
    >>> G = graphs.Sensor(N=100, seed=42)
    >>> G.estimate_lmax()

    Create a smooth ground truth signal:

    >>> filt = lambda x: 1 / (1 + 10*x)
    >>> filt = filters.Filter(G, filt)
    >>> rng = np.random.default_rng(42)
    >>> signal = filt.analyze(rng.normal(size=G.n_vertices))

    Construct a measurement signal from a binary mask:

    >>> mask = rng.uniform(0, 1, G.n_vertices) > 0.5
    >>> measures = signal.copy()
    >>> measures[~mask] = np.nan

    Solve the regression problem by reconstructing the signal:

    >>> recovery = learning.regression_tikhonov(G, measures, mask, tau=0)

    Plot the results:

    >>> fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(10, 3))
    >>> limits = [signal.min(), signal.max()]
    >>> _ = G.plot(signal, ax=ax1, limits=limits, title='Ground truth')
    >>> _ = G.plot(measures, ax=ax2, limits=limits, title='Measures')
    >>> _ = G.plot(recovery, ax=ax3, limits=limits, title='Recovery')
    >>> _ = fig.tight_layout()

    """

    if tau > 0:
        y = y.copy()
        y[M == False] = 0

        if sparse.issparse(G.L):

            def Op(x):
                return (M * x.T).T + tau * (G.L.dot(x))

            LinearOp = sparse.linalg.LinearOperator([G.N, G.N], Op)

            if y.ndim > 1:
                sol = np.empty(shape=y.shape)
                res = np.empty(shape=y.shape[1])
                for i in range(y.shape[1]):
                    sol[:, i], res[i] = sparse.linalg.cg(
                        LinearOp, y[:, i])
            else:
                sol, res = sparse.linalg.cg(LinearOp, y)

            # TODO: do something with the residual...
            return sol

        else:

            # Creating this matrix may be problematic in term of memory.
            # Consider using an operator instead...
            if type(G.L).__module__ == np.__name__:
                LinearOp = np.diag(M*1) + tau * G.L
            return np.linalg.solve(LinearOp, M * y)

    else:

        if np.prod(M.shape) != G.n_vertices:
            raise ValueError("M should be of size [G.n_vertices,]")

        indl = M
        indu = (M == False)

        Luu = G.L[indu, :][:, indu]
        Wul = - G.L[indu, :][:, indl]

        if sparse.issparse(G.L):
            sol_part = sparse.linalg.spsolve(Luu, Wul.dot(y[indl]))
        else:
            sol_part = np.linalg.solve(Luu, np.matmul(Wul, y[indl]))

        sol = y.copy()
        sol[indu] = sol_part

        return sol

def graph_log_degree(Z,a=1,b=1,w_0='zeroes',w_max=np.inf,tol=1e-5,maxiter=1000,gamma=0.04):
    r""" Learn graph from pairwise distances using negative log prior on nodes degrees

    The minimization problem solved is:

    ..math:: 
    minimize_W sum(sum(W * Z)) - a * sum(log(sum(W))) + b * ||W||_F^2/2 + c * ||W-W_0||_F^2/2

    Parameters
    ----------
    Z: array, matrix with squared pairwise distances of nodes
    a: float, Log prior constant (bigger a -> bigger weiights in W)
    b: float, W||_F^2 prior constant (bigger b -> more dense W)
    w_0: string or matrix, string 'zeroes' will set the w priors to be a 0 vector. 
         Otherwise a matrix with the priors can be passed.
    w_max: int or float, Maximum value for the estimated W matrix.
    tol: float, tolerance to end the iteration.
    maxiter: int, maximum number of iterations.
    gamma: float, step size. Number between (0,1)

    Returns
    -------
    W: array, weighted adjacency matrix

    References
    ----------
    Kalofolias, V. (2016). How to learn a graph from smooth signals (Version 1). arXiv. https://doi.org/10.48550/ARXIV.1601.02513

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

    >>> W = learning.graph_log_degree(Z, A, B, w_max=1)
    
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
    S,St = sum_squareform(N)

    if w_0 == 'zeroes':
        w_0 = squareform(np.zeros(Z.shape))

    w = w_0.copy()
    v_n = S@w

    for i in np.arange(maxiter):
    
        Y = w - gamma*(2*b*w + St@v_n)
        y = v_n + gamma * (S @ w)

        P = np.minimum(w_max,np.maximum(0,Y - 2*gamma*z))
        p = y - gamma*a*((y/(gamma*a)) + np.sqrt(((y/(gamma*a))**2) + (4 / (gamma*a))))/2 

        Q = P - gamma * (2*b*P + St@p)
        q = p + gamma * (S@P)

        w_new = w - Y + Q
        v_new = v_n - y + q

        w = w_new.copy()
        v_n = v_new.copy()

        if (np.linalg.norm(- Y + Q) / np.linalg.norm(w) < tol) and (np.linalg.norm(- y + q) / np.linalg.norm(v_n) < tol):
            break


    print(f'Found solution after {i} iterations')
    
    return squareform(w)


