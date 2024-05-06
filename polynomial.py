from numba import njit
import numpy as np

@njit
def legval(x, n):
    res = np.zeros((n,) + x.shape)
    res[0] = 1
    res[1] = x
    for i in range(2, n):
        res[i] = ((2 * i - 1) * x * res[i - 1] - (i - 1) * res[i - 2]) / i
    return res

@njit
def polyval(p, x):
    y = np.zeros((p.shape[1], x.shape[0]))
    for i in range(len(p)):
        y = y * x + p[i]
    return y


def radial(coefnorm, rhotab, k, rho):
    return coefnorm[k, np.newaxis] * polyval(rhotab.T[:, k, np.newaxis], rho)


@njit
def angular(m, theta):
    return np.cos(m.reshape(-1, 1) * theta)


@njit
def logsumexp(values, index=None):
    """Stole from scipy.special.logsumexp

    Parameters
    ----------
    values : array_like Input array.

    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned.
    """
    a_max = np.max(values)
    s = np.sum(np.exp(values - a_max))
    return np.log(s) + a_max
