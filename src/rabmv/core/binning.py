"""Bins of each feature: distinct values or weighted quantiles (unchanged from EBMP v1)."""
import numpy as np


def quantile_cuts(x, max_bins, w=None):
    """Cut points of a feature: its distinct values or, if there are many, weighted quantiles.

    Parameters
    ----------
    x : numpy.ndarray of shape (n,)
    max_bins : int
        Maximum number of bins.
    w : numpy.ndarray of shape (n,), optional
        Sample weights: cases with weight 0 are ignored and an integer weight is equivalent to
        repeating the case.

    Returns
    -------
    numpy.ndarray
        Increasing cuts; the bin of a value is ``searchsorted(cuts, value, 'right')``.
    """
    if w is not None:
        x, w = x[w > 0], w[w > 0]
    u = np.unique(x)
    if len(u) <= max_bins:
        return (u[:-1] + u[1:]) / 2.0                          # one bin per observed value
    if w is None:
        q = np.quantile(x, np.linspace(0, 1, max_bins + 1)[1:-1])
    else:                                                       # quantiles of the weighted distribution
        order = np.argsort(x, kind='stable')
        xs, cum = x[order], np.cumsum(w[order])
        targets = np.linspace(0, 1, max_bins + 1)[1:-1] * cum[-1]
        q = xs[np.minimum(np.searchsorted(cum, targets, side='left'), len(xs) - 1)]
    q = np.unique(q)
    return q[q > u[0]]                                         # no empty bins at the start
