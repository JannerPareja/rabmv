"""Final smoothing of an ordered shape function (EBMP v2, unchanged)."""
import numpy as np


def smooth_curve(g, W, kappa):
    """Smoothing of an ordered shape function: neighbouring bins borrow information.

    Solves the penalized least-squares problem (Whittaker smoother; order-0 P-splines of Eilers and
    Marx, 1996)

    .. math::

        \\min_f \\sum_b W_b (f_b - g_b)^2 + \\gamma \\sum_b (f_b - f_{b-1})^2,
        \\qquad \\gamma = \\kappa \\cdot \\bar W,

    whose solution is the tridiagonal linear system :math:`(\\operatorname{diag}(W) + \\gamma D^\\top D) f = W g`,
    with :math:`D` the first-difference matrix. A bin with few cases (small W) moves closer to its
    neighbours; one with many cases barely changes. ``kappa = 0`` returns ``g`` unchanged.

    Parameters
    ----------
    g : numpy.ndarray of shape (B,)
        Shape-function values per bin (learned by boosting).
    W : numpy.ndarray of shape (B,)
        Weight of each bin (sum of training-case weights).
    kappa : float
        Smoothing strength, relative to the mean weight per bin (≥ 0).

    Returns
    -------
    numpy.ndarray of shape (B,)
    """
    g = np.asarray(g, float)
    if kappa <= 0 or g.size <= 2:
        return g.copy()
    W = np.asarray(W, float) + 1e-9
    D = np.diff(np.eye(g.size), axis=0)
    return np.linalg.solve(np.diag(W) + kappa * W.mean() * (D.T @ D), W * g)
