"""Purification of pair tables (functional ANOVA) and pair grids nested in the main-effect grids.

A pair term :math:`f_{ab}(x_a, x_b)` is *purified* with respect to a weight table :math:`W` (the training
weight of each cell) when every row and every column has weighted mean zero:

.. math::

    \\sum_j W_{ij} f_{ab}(i, j) = 0 \\;\\; \\forall i, \\qquad \\sum_i W_{ij} f_{ab}(i, j) = 0 \\;\\; \\forall j.

The mass removed from the table is moved to the main effects, :math:`f_a(i) \\mathrel{+}= r_i` and
:math:`f_b(j) \\mathrel{+}= c_j`, so the prediction of every cell is unchanged:
:math:`f_a + f_b + f_{ab}` before = after (Lengerich, Tan, Chang, Hooker and Caruana, 2020, *Purifying
interaction effects with the functional ANOVA*, AISTATS).

Why RabMV needs it: when :math:`x_b` was not measured for a case, RabMV replaces every term that involves
:math:`x_b` by its average over :math:`x_b`. For a purified table the average of the pair term over
:math:`x_b` *given* :math:`x_a` is exactly zero,

.. math::

    \\mathbb{E}[f_{ab}(x_a, x_b) \\mid x_a] = \\frac{\\sum_j W_{ij} f_{ab}(i, j)}{\\sum_j W_{ij}} = 0,

so giving 0 to the pair is the exact conditional marginalization of that term (the expectation is over the
empirical distribution of :math:`x_b` given the coarse bin of :math:`x_a`, among the training cases with both
concepts measured). The main effect :math:`f_b` is treated as before: it is centred, so 0 is its
*unconditional* average; it equals the conditional average :math:`\\mathbb{E}[f_b(x_b) \\mid x_a]` only when
:math:`x_b` does not depend on the measured concepts. Without purification, a pair table would hide part of
the main effects of :math:`x_a`, and dropping it would also remove that part.

The algorithm alternates row and column centring (block coordinate projections). Each step removes the
weighted row (or column) means; the sequence converges to the purified table because it is the alternating
projection onto two linear subspaces (von Neumann), and the prediction of every cell — including cells with
weight 0 — is invariant at every step.
"""
import numpy as np


def purify(table, weights, tol=1e-12, max_iter=10_000):
    """Purify a pair table with respect to cell weights.

    Parameters
    ----------
    table : numpy.ndarray of shape (na, nb)
        Values of the pair term per cell.
    weights : numpy.ndarray of shape (na, nb)
        Non-negative training weight of each cell.
    tol : float
        Stop when the largest row or column correction is below ``tol`` (absolute, on the logit scale).
    max_iter : int

    Returns
    -------
    pure : numpy.ndarray of shape (na, nb)
        Purified table (``pure + rows[:, None] + cols[None, :] == table`` up to rounding).
    rows : numpy.ndarray of shape (na,)
        Mass moved to the first feature.
    cols : numpy.ndarray of shape (nb,)
        Mass moved to the second feature.
    """
    T = np.array(table, dtype=float, copy=True)
    W = np.asarray(weights, dtype=float)
    if T.shape != W.shape or T.ndim != 2:
        raise ValueError('table and weights must be matrices of the same shape')
    if np.any(W < 0):
        raise ValueError('weights must be non-negative')
    rows, cols = np.zeros(T.shape[0]), np.zeros(T.shape[1])
    wr, wc = W.sum(1), W.sum(0)
    for _ in range(int(max_iter)):
        r = np.divide((W * T).sum(1), wr, out=np.zeros_like(wr), where=wr > 0)
        T -= r[:, None]
        rows += r
        c = np.divide((W * T).sum(0), wc, out=np.zeros_like(wc), where=wc > 0)
        T -= c[None, :]
        cols += c
        if max(np.abs(r).max(initial=0.0), np.abs(c).max(initial=0.0)) < tol:
            break
    return T, rows, cols


def nested_cuts(coarse, fine):
    """Snap each coarse cut to the nearest fine cut, so that every fine bin lies inside one coarse bin.

    Parameters
    ----------
    coarse : numpy.ndarray
        Cuts of the pair grid (``quantile_cuts`` with ``interaction_bins``).
    fine : numpy.ndarray
        Cuts of the main-effect grid of the same feature.

    Returns
    -------
    numpy.ndarray
        Increasing subset of ``fine``.
    """
    coarse, fine = np.asarray(coarse, float), np.asarray(fine, float)
    if coarse.size == 0 or fine.size == 0:
        return fine[:0].copy()
    idx = np.abs(fine[None, :] - coarse[:, None]).argmin(1)
    return np.unique(fine[idx])


def coarse_of_fine(fine, coarse):
    """Coarse-bin index of each fine bin, when the coarse cuts are a subset of the fine cuts.

    Bin ``k`` of a grid with cuts ``e`` holds ``e[k-1] <= x < e[k]`` (``searchsorted(e, x, 'right')``);
    its lower limit (−∞ for ``k = 0``) identifies the coarse bin.

    Returns
    -------
    numpy.ndarray of int, shape (fine.size + 1,)
    """
    lower = np.concatenate([[-np.inf], np.asarray(fine, float)])
    return np.searchsorted(np.asarray(coarse, float), lower, side='right')
