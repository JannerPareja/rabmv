"""Step of each shape-function update (no numerical changes from EBMP v1).

Strategy pattern: ``SPLIT_STRATEGIES`` maps the value of the ``split`` hyperparameter to the function
that chooses the segments. ``'optimal'`` is the optimal partition by dynamic programming (RabMV's
contribution) and ``'greedy'`` reproduces the greedy tree of the standard EBM for the ablation study.
"""
import numpy as np


def optimal_segmentation(G, H, N, max_leaves, reg_lambda, min_samples_leaf):
    """Optimal piecewise Newton step over contiguous segments (dynamic programming), for K bags at once.

    Among all partitions of the B ordered bins into at most ``max_leaves`` contiguous segments with at
    least ``min_samples_leaf`` cases each, chooses the one that maximizes the gain
    :math:`\\sum_s G_s^2/(H_s+\\lambda)` and returns, for each bin, the Newton value of its segment,
    :math:`-G_s/(H_s+\\lambda)`.

    Parameters
    ----------
    G, H, N : numpy.ndarray of shape (K, B)
        Sum of gradients, sum of hessians and number of training cases per bin and bag.
    max_leaves : int
        Maximum number of segments (L).
    reg_lambda : float
        L2 regularization of each segment value (in hessian units).
    min_samples_leaf : int
        Minimum cases per segment.

    Returns
    -------
    numpy.ndarray of shape (K, B)
        Step value for each bin (0 if no partition is valid).

    Notes
    -----
    Bellman recurrence: ``V[l, e] = max_{s<e} V[l-1, s] + g(s, e)``, with ``g(s, e)`` the gain of the
    segment of bins ``[s, e)``. Cost O(K · L · B²).
    """
    K, B = G.shape
    zero = np.zeros((K, 1))
    Gc = np.concatenate([zero, np.cumsum(G, 1)], 1)            # cumulative sums (K, B+1)
    Hc = np.concatenate([zero, np.cumsum(H, 1)], 1)
    Nc = np.concatenate([zero, np.cumsum(N, 1)], 1)
    Gs = Gc[:, None, :] - Gc[:, :, None]                        # segment [s, e): index (K, s, e)
    Hs = Hc[:, None, :] - Hc[:, :, None]
    Ns = Nc[:, None, :] - Nc[:, :, None]
    s_idx, e_idx = np.arange(B + 1)[:, None], np.arange(B + 1)[None, :]
    valid = (e_idx > s_idx)[None] & (Ns >= min_samples_leaf) & (Hs > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        gain = np.where(valid, Gs ** 2 / (Hs + reg_lambda), -np.inf)

    V = [gain[:, 0, :]]                                         # one segment: [0, e)
    arg = [None]
    for _ in range(1, max_leaves):
        cand = V[-1][:, :, None] + gain                          # (K, s, e)
        arg.append(np.argmax(cand, axis=1))
        V.append(np.max(cand, axis=1))
    finals = np.stack([v[:, B] for v in V], 1)                  # (K, L): total gain with l segments
    step = np.zeros((K, B))
    for k in range(K):
        t = int(np.argmax(finals[k]))                          # index of the chosen number of segments
        if not np.isfinite(finals[k, t]):
            continue                                            # no valid segment: no update
        e = B
        while True:                                             # reconstruction of the segments
            s = 0 if t == 0 else int(arg[t][k, e])
            g, h = Gc[k, e] - Gc[k, s], Hc[k, e] - Hc[k, s]
            step[k, s:e] = -g / (h + reg_lambda)
            if t == 0:
                break
            e, t = s, t - 1
    return step


def greedy_segmentation(G, H, N, max_leaves, reg_lambda, min_samples_leaf):
    """Step over contiguous segments chosen greedily (ablation variant of the standard EBM).

    Reproduces the logic of a small best-first tree (like InterpretML's EBM): it starts from a single
    segment and, while there are fewer than ``max_leaves`` segments, applies the cut (of any segment)
    that most increases the Newton gain. Each decision is locally optimal, but the result may not be:
    its gain never exceeds that of ``optimal_segmentation``.

    Parameters
    ----------
    G, H, N : numpy.ndarray of shape (K, B)
        Sum of gradients, sum of hessians and number of training cases per bin and bag.
    max_leaves : int
        Maximum number of segments (L).
    reg_lambda : float
        L2 regularization of each segment value.
    min_samples_leaf : int
        Minimum cases per segment.

    Returns
    -------
    numpy.ndarray of shape (K, B)
        Step value for each bin (0 if not even a single segment is valid).

    Notes
    -----
    Cost O(K · L · B) per step, versus O(K · L · B²) for dynamic programming.
    """
    K, B = G.shape
    zero = np.zeros((K, 1))
    Gc = np.concatenate([zero, np.cumsum(G, 1)], 1)
    Hc = np.concatenate([zero, np.cumsum(H, 1)], 1)
    Nc = np.concatenate([zero, np.cumsum(N, 1)], 1)
    step = np.zeros((K, B))
    for k in range(K):
        def gain(a, b):
            """Newton gain of segment [a, b) (−inf if invalid)."""
            g, h, n = Gc[k, b] - Gc[k, a], Hc[k, b] - Hc[k, a], Nc[k, b] - Nc[k, a]
            return g * g / (h + reg_lambda) if (n >= min_samples_leaf and h > 0) else -np.inf

        if not np.isfinite(gain(0, B)):
            continue                                            # not even a single segment is valid
        segments = [(0, B)]
        while len(segments) < max_leaves:
            best = (0.0, None, None)                            # (gain increase, segment, cut)
            for i, (a, b) in enumerate(segments):
                base = gain(a, b)
                for c in range(a + 1, b):
                    increase = gain(a, c) + gain(c, b) - base
                    if increase > best[0]:
                        best = (increase, i, c)
            if best[1] is None:
                break                                           # no cut improves the gain
            a, b = segments.pop(best[1])
            segments += [(a, best[2]), (best[2], b)]
        for a, b in segments:
            step[k, a:b] = -(Gc[k, b] - Gc[k, a]) / (Hc[k, b] - Hc[k, a] + reg_lambda)
    return step


def binary_step(G, H, N, max_leaves, reg_lambda, min_samples_leaf):
    """Optimal step for a two-bin feature (closed-form special case of the segmentation).

    Compares a single segment (common shift) with two segments (one value per bin) and chooses the
    one with the larger Newton gain; equivalent to ``optimal_segmentation`` with B = 2, without the
    O(B²) table.

    Returns
    -------
    numpy.ndarray of shape (K, 2)
    """
    Gt, Ht, Nt = G.sum(1), H.sum(1), N.sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        one = np.where((Nt >= min_samples_leaf) & (Ht > 0), Gt ** 2 / (Ht + reg_lambda), -np.inf)
        ok2 = (N >= min_samples_leaf).all(1) & (H > 0).all(1) & (max_leaves >= 2)
        two = np.where(ok2, (G ** 2 / (H + reg_lambda)).sum(1), -np.inf)
        v_one = -Gt / (Ht + reg_lambda)
        v_two = -G / (H + reg_lambda)
    step = np.zeros_like(G)
    uses_two = ok2 & (two >= one)
    uses_one = ~uses_two & np.isfinite(one)
    step[uses_two] = v_two[uses_two]
    step[uses_one] = v_one[uses_one, None]
    return step


SPLIT_STRATEGIES = {'optimal': optimal_segmentation, 'greedy': greedy_segmentation}
