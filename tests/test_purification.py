"""Purified interactions with marginalization (registry-aware pairs). Synthetic data only.

Checks the purification algorithm (rows and columns with weighted mean 0, predictions invariant), the nested
pair grid, and the model: pairs are fitted only on cases with both concepts measured, a pair with an
unmeasured concept contributes 0, which equals the conditional average of the pair over that concept, and the
v1 path (``missing='reject'``) is untouched.
"""
import numpy as np
import pytest

from rabmv import RabMVClassifier
from rabmv.core.purification import coarse_of_fine, nested_cuts, purify


def test_purify_rows_and_columns_have_weighted_mean_zero_and_cells_are_invariant():
    rng = np.random.default_rng(0)
    T = rng.normal(size=(5, 4))
    W = rng.integers(0, 6, size=(5, 4)).astype(float)
    W[2, :] = 0                                       # an empty row: its values only need invariance
    pure, r, c = purify(T, W)
    assert np.allclose((W * pure).sum(1), 0, atol=1e-10)
    assert np.allclose((W * pure).sum(0), 0, atol=1e-10)
    assert np.allclose(pure + r[:, None] + c[None, :], T, atol=1e-12)


def test_purify_of_an_additive_table_is_zero():
    """A table that is already a sum of row and column effects has no interaction."""
    a, b = np.array([1.0, -2.0, 0.5]), np.array([0.3, 0.0, -1.0, 2.0])
    W = np.ones((3, 4))
    pure, r, c = purify(a[:, None] + b[None, :], W)
    assert np.abs(pure).max() < 1e-10


def test_purify_is_linear():
    rng = np.random.default_rng(1)
    W = rng.uniform(0.1, 3, size=(4, 3))
    A, B = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
    assert np.allclose(purify(A + 2 * B, W)[0], purify(A, W)[0] + 2 * purify(B, W)[0], atol=1e-10)


def test_purify_rejects_bad_input():
    with pytest.raises(ValueError):
        purify(np.zeros((2, 2)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        purify(np.zeros((2, 2)), -np.ones((2, 2)))


def test_nested_cuts_are_a_subset_and_each_fine_bin_is_inside_one_coarse_bin():
    fine = np.linspace(-3, 3, 31)
    coarse = nested_cuts(np.array([-1.27, 0.04, 1.9]), fine)
    assert set(coarse) <= set(fine)
    m = coarse_of_fine(fine, coarse)
    x = np.random.default_rng(2).uniform(-4, 4, 5000)
    assert np.array_equal(m[np.searchsorted(fine, x, side='right')], np.searchsorted(coarse, x, side='right'))
    assert nested_cuts(np.array([]), fine).size == 0


def data(n=4000, seed=0):
    """logit = −2 + x0 + 0.8·x1 + 1.2·x0·x1 (a true interaction); x2 is noise."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.normal(size=n), rng.integers(0, 2, n).astype(float), rng.normal(size=n)])
    eta = -2 + X[:, 0] + 0.8 * X[:, 1] + 1.2 * X[:, 0] * X[:, 1]
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    return X, y


def fitted(X, y, **kw):
    return RabMVClassifier(missing='marginalize', interactions=2, n_bags=3, max_rounds=300, random_state=3,
                           **kw).fit(X, y)


def test_pairs_with_missing_values_fit_and_are_purified():
    X, y = data()
    Xm = X.copy()
    Xm[np.random.default_rng(4).random(len(y)) < 0.25, 1] = np.nan
    m = fitted(Xm, y)
    assert m.interaction_pairs_ and (0, 1) in m.interaction_pairs_
    t = len(m.bin_edges_) + m.interaction_pairs_.index((0, 1))
    na, nb = m.pair_bin_edges_[0].size + 1, m.pair_bin_edges_[1].size + 1
    both = ~np.isnan(Xm[:, 0]) & ~np.isnan(Xm[:, 1])
    Xp = m._bin_pairs(Xm)
    W = np.bincount(m._pair_cell(0, 1, Xp)[both], minlength=na * nb).reshape(na, nb)
    T = m.term_scores_[t].reshape(na, nb)
    assert np.allclose((W * T).sum(1), 0, atol=1e-8) and np.allclose((W * T).sum(0), 0, atol=1e-8)
    assert all(set(m.pair_bin_edges_[j]) <= set(m.bin_edges_[j]) for j in range(3))
    # exact additive explanation still holds, and an unmeasured concept zeroes its pairs
    C = m.contributions(Xm)
    assert np.allclose(m.decision_function(Xm), m.intercept_ + C.sum(1))
    miss = np.isnan(Xm[:, 1])
    assert np.all(C[miss, t] == 0) and np.all(C[miss, 1] == 0)
    assert np.isfinite(m.term_scores_std_[t]).all()


def test_zero_for_an_unmeasured_concept_is_the_conditional_average_of_the_pair():
    """For a case with x1 not measured, the pair contribution (0) equals the average of the pair over the
    training cases with both measured that share the coarse bin of x0."""
    X, y = data(seed=5)
    Xm = X.copy()
    Xm[np.random.default_rng(6).random(len(y)) < 0.3, 1] = np.nan
    m = fitted(Xm, y)
    t = len(m.bin_edges_) + m.interaction_pairs_.index((0, 1))
    both = ~np.isnan(Xm[:, 1])
    C = m.contributions(Xm[both])
    coarse0 = m._bin_pairs(Xm[both])[:, 0]
    for i in np.unique(coarse0):
        assert abs(C[coarse0 == i, t].mean()) < 1e-8


def test_purification_does_not_change_predictions_of_complete_cases(monkeypatch):
    """Same fit with and without the purification step: identical logits where every concept is measured;
    where x0 is missing, the purified model adds the conditional mean of the pair (moved to the main effect
    of x1), the unpurified one drops it."""
    X, y = data(seed=7)
    Xm = X.copy()
    Xm[np.random.default_rng(8).random(len(y)) < 0.2, 0] = np.nan
    purified = fitted(Xm, y)
    monkeypatch.setattr(RabMVClassifier, '_purify_pairs', lambda self, *a, **k: None)
    raw = fitted(Xm, y)
    complete = ~np.isnan(Xm).any(1)
    assert np.allclose(purified.decision_function(Xm[complete]), raw.decision_function(Xm[complete]), atol=1e-9)
    assert not np.allclose(purified.decision_function(Xm[~complete]), raw.decision_function(Xm[~complete]))


def test_without_missing_values_and_reject_mode_v1_is_untouched():
    """The reject (v1) path keeps its own pair grid (not nested) and is not purified."""
    X, y = data(n=1500, seed=9)
    r = RabMVClassifier(interactions=2, n_bags=2, max_rounds=100).fit(X, y)
    assert r.interaction_pairs_
    t = len(r.bin_edges_)
    a, b = r.interaction_pairs_[0]
    na, nb = r.pair_bin_edges_[a].size + 1, r.pair_bin_edges_[b].size + 1
    W = np.bincount(r._pair_cell(a, b, r._bin_pairs(X)), minlength=na * nb).reshape(na, nb)
    assert np.abs((W * r.term_scores_[t].reshape(na, nb)).sum(1)).max() > 1e-6   # not purified (v1)


def test_marginalized_pairs_keep_the_mean_probability():
    X, y = data(seed=10)
    Xm = X.copy()
    Xm[np.random.default_rng(11).random(len(y)) < 0.3, 1] = np.nan
    m = fitted(Xm, y, smoothing=0.5)
    assert abs(m.predict_proba(Xm)[:, 1].mean() - y.mean()) < 0.01
