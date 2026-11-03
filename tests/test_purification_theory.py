"""Numerical checks of the statements in docs/theory/purification.md. Synthetic data only."""
import numpy as np
import pytest

from rabmv import RabMVClassifier
from rabmv.core.purification import purify


def _wls_residual(T, W):
    """Independent computation: weighted least squares of T on row + column indicators, on the support."""
    na, nb = T.shape
    ii, kk = np.nonzero(W > 0)
    A = np.zeros((ii.size, na + nb))
    A[np.arange(ii.size), ii] = 1.0
    A[np.arange(ii.size), na + kk] = 1.0
    sw = np.sqrt(W[ii, kk])
    coef = np.linalg.lstsq(A * sw[:, None], T[ii, kk] * sw, rcond=None)[0]
    return ii, kk, T[ii, kk] - A @ coef


def test_purify_equals_the_weighted_least_squares_residual():
    """Theorem 1: the purified table is the W-orthogonal residual onto additive tables."""
    rng = np.random.default_rng(10)
    for _ in range(20):
        T = rng.normal(size=(6, 5))
        W = rng.integers(0, 8, size=(6, 5)).astype(float)
        W[rng.random(W.shape) < 0.2] = 0
        pure, r, c = purify(T, W)
        ii, kk, res = _wls_residual(T, W)
        assert np.allclose(pure[ii, kk], res, atol=1e-9)


def test_purified_table_is_unique_on_a_disconnected_support():
    """Theorem 1: two blocks with no shared row or column; the table is unique on the support, while the row
    and column shifts can trade a constant inside each block."""
    rng = np.random.default_rng(11)
    W = np.zeros((4, 4))
    W[:2, :2] = rng.uniform(1, 3, (2, 2))
    W[2:, 2:] = rng.uniform(1, 3, (2, 2))
    T = rng.normal(size=(4, 4))
    pure, r, c = purify(T, W)
    ii, kk, res = _wls_residual(T, W)
    assert np.allclose(pure[ii, kk], res, atol=1e-9)
    shifted = T + np.r_[1.0, 1.0, -2.0, -2.0][:, None] - np.r_[0.0, 0.0, 0.0, 0.0][None, :]
    assert np.allclose(purify(shifted, W)[0][ii, kk], pure[ii, kk], atol=1e-9)
    assert np.allclose(pure + r[:, None] + c[None, :], T, atol=1e-12)        # off-support cells: invariance


def test_purify_converges_geometrically():
    """Theorem 2: the largest correction decreases geometrically and the tolerance is reached quickly."""
    rng = np.random.default_rng(12)
    T = rng.normal(size=(8, 8))
    W = rng.uniform(0.1, 5, size=(8, 8))
    _, kk, target = _wls_residual(T, W)
    errors = []
    for it in range(1, 9):
        pure = purify(T, W, tol=0.0, max_iter=it)[0]
        errors.append(np.sqrt(np.sum(W.ravel() * (pure.ravel() - target) ** 2)))
    ratios = np.array(errors[1:]) / np.array(errors[:-1])
    assert np.all(ratios < 1) and ratios.max() < 0.9
    pure = purify(T, W, tol=1e-12, max_iter=10_000)[0]
    assert np.allclose(pure.ravel(), target, atol=1e-10)


def test_dropping_an_unpurified_pair_loses_a_main_effect():
    """Proposition 5: a pair table that only depends on x_a holds part of the effect of x_a; purification
    moves all of it to the main effect, so setting the pair to 0 for a missing x_b keeps it."""
    g = np.array([-1.0, 0.0, 2.0])
    T = np.repeat(g[:, None], 4, axis=1)
    W = np.random.default_rng(13).uniform(1, 3, (3, 4))
    pure, r, c = purify(T, W)
    assert np.abs(pure).max() < 1e-10
    assert np.allclose(r[:, None] + c[None, :], T, atol=1e-10)          # all the mass is now additive
    assert np.allclose(r - r.mean(), g - g.mean(), atol=1e-10)          # and it is the effect of x_a


def test_purify_commutes_with_the_mean_over_bags():
    """Corollary 7: purification is linear, so the mean of purified bags is the purified mean."""
    rng = np.random.default_rng(14)
    W = rng.uniform(0, 4, (5, 3))
    bags = rng.normal(size=(7, 5, 3))
    mean_of_pure = np.mean([purify(b, W)[0] for b in bags], axis=0)
    assert np.allclose(mean_of_pure, purify(bags.mean(0), W)[0], atol=1e-9)


def test_marginal_main_effect_is_the_unconditional_mean():
    """Proposition 6: a missing x1 contributes 0, the unconditional (centred) mean of f1 over measured cases;
    when x1 depends on x0, the conditional mean of f1 given x0 differs from 0."""
    rng = np.random.default_rng(15)
    n = 6000
    x0 = rng.normal(size=n)
    x1 = (rng.random(n) < 1 / (1 + np.exp(-2 * x0))).astype(float)       # x1 depends strongly on x0
    eta = -2 + 0.5 * x0 + 1.5 * x1
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    X = np.column_stack([x0, x1])
    m = RabMVClassifier(missing='marginalize', n_bags=3, max_rounds=300, random_state=0).fit(X, y)
    C = m.contributions(X)
    assert abs(C[:, 1].mean()) < 1e-8                                   # centred: unconditional mean 0
    high, low = x0 > 1, x0 < -1
    assert C[high, 1].mean() - C[low, 1].mean() > 0.5                   # conditional means are not 0
    Xm = X.copy()
    Xm[:, 1] = np.nan
    assert np.all(m.contributions(Xm)[:, 1] == 0)


@pytest.mark.parametrize('seed', [16, 17])
def test_marginalized_logit_matches_the_definition(seed):
    """End to end: with x1 missing, the logit is the intercept plus the measured main effects only."""
    rng = np.random.default_rng(seed)
    n = 3000
    X = np.column_stack([rng.normal(size=n), rng.integers(0, 3, n).astype(float), rng.normal(size=n)])
    eta = -1.5 + X[:, 0] + 0.6 * X[:, 1] + 0.8 * X[:, 0] * (X[:, 1] == 2)
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    Xm = X.copy()
    Xm[rng.random(n) < 0.3, 1] = np.nan
    m = RabMVClassifier(missing='marginalize', interactions=2, n_bags=3, max_rounds=300, random_state=1).fit(Xm, y)
    miss = np.isnan(Xm[:, 1])
    p = X.shape[1]
    C = m.contributions(Xm[miss])
    pairs_with_1 = [p + t for t, pair in enumerate(m.interaction_pairs_) if 1 in pair]
    assert np.all(C[:, [1] + pairs_with_1] == 0)
    assert np.allclose(m.decision_function(Xm[miss]), m.intercept_ + C.sum(1))
