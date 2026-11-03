"""Unit tests of the estimator: the 158 tests of EBMP v1 (c1/ebm_propio.py), adapted to the rabmv package.

The tests are the same as in v1; only the names change. The four minimum tests of the course guide are
marked (i) to (iv); the others check the properties of the algorithm. They use synthetic data only.
"""
import itertools

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from rabmv import RabMVClassifier
from rabmv.core import binary_step, greedy_segmentation, optimal_segmentation, smooth_curve


def known_data(n=3000, seed=0):
    """Synthetic data with a known solution: logit = -2 + f0(x0) + 1.5·x1 (x1 binary); x2, x3 are noise.

    f0 is a step function: -1 if x0 < -0.5; 0 if -0.5 ≤ x0 < 0.5; 1.5 if x0 ≥ 0.5.
    """
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(-2, 2, n), rng.integers(0, 2, n), rng.normal(size=n), rng.normal(size=n)])
    f0 = np.select([X[:, 0] < -0.5, X[:, 0] < 0.5], [-1.0, 0.0], 1.5)
    logit = -2 + f0 + 1.5 * X[:, 1]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y, f0


# ----------------------------------------------------------------------------- (i) learns
def test_i_learns_the_known_solution():
    """(i) Recovers the step function and the effect of the binary feature, and ignores the noise."""
    X, y, f0 = known_data()
    m = RabMVClassifier(random_state=1).fit(X, y)
    c = m.contributions(X)
    f0_c = f0 - f0.mean()
    assert np.corrcoef(c[:, 0], f0_c)[0, 1] > 0.97                  # shape of f0
    jump = c[X[:, 1] == 1, 1].mean() - c[X[:, 1] == 0, 1].mean()
    assert abs(jump - 1.5) < 0.35                                    # effect of x1 ≈ 1.5 on the logit
    assert m.term_importances_[2] < 0.25 * m.term_importances_[0]    # noise with almost no importance
    assert m.term_importances_[3] < 0.25 * m.term_importances_[0]
    Xt, yt, _ = known_data(seed=7)
    oracle = roc_auc_score(yt, known_data(seed=7)[2] + 1.5 * Xt[:, 1])
    assert roc_auc_score(yt, m.predict_proba(Xt)[:, 1]) > oracle - 0.02


# ----------------------------------------------------------------------------- (ii) seed
def test_ii_reproducible_with_the_same_seed():
    """(ii) Same seed → identical predictions; another seed → other bag splits."""
    X, y, _ = known_data(n=800)
    a = RabMVClassifier(random_state=3).fit(X, y).predict_proba(X)
    b = RabMVClassifier(random_state=3).fit(X, y).predict_proba(X)
    c = RabMVClassifier(random_state=4).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


# ----------------------------------------------------------------------------- (iii) invalid inputs
@pytest.mark.parametrize('change', [
    lambda X, y: (np.where(np.arange(X.size).reshape(X.shape) == 5, np.nan, X), y),   # empty value
    lambda X, y: (np.where(np.arange(X.size).reshape(X.shape) == 5, np.inf, X), y),   # infinite value
    lambda X, y: (X, np.arange(len(y)) % 3),                                         # three classes
    lambda X, y: (X, np.zeros(len(y))),                                              # a single class
    lambda X, y: (X, np.linspace(0, 1, len(y))),                                     # continuous target
    lambda X, y: (X, y[:-1]),                                                        # different sizes
])
def test_iii_rejects_invalid_data(change):
    """(iii) Invalid data in fit raise ValueError with a message."""
    X, y, _ = known_data(n=200)
    Xm, ym = change(X, y)
    with pytest.raises(ValueError):
        RabMVClassifier(n_bags=1).fit(Xm, ym)


@pytest.mark.parametrize('params', [
    {'learning_rate': 0}, {'learning_rate': 1.5}, {'max_bins': 1}, {'max_leaves': 0},
    {'reg_lambda': -1}, {'validation_size': 0.7}, {'n_bags': 0}, {'interactions': -1},
    {'split': 'other'}, {'newton': 'yes'}, {'smoothing': -1.0}, {'smoothing': float('inf')},
])
def test_iii_rejects_invalid_hyperparameters(params):
    """(iii) Out-of-range hyperparameters raise ValueError."""
    X, y, _ = known_data(n=200)
    with pytest.raises(ValueError):
        RabMVClassifier(**params).fit(X, y)


def test_iii_invalid_prediction():
    """(iii) Predicting without fitting or with another number of columns fails with a clear error."""
    X, y, _ = known_data(n=200)
    with pytest.raises(NotFittedError):
        RabMVClassifier().predict_proba(X)
    m = RabMVClassifier(n_bags=1).fit(X, y)
    with pytest.raises(ValueError):
        m.predict_proba(X[:, :3])


# ----------------------------------------------------------------------------- (iv) clone
def test_iv_compatible_with_clone_and_pipeline():
    """(iv) clone copies hyperparameters without the fit; works in Pipeline and cross-validation."""
    m = RabMVClassifier(learning_rate=0.1, max_leaves=2, random_state=5)
    c = clone(m)
    assert c.get_params() == m.get_params() and not hasattr(c, 'term_scores_')
    X, y, _ = known_data(n=600)
    scores = cross_val_score(make_pipeline(StandardScaler(), c), X, y, cv=3, scoring='roc_auc')
    assert np.all(scores > 0.7)


def test_iv_passes_the_scikit_learn_checks():
    """(iv) Passes the official scikit-learn checks (check_estimator).

    Only exception: the exact equivalence between weights and repeated cases with random validation bags
    (the splits depend on which rows exist). Without random bags the equivalence holds: see
    test_weights_equal_repeated_cases.
    """
    exceptions = {'check_sample_weight_equivalence_on_dense_data':
                  'the validation bags are drawn over the rows that are present'}
    check_estimator(RabMVClassifier(n_bags=2, max_rounds=50), expected_failed_checks=exceptions)


# ----------------------------------------------------------------------------- properties of the algorithm
def _brute_force(G, H, N, L, lam, min_n):
    """Best gain by exhaustive enumeration of every partition into ≤ L segments."""
    B, best = len(G), -np.inf
    for n_seg in range(1, L + 1):
        for cuts in itertools.combinations(range(1, B), n_seg - 1):
            lim = (0, *cuts, B)
            segments = [(lim[i], lim[i + 1]) for i in range(n_seg)]
            if any(N[s:e].sum() < min_n or H[s:e].sum() <= 0 for s, e in segments):
                continue
            best = max(best, sum(G[s:e].sum() ** 2 / (H[s:e].sum() + lam) for s, e in segments))
    return best


@pytest.mark.parametrize('seed', range(25))
def test_segmentation_is_optimal(seed):
    """Dynamic programming reaches the optimum of the exhaustive enumeration (Bellman)."""
    rng = np.random.default_rng(seed)
    B, L = int(rng.integers(2, 9)), int(rng.integers(1, 5))
    G, H = rng.normal(size=(1, B)), rng.uniform(0.05, 1, (1, B))
    N = rng.integers(0, 15, (1, B)).astype(float)
    lam, min_n = float(rng.uniform(0, 2)), int(rng.integers(1, 12))
    step = optimal_segmentation(G, H, N, L, lam, min_n)[0]
    expected = _brute_force(G[0], H[0], N[0], L, lam, min_n)
    if not np.isfinite(expected):
        assert np.all(step == 0)
        return
    # gain of the returned solution: segments = contiguous blocks with the same value
    lim = [0] + [b for b in range(1, B) if step[b] != step[b - 1]] + [B]
    obtained = sum(G[0, s:e].sum() ** 2 / (H[0, s:e].sum() + lam) for s, e in zip(lim[:-1], lim[1:]))
    assert len(lim) - 1 <= L
    assert obtained == pytest.approx(expected, rel=1e-9)
    for s, e in zip(lim[:-1], lim[1:]):                                  # Newton value of each segment
        assert step[s] == pytest.approx(-G[0, s:e].sum() / (H[0, s:e].sum() + lam))


@pytest.mark.parametrize('seed', range(25))
def test_binary_step_matches_dynamic_programming(seed):
    """The closed form for binary features gives the same step as the optimal segmentation."""
    rng = np.random.default_rng(100 + seed)
    K = 4
    G, H = rng.normal(size=(K, 2)), rng.uniform(0.0, 1, (K, 2))
    N = rng.integers(0, 15, (K, 2)).astype(float)
    L, lam, min_n = int(rng.integers(1, 4)), float(rng.uniform(0, 2)), int(rng.integers(1, 12))
    np.testing.assert_allclose(binary_step(G, H, N, L, lam, min_n), optimal_segmentation(G, H, N, L, lam, min_n))


def test_contributions_explain_the_prediction_exactly():
    """intercept + Σ contributions = logit, case by case (exact local explanation)."""
    X, y, _ = known_data(n=500)
    for interactions in (0, 2):
        m = RabMVClassifier(interactions=interactions, n_bags=2).fit(X, y)
        np.testing.assert_allclose(m.intercept_ + m.contributions(X).sum(1), m.decision_function(X))
        assert len(m.term_names_) == X.shape[1] + len(m.interaction_pairs_)


def test_functions_are_centred():
    """Each shape function has mean 0 on the training data (identifiability)."""
    X, y, _ = known_data(n=500)
    m = RabMVClassifier(n_bags=2).fit(X, y)
    np.testing.assert_allclose(m.contributions(X).mean(0), 0, atol=1e-9)


def test_weights_equal_repeated_cases():
    """Without random bags, weight 2 equals repeating the case exactly, and weight 0 equals removing it."""
    X, y, _ = known_data(n=400)
    rng = np.random.default_rng(0)
    w = rng.integers(0, 3, len(y))
    kw = dict(n_bags=1, validation_size=0, max_rounds=40, reg_lambda=0.0, min_samples_leaf=1)
    a = RabMVClassifier(**kw).fit(X, y, sample_weight=w)
    b = RabMVClassifier(**kw).fit(np.repeat(X, w, 0), np.repeat(y, w))
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X), rtol=1e-7, atol=1e-9)


def test_early_stopping_reduces_rounds():
    """Early stopping stops each bag before the maximum and keeps its best round."""
    X, y, _ = known_data(n=800)
    m = RabMVClassifier(max_rounds=3000, early_stopping_rounds=20, n_bags=3).fit(X, y)
    assert np.all(m.best_rounds_ < 3000) and np.all(m.best_rounds_ >= 1)
    assert m.validation_history_.shape[1] == 3


# ----------------------------------------------------------------------------- ablation variants
def _gain(G, H, step, lam):
    """Newton gain of the partition implicit in a step (segments = blocks with the same value)."""
    B = len(step)
    lim = [0] + [b for b in range(1, B) if step[b] != step[b - 1]] + [B]
    return sum(G[s:e].sum() ** 2 / (H[s:e].sum() + lam) for s, e in zip(lim[:-1], lim[1:])), len(lim) - 1


@pytest.mark.parametrize('seed', range(40))
def test_greedy_never_beats_optimal(seed):
    """The greedy partition is valid and its gain never exceeds that of dynamic programming."""
    rng = np.random.default_rng(200 + seed)
    B, L = int(rng.integers(3, 12)), int(rng.integers(1, 5))
    G, H = rng.normal(size=(1, B)), rng.uniform(0.05, 1, (1, B))
    N = rng.integers(0, 15, (1, B)).astype(float)
    lam, min_n = float(rng.uniform(0, 2)), int(rng.integers(1, 12))
    greedy = greedy_segmentation(G, H, N, L, lam, min_n)[0]
    optimal = optimal_segmentation(G, H, N, L, lam, min_n)[0]
    if np.all(optimal == 0):
        return
    g_g, l_g = _gain(G[0], H[0], greedy, lam)
    g_o, _ = _gain(G[0], H[0], optimal, lam)
    assert l_g <= L
    assert g_g <= g_o + 1e-12


def test_greedy_can_be_suboptimal():
    """There is a case where greedy loses gain: the first cut blocks the best partition."""
    G, H, N = np.array([[2.0, -2.0, -3.0, 3.0, -3.0]]), np.ones((1, 5)), np.ones((1, 5))
    greedy = greedy_segmentation(G, H, N, 3, 0.0, 1)[0]
    optimal = optimal_segmentation(G, H, N, 3, 0.0, 1)[0]
    assert _gain(G[0], H[0], greedy, 0.0)[0] == pytest.approx(16.5)     # greedy
    assert _gain(G[0], H[0], optimal, 0.0)[0] == pytest.approx(21.0)    # optimal: 27 % more


@pytest.mark.parametrize('seed', range(20))
def test_greedy_with_two_segments_is_optimal(seed):
    """With at most two segments a single cut decides everything: greedy equals optimal."""
    rng = np.random.default_rng(300 + seed)
    B = int(rng.integers(2, 10))
    G, H = rng.normal(size=(3, B)), rng.uniform(0.05, 1, (3, B))
    N = rng.integers(1, 15, (3, B)).astype(float)
    lam, min_n = float(rng.uniform(0.01, 2)), int(rng.integers(1, 8))
    g = greedy_segmentation(G, H, N, 2, lam, min_n)
    o = optimal_segmentation(G, H, N, 2, lam, min_n)
    for k in range(3):
        assert _gain(G[k], H[k], g[k], lam)[0] == pytest.approx(_gain(G[k], H[k], o[k], lam)[0], rel=1e-9)


@pytest.mark.parametrize('variant', [dict(split='greedy'), dict(newton=False), dict(split='greedy', newton=False)])
def test_ablation_variants_learn(variant):
    """The ablation variants are valid models: they learn the known solution."""
    X, y, f0 = known_data()
    m = RabMVClassifier(random_state=1, n_bags=3, **variant).fit(X, y)
    assert np.corrcoef(m.contributions(X)[:, 0], f0)[0, 1] > 0.9
    np.testing.assert_allclose(m.intercept_ + m.contributions(X).sum(1), m.decision_function(X))


def test_defaults_are_those_of_rabmv():
    """By default RabMV uses dynamic programming and the Newton step (the variants are optional)."""
    m = RabMVClassifier()
    assert m.split == 'optimal' and m.newton is True
    X, y, _ = known_data(n=600)
    a = RabMVClassifier(n_bags=2).fit(X, y).predict_proba(X)
    b = RabMVClassifier(n_bags=2, split='optimal', newton=True).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)


# ----------------------------------------------------------------------------- EBMP v2: smoothing
def test_v2_smoothing_solved_by_hand():
    """Desk test: g = [1, 0, 1], W = [1, 1, 1], κ = 1 → f = [0.75; 0.5; 0.75].

    System (diag(W) + γ DᵀD) f = W g with γ = 1: [[2, −1, 0], [−1, 3, −1], [0, −1, 2]] f = [1, 0, 1].
    By symmetry f₁ = f₃ = a, f₂ = c: 2a − c = 1 and −2a + 3c = 0 → a = 3/4, c = 1/2.
    """
    np.testing.assert_allclose(smooth_curve([1.0, 0.0, 1.0], [1.0, 1.0, 1.0], 1.0), [0.75, 0.5, 0.75])


def test_v2_smoothing_respects_the_evidence_of_each_bin():
    """A bin with many cases barely changes; one with few moves closer to its neighbours."""
    f = smooth_curve([0.0, 2.0, 0.0], [100.0, 1.0, 100.0], 0.05)
    assert f[1] < 1.0 and abs(f[0]) < 0.05 and abs(f[2]) < 0.05
    g = np.array([0.3, -0.2, 0.5, 0.1])
    np.testing.assert_array_equal(smooth_curve(g, np.ones(4), 0.0), g)                 # κ = 0: unchanged
    np.testing.assert_array_equal(smooth_curve([1.0, 2.0], [1.0, 1.0], 5.0), [1.0, 2.0])   # binaries: unchanged


@pytest.mark.parametrize('seed', range(10))
def test_v2_smoothing_reduces_roughness(seed):
    """The solution is never rougher (Σ differences²) than the original curve and keeps its weighted mean."""
    rng = np.random.default_rng(seed)
    g, W = rng.normal(size=12), rng.uniform(0.5, 20, 12)
    f = smooth_curve(g, W, float(rng.uniform(0.1, 5)))
    assert np.sum(np.diff(f) ** 2) <= np.sum(np.diff(g) ** 2) + 1e-12
    assert np.average(f, weights=W + 1e-9) == pytest.approx(np.average(g, weights=W + 1e-9), abs=1e-9)


def test_v2_zero_smoothing_is_v1():
    """smoothing = 0 reproduces EBMP v1 bit for bit."""
    X, y, _ = known_data(n=600)
    a = RabMVClassifier(n_bags=2).fit(X, y).predict_proba(X)
    b = RabMVClassifier(n_bags=2, smoothing=0.0).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)


def test_v2_keeps_calibration_and_exact_explanation():
    """With smoothing: mean probability = observed proportion, centred functions and exact contributions."""
    X, y, f0 = known_data()
    m = RabMVClassifier(n_bags=3, smoothing=3.0).fit(X, y)
    assert m.predict_proba(X)[:, 1].mean() == pytest.approx(y.mean(), abs=1e-6)
    np.testing.assert_allclose(m.contributions(X).mean(0), 0, atol=1e-9)
    np.testing.assert_allclose(m.intercept_ + m.contributions(X).sum(1), m.decision_function(X))
    assert np.corrcoef(m.contributions(X)[:, 0], f0)[0, 1] > 0.95          # it still learns the shape


def test_v2_smooths_the_learned_functions():
    """The v2 shape functions are less rough than the v1 ones with the same data and seed."""
    X, y, _ = known_data(n=800)
    r = lambda m: np.sum(np.diff(m.term_scores_[0]) ** 2)  # noqa: E731
    assert r(RabMVClassifier(n_bags=2, smoothing=3.0).fit(X, y)) < r(RabMVClassifier(n_bags=2).fit(X, y))


def test_v2_passes_the_scikit_learn_checks():
    """EBMP v2 also passes check_estimator (with the same single documented exception)."""
    exceptions = {'check_sample_weight_equivalence_on_dense_data':
                  'the validation bags are drawn over the rows that are present'}
    check_estimator(RabMVClassifier(n_bags=2, max_rounds=50, smoothing=1.0), expected_failed_checks=exceptions)
