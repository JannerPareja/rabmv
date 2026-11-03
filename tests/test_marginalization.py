"""Tests of the declared core changes (phase 2d): marginalization and categorical columns.

With ``missing='marginalize'``, a NaN means "this case does not have this concept measured". It is
checked that: the case does not contribute to that feature's function nor receive its update, cuts and
centring use only measured cases, at prediction time it contributes 0, and the model recovers the truth
when the missingness pattern depends on the class (like the schooling of the 2016 attempts). Synthetic
data only.
"""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.utils.estimator_checks import check_estimator

from rabmv import RabMVClassifier
from rabmv.core import quantile_cuts


def data(n=3000, seed=0):
    """logit = -2 + f0(x0) + 1.5·x1; x2 is noise. f0 is a step function: -1, 0 or 1.5 depending on x0."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(-2, 2, n), rng.integers(0, 2, n), rng.normal(size=n)])
    f0 = np.select([X[:, 0] < -0.5, X[:, 0] < 0.5], [-1.0, 0.0], 1.5)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2 + f0 + 1.5 * X[:, 1])))).astype(int)
    return X, y, f0


def with_missing(X, y, column, fraction, only_class=None, seed=1):
    """Copy of X with NaN in a column (at random, or only in the cases of one class)."""
    rng = np.random.default_rng(seed)
    X = X.copy()
    eligible = np.ones(len(y), bool) if only_class is None else (y == only_class)
    X[eligible & (rng.random(len(y)) < fraction), column] = np.nan
    return X


def test_by_default_rejects_nan_like_v1():
    """Without marginalizing, a NaN is an error (v1 behaviour)."""
    X, y, _ = data(n=300)
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        RabMVClassifier().fit(X, y)


def test_an_unmeasured_value_contributes_zero_and_the_prediction_is_exact():
    """The contribution of an unmeasured feature is 0 and the logit is still the exact sum."""
    X, y, _ = data()
    Xn = with_missing(X, y, 0, 0.3)
    m = RabMVClassifier(missing='marginalize', random_state=1).fit(Xn, y)
    c = m.contributions(Xn)
    absent = np.isnan(Xn[:, 0])
    assert (c[absent, 0] == 0).all()
    assert (c[~absent, 0] != 0).any()
    np.testing.assert_allclose(m.decision_function(Xn), m.intercept_ + c.sum(1), atol=1e-12)


def test_cuts_and_centring_use_only_measured_cases():
    """The cuts are those of the measured cases and the function is centred on them."""
    X, y, _ = data()
    Xn = with_missing(X, y, 0, 0.3)
    m = RabMVClassifier(missing='marginalize', random_state=1).fit(Xn, y)
    measured = ~np.isnan(Xn[:, 0])
    np.testing.assert_array_equal(m.bin_edges_[0], quantile_cuts(Xn[measured, 0], m.max_bins, np.ones(measured.sum())))
    assert abs(m.contributions(Xn)[measured, 0].mean()) < 1e-10


def test_unmeasured_cases_do_not_influence_the_function():
    """Changing the class of unmeasured cases does not change that feature's function... except through others.

    Direct test: with a single feature (x0) and a single bag without early stopping, the function learned
    with the unmeasured cases is the same as the one learned removing them, and the intercept absorbs the
    difference in prevalence.
    """
    X, y, _ = data(n=1500)
    X1 = X[:, :1].copy()
    absent = np.random.default_rng(3).random(len(y)) < 0.4
    X1n = X1.copy()
    X1n[absent, 0] = np.nan
    params = dict(n_bags=1, validation_size=0.0, max_rounds=40, random_state=0)
    with_nan = RabMVClassifier(missing='marginalize', **params).fit(X1n, y)
    without = RabMVClassifier(**params).fit(X1[~absent], y[~absent])
    np.testing.assert_array_equal(with_nan.bin_edges_[0], without.bin_edges_[0])
    # same shape (up to a constant, because the starting intercept includes the unmeasured cases)
    diff = with_nan.term_scores_[0] - without.term_scores_[0]
    assert np.ptp(diff) < 0.15
    assert np.corrcoef(with_nan.term_scores_[0], without.term_scores_[0])[0, 1] > 0.99


def test_class_linked_missingness_does_not_create_a_shortcut():
    """If a feature is only missing in one class (like the 2016 schooling of attempts), the model does not learn it.

    With naive imputation (NaN → 0) missingness becomes a predictor of the class; with marginalization the
    contribution of the missing value is 0 and the shape of f0 is recovered.
    """
    X, y, f0 = data(n=4000, seed=5)
    Xn = with_missing(X, y, 0, 0.5, only_class=0, seed=6)
    absent = np.isnan(Xn[:, 0])
    marg = RabMVClassifier(missing='marginalize', random_state=2).fit(Xn, y)
    c = marg.contributions(Xn)
    assert (c[absent, 0] == 0).all()
    assert np.corrcoef(c[~absent, 0], f0[~absent] - f0[~absent].mean())[0, 1] > 0.95
    # contrast: imputing with 0 makes the missing value push towards "attempt"
    Xi = np.where(np.isnan(Xn), 0.0, Xn)
    naive = RabMVClassifier(random_state=2).fit(Xi, y)
    ci = naive.contributions(Xi)
    assert ci[absent, 0].mean() < -0.2                # the shortcut exists in the naive model
    assert abs(c[absent, 0].mean()) == 0.0             # and does not exist in the marginalized one


def test_recovers_the_truth_with_random_missingness():
    """With 30 % random missing values in x0, the shape and the discrimination are kept."""
    X, y, f0 = data()
    Xn = with_missing(X, y, 0, 0.3)
    m = RabMVClassifier(missing='marginalize', random_state=1).fit(Xn, y)
    measured = ~np.isnan(Xn[:, 0])
    c = m.contributions(Xn)
    assert np.corrcoef(c[measured, 0], f0[measured] - f0[measured].mean())[0, 1] > 0.97
    Xt, yt, f0t = data(seed=9)
    oracle = roc_auc_score(yt, f0t + 1.5 * Xt[:, 1])
    assert roc_auc_score(yt, m.predict_proba(Xt)[:, 1]) > oracle - 0.02


def test_weights_and_smoothing_with_missing_values():
    """Works with sample weights and with v2 smoothing; global calibration is preserved."""
    X, y, _ = data(n=2000)
    Xn = with_missing(X, y, 0, 0.3)
    w = np.where(y == 1, 3.0, 1.0)
    m = RabMVClassifier(missing='marginalize', smoothing=0.5, random_state=1).fit(Xn, y, sample_weight=w)
    p = m.predict_proba(Xn)[:, 1]
    assert abs(np.average(p, weights=w) - np.average(y, weights=w)) < 1e-6
    assert np.isfinite(m.term_importances_).all()


def test_clear_errors():
    """Column without any measured value and infinite values."""
    X, y, _ = data(n=400)
    Xv = X.copy()
    Xv[:, 2] = np.nan
    with pytest.raises(ValueError, match='no measured value'):
        RabMVClassifier(missing='marginalize').fit(Xv, y)
    Xi = X.copy()
    Xi[0, 0] = np.inf
    with pytest.raises(ValueError):
        RabMVClassifier(missing='marginalize').fit(Xi, y)
    with pytest.raises(ValueError, match='missing'):
        RabMVClassifier(missing='impute').fit(X, y)


def test_compatible_with_scikit_learn():
    """clone, cross-validation with NaN and the scikit-learn checks."""
    X, y, _ = data(n=1500)
    Xn = with_missing(X, y, 0, 0.3)
    m = clone(RabMVClassifier(missing='marginalize', max_rounds=50))
    assert m.get_params()['missing'] == 'marginalize'
    assert np.isfinite(cross_val_score(m, Xn, y, cv=3, scoring='roc_auc')).all()
    exceptions = {'check_sample_weight_equivalence_on_dense_data':      # the same documented exception as v1
                  'the validation bags are drawn over the rows that are present'}
    check_estimator(RabMVClassifier(missing='marginalize', max_rounds=30, n_bags=2), expected_failed_checks=exceptions)


# ----------------------------------------------------------------------------- categorical columns
def categorical_data(n=4000, seed=0):
    """A 4-category nominal feature whose effects do NOT follow the order of the codes, and a numeric one."""
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, 4, n).astype(float)
    effect = np.array([0.0, 1.5, -1.0, 1.2])[cat.astype(int)]
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + effect + 0.5 * x)))).astype(int)
    return np.column_stack([cat, x]), y, effect


def test_categorical_recovers_unordered_effects():
    """The per-category effects (which do not follow the order of the codes) are recovered."""
    X, y, effect = categorical_data()
    m = RabMVClassifier(categorical_features=[0], random_state=0).fit(X, y)
    assert np.corrcoef(m.contributions(X)[:, 0], effect - effect.mean())[0, 1] > 0.98
    assert m.categorical_features_ == [0]


def test_categorical_does_not_depend_on_how_categories_were_numbered():
    """Renumbering the categories (e.g. "subsidized" = 1 or = 5) changes nothing if the column is categorical.

    An ordered column does change, because each step groups "neighbouring" categories by an order that is
    arbitrary for a nominal feature. That is the reason to treat them as categorical.
    """
    X, y, _ = categorical_data()
    renumber = np.array([3.0, 0.0, 2.0, 1.0])                     # bijection of codes 0..3
    Xr = X.copy()
    Xr[:, 0] = renumber[X[:, 0].astype(int)]
    cat = RabMVClassifier(categorical_features=[0], random_state=0)
    np.testing.assert_allclose(cat.fit(X, y).predict_proba(X), clone(cat).fit(Xr, y).predict_proba(Xr),
                               rtol=0, atol=1e-10)
    ordered = RabMVClassifier(random_state=0)
    diff = np.abs(ordered.fit(X, y).predict_proba(X) - clone(ordered).fit(Xr, y).predict_proba(Xr)).max()
    assert diff > 1e-3


def test_categorical_with_missing_values():
    """Categorical features and marginalization together: unmeasured contributes 0 and effects are recovered."""
    X, y, effect = categorical_data(seed=2)
    Xn = with_missing(X, y, 0, 0.3, seed=4)
    m = RabMVClassifier(categorical_features=[0], missing='marginalize', random_state=0).fit(Xn, y)
    absent = np.isnan(Xn[:, 0])
    c = m.contributions(Xn)
    assert (c[absent, 0] == 0).all()
    assert np.corrcoef(c[~absent, 0], effect[~absent] - effect[~absent].mean())[0, 1] > 0.98


def test_categorical_features_are_not_smoothed():
    """v2 smoothing does not mix categories (they have no neighbours)."""
    X, y, _ = categorical_data(n=2000)
    plain = RabMVClassifier(categorical_features=[0], random_state=0).fit(X, y)
    smooth = RabMVClassifier(categorical_features=[0], smoothing=3.0, random_state=0).fit(X, y)
    np.testing.assert_allclose(np.ptp(smooth.term_scores_[0]), np.ptp(plain.term_scores_[0]), rtol=1e-12)


@pytest.mark.parametrize('categorical, message', [
    ([5], 'between 0 and 1'), ([0, 0], 'distinct'), (['a'], 'indices'), ('0', 'indices'),
])
def test_invalid_categorical_features(categorical, message):
    """Indices out of range, repeated or not integers."""
    X, y, _ = categorical_data(n=300)
    with pytest.raises(ValueError, match=message):
        RabMVClassifier(categorical_features=categorical).fit(X, y)


def test_categorical_with_too_many_categories():
    """A nominal feature with more categories than max_bins is an error; it is not grouped silently."""
    X, y, _ = data(n=300)
    with pytest.raises(ValueError, match='more categories than max_bins'):
        RabMVClassifier(categorical_features=[0], max_bins=8).fit(X, y)
