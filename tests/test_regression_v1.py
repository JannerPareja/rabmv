"""Regression test: the rabmv package reproduces EBMP v1 EXACTLY.

It is compared against the frozen copy of v1 (tests/reference/ebm_propio_v1.py, the class used in
Competencia 1, SHA-256 fingerprint 0396E530B408B66A…; kept unchanged, in Spanish, as the historical
reference). With the same synthetic data, the same seed and the same hyperparameters, everything learned
and everything predicted must be identical bit for bit. It covers the default model and each ablation
variant (greedy split, no Newton, one bag, no early stopping, no regularization, interactions, v2
smoothing), with and without sample weights, and with both values of ``missing``: without any NaN,
marginalizing missing values does not change a single bit of the v1 model. The only exception, by design
(0.1.0.dev10): with interactions, ``missing='marginalize'`` nests the pair grid in the main-effect grid and
purifies the pairs (``test_purification.py``), so it is a different model; ``missing='reject'`` with
interactions is still v1.
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from rabmv import RabMVClassifier
from tests.reference.ebm_propio_v1 import EBMP as EBMPv1

REFERENCE = Path(__file__).parent / 'reference' / 'ebm_propio_v1.py'
FINGERPRINT_V1 = '0396e530b408b66a'

# (rabmv hyperparameters, the same hyperparameters with v1 names)
CONFIGURATIONS = [
    (dict(), dict()),
    (dict(split='greedy'), dict(paso='voraz')),
    (dict(newton=False), dict(newton=False)),
    (dict(n_bags=1), dict(n_bags=1)),
    (dict(validation_size=0.0, max_rounds=60), dict(validation_size=0.0, max_rounds=60)),
    (dict(reg_lambda=0.0), dict(reg_lambda=0.0)),
    (dict(interactions=3), dict(interactions=3)),
    (dict(smoothing=0.5), dict(suavizado=0.5)),
    (dict(max_leaves=4, min_samples_leaf=2, learning_rate=0.15),
     dict(max_leaves=4, min_samples_leaf=2, learning_rate=0.15)),
    (dict(max_bins=8, max_leaves=2, min_samples_leaf=30), dict(max_bins=8, max_leaves=2, min_samples_leaf=30)),
]
# v1 hyperparameter name → rabmv name
V1_NAMES = {'paso': 'split', 'suavizado': 'smoothing'}
V1_VALUES = {'optimo': 'optimal', 'voraz': 'greedy'}


def data(n=1500, seed=3):
    """Synthetic data with continuous, binary and few-valued ordinal features."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.normal(size=n), rng.uniform(0, 80, n), rng.integers(0, 2, n),
                         rng.integers(0, 6, n), rng.integers(0, 2, n), rng.normal(size=n)])
    logit = -2.2 + 0.8 * X[:, 0] + 0.03 * (X[:, 1] - 40) + 1.2 * X[:, 2] - 0.3 * X[:, 3] + 0.6 * X[:, 4]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def equal(a, b):
    """Exact equality of arrays (including NaN in the same position)."""
    a, b = np.asarray(a), np.asarray(b)
    numeric = np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=numeric)


def test_the_reference_is_the_frozen_v1():
    """The reference copy has not been modified: its fingerprint is that of v1."""
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest()[:16] == FINGERPRINT_V1


@pytest.mark.parametrize('missing', ['reject', 'marginalize'])
@pytest.mark.parametrize('weights', [False, True], ids=['no_weights', 'weights'])
@pytest.mark.parametrize('config', CONFIGURATIONS,
                         ids=lambda c: '-'.join(f'{k}={v}' for k, v in c[0].items()) or 'default')
def test_reproduces_v1_exactly(config, weights, missing):
    """Everything learned and everything predicted is identical to v1."""
    new_config, v1_config = config
    if missing == 'marginalize' and new_config.get('interactions'):
        pytest.skip('registry-aware pairs are purified on a nested grid: a declared change, not v1')
    X, y = data()
    Xt, _ = data(n=700, seed=11)
    w = np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (len(y) - y.sum()))) if weights else None
    new = RabMVClassifier(random_state=7, missing=missing, **new_config).fit(X, y, sample_weight=w)
    old = EBMPv1(random_state=7, **v1_config).fit(X, y, sample_weight=w)

    assert new.intercept_ == old.intercepto_
    assert new.term_names_ == old.terminos_
    assert new.interaction_pairs_ == old.pares_
    for a, b in zip(new.bin_edges_, old.cortes_, strict=True):
        assert equal(a, b)
    for a, b in zip(new.term_scores_, old.funciones_, strict=True):
        assert equal(a, b)
    for a, b in zip(new.term_scores_std_, old.funciones_de_, strict=True):
        assert equal(a, b)
    assert equal(new.term_importances_, old.importancias_)
    assert equal(new.best_rounds_, old.rondas_)
    assert equal(new.validation_history_, old.historial_)
    assert equal(new.predict_proba(Xt), old.predict_proba(Xt))
    assert equal(new.contributions(Xt), old.contribuciones(Xt))
    assert equal(new.predict(Xt), old.predict(Xt))
    params = new.get_params()
    assert params.pop('missing') == missing
    assert params.pop('categorical_features') is None            # hyperparameters added in phase 2d
    old_params = {V1_NAMES.get(k, k): V1_VALUES.get(v, v) if isinstance(v, str) else v
                  for k, v in old.get_params().items()}
    assert params == old_params


def test_reproduces_v1_with_text_labels():
    """With text classes, the classes and predictions also match."""
    X, y = data(n=800)
    labels = np.where(y == 1, 'completed', 'attempt')
    new = RabMVClassifier(random_state=2).fit(X, labels)
    old = EBMPv1(random_state=2).fit(X, labels)
    assert equal(new.classes_, old.classes_)
    assert equal(new.predict(X), old.predict(X))
