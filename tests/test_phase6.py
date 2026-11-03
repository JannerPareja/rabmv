"""Tests of phase 6 (confirmation): design, folds, statistics of the protocol, ablation and report helpers.

Synthetic data only. They need the development dependencies; without them they are skipped.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

pytest.importorskip('imblearn')
pytest.importorskip('interpret')
pytest.importorskip('optuna')

from development import analysis as A  # noqa: E402
from development import config as C  # noqa: E402
from development import execution as X  # noqa: E402
from development import models as M  # noqa: E402
from development import report as R  # noqa: E402


def test_phase6_design_is_the_one_of_the_protocol():
    assert C.LEVELS['confirmation']['folds'] == list(range(5, 25))
    assert C.LEVELS['confirmation']['budget'] == 24 and C.LEVELS['confirmation']['inner'] == 5
    for level in C.PHASE6_LEVELS:
        assert C.combinations(level) == [(v, 'none', 'bayes') for v in C.VARIANTS]
    assert len(C.combinations()) == 56                         # phase 5 unchanged
    assert C.NI_MARGIN == 0.01 and C.ROPE == 0.01


def test_protocol_file_is_the_registered_one():
    path = Path(A.__file__).resolve().parent / 'PROTOCOL_PHASE6.md'
    h = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    assert h == C.PROTOCOL_SHA256
    assert A.protocol_fingerprint() == (h, True)


def test_temporal_and_final_folds():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.2).astype(int)
    year = rng.integers(2016, 2023, 300)
    fit, thr, te = X.fold_indices(y, None, C.TEMPORAL_FOLD, year)
    assert set(year[np.concatenate([fit, thr])]) <= set(range(2016, 2021))
    assert set(year[te]) <= {2021, 2022} and len(fit) + len(thr) + len(te) == 300
    fit, thr, te = X.fold_indices(y, None, C.FINAL_FOLD, year)
    assert len(fit) == 300 and len(thr) == 0 and len(te) == 0


def test_folds_5_to_24_are_repetitions_2_to_5_and_disjoint_within_each():
    from development import data as D
    y = np.r_[np.zeros(200, int), np.ones(40, int)]
    parts = D.splits(y)
    assert len(parts) == 25
    for rep in range(5):
        tests = np.concatenate([parts[f][1] for f in range(rep * 5, rep * 5 + 5)])
        assert sorted(tests) == list(range(len(y)))


def test_corrected_t_by_hand():
    d = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    r = A.corrected_t(d, n_train=1880, n_test=470, margin=-0.01)
    se = np.sqrt((1 / 5 + 470 / 1880) * d.var(ddof=1))
    assert np.isclose(r['se_corrected'], se)
    assert np.isclose(r['t'], (d.mean() + 0.01) / se)
    assert np.isclose(r['lower_95_one_sided'], d.mean() - stats.t.ppf(0.95, 4) * se)
    assert np.isclose(r['p_greater'], stats.t.sf((d.mean() + 0.01) / se, 4))


def test_bayesian_correlated_probabilities_sum_to_one_and_use_rho():
    d = np.array([0.02, 0.01, 0.03, 0.015, 0.025, 0.02])
    b = A.bayes_correlated(d, 1880, 470, rope=0.01)
    assert np.isclose(b['p_left'] + b['p_rope'] + b['p_right'], 1)
    rho = 470 / 2350
    assert np.isclose(b['scale'], np.sqrt((1 / 6 + rho / (1 - rho)) * d.var(ddof=1)))
    assert b['p_right'] > b['p_left']


def test_holm_kendall_cliff():
    assert np.allclose(A.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.isnan(A.holm([np.nan, 0.02])[0])
    ranks = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    assert np.isclose(A.kendalls_w(ranks), 1.0)
    assert A.cliffs_delta([2, 3], [0, 1]) == 1.0 and A.cliffs_delta([0, 1], [0, 1]) == 0.0
    assert A.suppress(3) == '<5' and A.suppress(0) == '0' and A.suppress(12) == '12'


def test_primary_decision_follows_the_protocol():
    rows = []
    for f in range(25):
        for v, auc in (('RabMV', 0.50 + 0.001 * (f % 3)), ('EBM', 0.50)):
            rows.append(dict(variant=v, fold=f, repetition=f // 5 + 1, probabilities='original', n_train=1880,
                             n_test=470, auc_pr=auc, calib_intercept=0.05, calib_slope=1.0))
    fm = pd.DataFrame(rows)
    sizes = fm.groupby('fold')[['n_train', 'n_test']].mean()
    p = A.primary_decision(fm, sizes)
    assert p['folds'] == 20 and p['noninferior_auc_pr'] and p['weak_calibration'] and p['H1']
    fm.loc[fm.variant == 'RabMV', 'calib_slope'] = 0.6                  # badly calibrated → H1 fails
    p = A.primary_decision(fm, sizes)
    assert not p['H1'] and p['H2'].startswith('not tested')


def _ebm_with_missing():
    from interpret.glassbox import ExplainableBoostingClassifier
    rng = np.random.default_rng(1)
    n = 400
    y = (rng.random(n) < 0.3).astype(int)
    x1 = rng.normal(size=n) + y
    x2 = np.where(y == 0, np.where(rng.random(n) < 0.4, np.nan, rng.normal(size=n)), rng.normal(size=n))
    Xd = pd.DataFrame({'a': x1, 'b': x2})
    ebm = ExplainableBoostingClassifier(outer_bags=2, interactions=1, random_state=0, max_rounds=200, n_jobs=1)
    from sklearn.pipeline import Pipeline
    return Pipeline([('clf', ebm.fit(Xd, y))]), Xd, y


def test_ablation_zeroes_only_the_missing_scores():
    model, Xd, y = _ebm_with_missing()
    before = [np.array(s, float).copy() for s in model.steps[-1][1].term_scores_]
    ablated, changed = A.ablate_missing(model)
    after = ablated.steps[-1][1].term_scores_
    assert changed >= 1
    for s0, s1 in zip(before, after):
        assert np.all(np.take(s1, 0, axis=0) == 0)
        if s0.ndim == 2:
            assert np.all(s1[:, 0] == 0)
            assert np.array_equal(s0[1:, 1:], s1[1:, 1:])
        else:
            assert np.array_equal(s0[1:], s1[1:])
    # the original model is not modified and complete cases are scored the same in the main terms
    assert np.array_equal(model.steps[-1][1].term_scores_[0], before[0])
    complete = Xd.notna().all(axis=1).to_numpy()
    p0, p1 = model.predict_proba(Xd)[:, 1], ablated.predict_proba(Xd)[:, 1]
    assert np.allclose(p0[complete], p1[complete])
    assert not np.allclose(p0[~complete], p1[~complete])


def test_numeric_edges_draw_years_not_bin_numbers():
    sub = pd.DataFrame({'lower': [-np.inf, 13, 14, 15] + list(range(16, 30)),
                        'upper': [13, 14, 15, 16] + list(range(17, 30)) + [np.inf],
                        'bin_en': [''] * 18, 'score': np.zeros(18), 'std': np.zeros(18)})
    e = R._numeric_edges(sub)
    assert e[1] == 13 and e[-2] == 29 and e[0] == 12 and e[-1] == 30
    cats = pd.DataFrame({'lower': [-np.inf, 0.5], 'upper': [0.5, np.inf], 'bin_en': ['Female', 'Male'],
                         'score': [0., 1.], 'std': [0., 0.]})
    assert R._numeric_edges(cats) is None


def test_net_benefit_by_hand():
    y = np.array([1, 0, 0, 1, 0])
    p = np.array([0.9, 0.6, 0.1, 0.4, 0.3])
    nb = A.net_benefit(y, p, [0.5])
    assert np.isclose(nb[0, 0], 1 / 5 - 1 / 5 * 1.0)             # TP = 1 (0.9), FP = 1 (0.6)
    assert np.isclose(nb[0, 1], 0.4 - 0.6 * 1.0)


def test_final_model_task_saves_out_of_fold_probabilities(tmp_path):
    """A final task on synthetic data: refit on every case, OOF probabilities, no test fold."""
    from development import data as D
    rng = np.random.default_rng(3)
    n = 260
    y = (rng.random(n) < 0.2).astype(int)
    Xd = pd.DataFrame({'edad': rng.integers(10, 80, n), 'estrato': rng.integers(0, 7, n), 'sexo': rng.integers(0, 2, n),
                       'ocupacion': rng.integers(0, 2, n), 'escolaridad': rng.integers(0, 13, n).astype(float),
                       'area': rng.integers(1, 4, n), 'tip_ss': rng.choice(list('CSPENI'), n),
                       'estado_civil': rng.integers(1, 6, n)})
    d = D.Data(X=Xd, X_common_education=Xd, X_registry=Xd, categorical=[], y=y, sex=Xd.sexo.to_numpy(),
               fingerprint='synthetic', year=np.full(n, 2018))
    key = ('synthetic', 'synthetic')
    X._CACHE[key] = (d, D.splits(y))
    saved = dict(C.LEVELS['final'])
    C.LEVELS['final'].update(inner=2, budget=2)
    try:
        r = X.run_task(dict(variant='RabMV', balancing='none', optimizer='bayes', fold=C.FINAL_FOLD, level='final',
                            workdir=str(tmp_path), data='synthetic', schema='synthetic', check_fingerprint=False))
    finally:
        C.LEVELS['final'].update(saved)
        X._CACHE.pop(key, None)
    assert r['error'] is None and r['n_test'] == n and r['n_thresh'] == 0
    assert np.isfinite(r['p_test']).all() and len(r['p_test']) == n
    assert {'original', 'apparent'} <= set(r['metrics'])
    assert M.model_card(r['model'])['classifier'] == 'RabMVClassifier'


def test_missing_to_mean_follows_rabmv_rule_and_exclusive_categories():
    from development import exploratory as XP
    model, Xd, y = _ebm_with_missing()
    m = XP.missing_to_mean(model, Xd)
    ebm, orig = m.steps[-1][1], model.steps[-1][1]
    contrib = orig.eval_terms(Xd)
    for t, feats in enumerate(orig.term_features_):
        if len(feats) == 1:
            measured = Xd.iloc[:, feats[0]].notna().to_numpy()
            assert np.isclose(ebm.term_scores_[t][0], contrib[measured, t].mean())
            assert np.array_equal(ebm.term_scores_[t][1:], orig.term_scores_[t][1:])
    # exclusive categories: a category present only in class 0 is found and blanked
    from development import data as D
    n = 300
    rng = np.random.default_rng(5)
    yy = (rng.random(n) < 0.2).astype(int)
    reg = np.where(yy == 1, rng.integers(0, 2, n), rng.integers(0, 3, n)).astype(float)   # code 2 only in class 0
    Xr = pd.DataFrame({'regimen': reg, 'edad': rng.integers(10, 80, n).astype(float)})
    d = D.Data(X=Xr, X_common_education=Xr, X_registry=Xr, categorical=[0], y=yy, sex=np.zeros(n, int),
               fingerprint='s', year=np.full(n, 2018))
    table, mask = XP.exclusive_categories(d, None)
    assert mask == {'regimen': [2.0]} and table.form_exclusive.sum() == 1
    Xb = XP.blank_exclusive(Xr, mask)
    assert Xb.regimen.isna().sum() == (reg == 2).sum() and Xr.regimen.notna().all()
