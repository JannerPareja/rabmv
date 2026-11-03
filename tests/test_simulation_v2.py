"""Tests of simulation v2 (generator, audits, post-hoc marginalization of EBM, calibration, task list)."""
import numpy as np
import pandas as pd
import pytest

from rabmv import evaluate_eligibility, harmonize
from simulation_v2 import audits
from simulation_v2.generator import CONDITIONS, condition_params, record, schema, target_params, training_population
from simulation_v2.methods import calibration
from simulation_v2.run import tasks


def _registry(condition, seed=101):
    p = condition_params(condition)
    pop = training_population(seed, 180, p)
    return p, pop, record(pop, p, seed)


def test_form_is_the_class_when_pi_is_one():
    p, pop, d = _registry('E_pi100_education')
    assert ((d['form'] == 'B') == (pop['y'] == 1)).all()


def test_structural_gap_only_in_form_a_2016():
    p, pop, d = _registry('E_pi100_occupation')
    gap = d['occupation'].isna()
    year = pop['year']
    assert gap.any()
    assert (gap == ((d['form'] == 'A') & (year == 2016))).all()


def test_null_has_one_form_and_no_gap():
    p, pop, d = _registry('no_difference')
    assert (d['form'] == 'A').all()
    assert not d['education'].isna().any() and not d['occupation'].isna().any()


def test_target_t3_moves_the_gap_to_form_b():
    p = condition_params('E_pi100_education')
    tp = target_params(p, 'T3')
    pop = training_population(5, 180, p)
    d = record(pop, tp, 5)
    assert (d['education'].isna() == (d['form'] == 'B')).all()


def test_declared_gap_is_marginalized_and_undeclared_gap_is_detected():
    p, pop, d = _registry('E_pi100_education')
    y = pop['y'].to_numpy()
    s = schema(p)
    h = harmonize(d, s)
    m = evaluate_eligibility(h, pd.Series(y), s.catalog).matrix(h)
    assert m.X['education'].isna().sum() == ((d['form'] == 'A') & (pop['year'] == 2016)).sum()
    assert 'education' in audits.rabmv_checks(d, y, schema(p, 'all_exact'))
    assert 'education' not in audits.rabmv_checks(d, y, s)


def test_shukla_test_a_when_form_is_the_class():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(int)
    r = audits.test_a(y, np.where(y == 1, 'B', 'A'), rng)
    assert r['V'] == pytest.approx(1.0) and r['flag'] == 'FAIL' and r['p'] == pytest.approx(1 / 501)


def test_shukla_test_c_and_overall_rule():
    year = np.repeat(np.arange(2016, 2023), 50)
    y = np.tile(np.r_[np.ones(5), np.zeros(45)], 7).astype(int)
    assert audits.test_c(y, year)['flag'] == 'PASS'
    y2 = y.copy()
    y2[year >= 2019] = 0
    assert audits.test_c(y2, year)['flag'] == 'FAIL'
    assert audits.overall(['FAIL', 'FAIL', 'PASS']) == 'FAIL'
    assert audits.overall(['CAUTION', 'CAUTION', 'PASS']) == 'CAUTION'
    assert audits.overall(['FAIL', 'PASS']) == 'CAUTION'
    assert audits.overall(['PASS', 'CAUTION']) == 'PASS'


def test_posthoc_marginalization_of_ebm():
    pytest.importorskip('interpret')
    from simulation_v2.methods import Engine, marginalize_ebm
    rng = np.random.default_rng(1)
    n = 1500
    X = pd.DataFrame({'a': rng.normal(size=n), 'b': rng.integers(0, 4, n).astype(float)})
    y = (rng.random(n) < 0.2).astype(int)
    X.loc[(y == 0) & (rng.random(n) < 0.3), 'b'] = np.nan
    e = Engine('ebm', ['a', 'b'], {'a': 'numeric', 'b': 'nominal'},
               dict(learning_rate=0.05, min_samples_leaf=2, max_bins=32, smoothing_rounds=0)).fit(X, y)
    em = marginalize_ebm(e)
    s, w = e.model.term_scores_[1], e.model.bin_weights_[1]
    assert em.model.term_scores_[1][0] == pytest.approx(np.sum(w[1:-1] * s[1:-1]) / np.sum(w[1:-1]))
    assert e.model.term_scores_[1][0] != em.model.term_scores_[1][0]          # the original is untouched
    measured = X['b'].notna().to_numpy()
    assert np.allclose(e.predict_proba(X)[measured], em.predict_proba(X)[measured])


def test_calibration_of_a_calibrated_model():
    rng = np.random.default_rng(2)
    lp = rng.normal(-2.5, 1.2, 200_000)
    p = 1 / (1 + np.exp(-lp))
    y = (rng.random(len(p)) < p).astype(int)
    a, b = calibration(y, p)
    assert abs(a) < 0.03 and abs(b - 1) < 0.03


def test_task_list():
    assert len(CONDITIONS) == 12
    assert len(tasks('test')) == 3 * 12 + 3 * 2
    assert len(tasks('evaluation')) == 100 * 12 + 100 * 2


def test_chi2_matches_scipy():
    from scipy.stats import chi2_contingency
    rng = np.random.default_rng(3)
    y = (rng.random(800) < 0.2).astype(int)
    e = rng.integers(0, 7, 800)
    chi2, v = audits._chi2(y, e)
    ref = chi2_contingency(pd.crosstab(y, e).to_numpy(), correction=False)[0]
    assert chi2 == pytest.approx(ref)
    assert v == pytest.approx(np.sqrt(ref / (800 * 1)))


def test_memoized_normalization_gives_identical_harmonization():
    import rabmv.availability as av
    import simulation_v2
    from rabmv.schema.normalization import normalize_value
    p, pop, d = _registry('E_pi85_occupation')
    s = schema(p)
    fast = harmonize(d, s)
    av.normalize_value = normalize_value
    try:
        slow = harmonize(d, s)
    finally:
        av.normalize_value = simulation_v2._memo_normalize
    pd.testing.assert_frame_equal(fast.values, slow.values)
    pd.testing.assert_frame_equal(fast.status, slow.status)


def test_calibration_intercept_of_a_very_badly_calibrated_model():
    rng = np.random.default_rng(4)
    y = (rng.random(5000) < 0.08).astype(int)
    p = np.where(y == 1, 1e-12, 1 - 1e-12)          # perfectly wrong and extreme
    a, b = calibration(y, p)
    q = 1 / (1 + np.exp(-(a + np.log(p / (1 - p)))))
    assert np.isfinite(a) and abs(q.sum() - y.sum()) < 1e-6 * len(y)
