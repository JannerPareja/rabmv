"""Tests of the development with real data (phase 5), with synthetic data.

They need the development dependencies (interpret-core, imbalanced-learn, optuna, deap); without them they
are skipped.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('imblearn')
pytest.importorskip('interpret')
pytest.importorskip('optuna')
pytest.importorskip('deap')

from development import config as C  # noqa: E402
from development import evaluation as E  # noqa: E402
from development import models as M  # noqa: E402
from development import optimizers as O  # noqa: E402
from development import run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _core(n=400, seed=0):
    """Synthetic core with the Competencia 1 encoding and its registry-aware input."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.12).astype(int)
    X = pd.DataFrame({'edad': rng.integers(10, 80, n), 'estrato': rng.integers(0, 7, n),
                      'sexo': (rng.random(n) < 0.5 + 0.3 * y).astype(int), 'ocupacion': rng.integers(0, 2, n),
                      'escolaridad': np.where(rng.random(n) < 0.1, np.nan, rng.integers(0, 13, n)),
                      'area': rng.integers(1, 4, n), 'tip_ss': rng.choice(list('CSPENI'), n),
                      'estado_civil': rng.integers(1, 6, n)})
    Xr = pd.DataFrame({'edad': X.edad.astype(float), 'sexo': X.sexo.astype(float), 'area': X.area - 1.0,
                       'regimen': pd.Series(rng.integers(0, 6, n), dtype=float),
                       'ocupacion_activa': X.ocupacion.astype(float), 'estrato': X.estrato.astype(float),
                       'estado_civil': X.estado_civil - 1.0,
                       'escolaridad': np.where(rng.random(n) < 0.2, np.nan, rng.integers(0, 5, n)).astype(float)})
    return X, Xr, y


def test_design_has_40_combinations_and_grids_of_24():
    assert len(C.combinations()) == 16 + 16 + 8 + 8 + 8
    assert all(M.grid_size(v) == 24 for v in C.VARIANTS)
    assert C.LEVELS['development']['folds'] == [0, 1, 2, 3, 4]


def test_registry_variant_does_not_accept_oversampling():
    with pytest.raises(ValueError):
        M.pipeline('RabMV_registry', 'smote', [1, 2])


@pytest.mark.parametrize('variant,balancing', [(v, b) for v in C.VARIANTS for b in C.BALANCING_BY_VARIANT[v]])
def test_each_pipeline_fits_and_predicts(variant, balancing):
    X, Xr, y = _core()
    data = {'RabMV_registry': Xr, 'EBM_registry': Xr,
            'RabMV_common_education': X.assign(escolaridad=Xr.escolaridad)}.get(variant, X)
    # bounded max_rounds: the test checks that the pipeline works, not the fit (with noise data and SMOTE,
    # EBM can reach its default 50,000 rounds (InterpretML 0.7.8) and take more than 20 minutes)
    p = M.pipeline(variant, balancing, [1, 2, 3, 6]).set_params(
        **M.with_prefix({'learning_rate': 0.2, 'max_leaves': 3, 'max_rounds': 50}, balancing))
    p.fit(data.iloc[:300], y[:300])
    prob = p.predict_proba(data.iloc[300:])[:, 1]
    assert prob.shape == (100,) and np.all((prob >= 0) & (prob <= 1))


def test_weights_equal_balanced():
    y = np.array([0] * 9 + [1])
    w = M.balanced_weights(y)
    assert np.isclose(w[y == 1].sum(), w[y == 0].sum()) and np.isclose(w.mean(), 1.0)


@pytest.mark.parametrize('method', C.OPTIMIZERS)
def test_four_optimizers_respect_the_budget(method):
    from sklearn.model_selection import StratifiedKFold
    X, _, y = _core(300)
    cv = StratifiedKFold(3, shuffle=True, random_state=C.SEED)
    best, history, _ = O.optimize(method, 'RabMV', 'none', X, y, cv, 3)
    assert len(history) == 3 and best is not None


def test_f1_threshold_and_calibration_by_hand():
    y = np.array([0, 0, 1, 1])
    assert E.f1_threshold(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.35)
    intercept, slope = E.calibration_intercept_slope(np.r_[np.zeros(80), np.ones(20)], np.full(100, 0.2))
    assert abs(intercept) < 1e-4


def test_workdir_inside_the_repository_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        run.main(['execute', '--level', 'test', '--data', str(tmp_path / 'x.csv'), '--schema', str(tmp_path),
                  '--workdir', str(ROOT / 'development' / 'work')])


def test_prior_correction_shifts_the_logit_by_the_weight_ratio():
    """σ(logit p_w − log(n₀/n₁)): same ranking, mean probability back near the prevalence."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(3)
    X = rng.normal(size=(4000, 2))
    y = (rng.random(4000) < 1 / (1 + np.exp(-(-3 + X[:, 0])))).astype(int)
    raw = M.BalancedWeights(LogisticRegression(), prior_correction=False).fit(X, y)
    cor = M.BalancedWeights(LogisticRegression()).fit(X, y)
    n1 = y.sum()
    assert np.isclose(cor.offset_, np.log((len(y) - n1) / n1))
    pr, pc = raw.predict_proba(X)[:, 1], cor.predict_proba(X)[:, 1]
    assert np.allclose(np.log(pc / (1 - pc)), np.log(pr / (1 - pr)) - cor.offset_)
    assert np.array_equal(np.argsort(pr, kind='stable'), np.argsort(pc, kind='stable'))
    assert abs(pc.mean() - y.mean()) < 0.01 < abs(pr.mean() - y.mean())


def test_rabmv_space_has_the_interactions_of_ebm():
    ebm = {p.name: p for p in M.SPACE_EBM}['interactions']
    rab = {p.name: p for p in M.SPACE_RABMV}['interactions']
    assert (rab.low, rab.high, rab.grid) == (ebm.low, ebm.high, ebm.grid)


def test_checkpoint_of_another_design_version_is_rejected(tmp_path):
    import pickle

    from development import execution as X
    path = X.task_path(str(tmp_path), 'test', 'EBM', 'none', 'grid', 0)
    Path(path).parent.mkdir(parents=True)
    Path(path).write_bytes(pickle.dumps({'variant': 'EBM'}))
    with pytest.raises(RuntimeError, match='design version'):
        X.run_task(dict(variant='EBM', balancing='none', optimizer='grid', fold=0, level='test',
                        workdir=str(tmp_path), data='', schema=''))


@pytest.mark.parametrize('variant,balancing', [('EBM', 'smote'), ('RabMV', 'adasyn'), ('RabMV', 'weights'),
                                               ('RabMV_registry', 'none'), ('RabMV_common_education', 'weights'),
                                               ('EBM_registry', 'weights')])
def test_saved_model_predicts_the_same_and_holds_no_rows(variant, balancing):
    """The saved model (pipeline without the sampler) gives the same probabilities, has a model card and, with
    oversampling, diagnostics of the synthetic cases; nothing in it stores the training rows."""
    import pickle
    X, Xr, y = _core()
    data = {'RabMV_registry': Xr, 'EBM_registry': Xr,
            'RabMV_common_education': X.assign(escolaridad=Xr.escolaridad)}.get(variant, X)
    p = M.pipeline(variant, balancing, [1, 2, 3, 6]).set_params(
        **M.with_prefix({'learning_rate': 0.2, 'max_leaves': 3, 'max_rounds': 50}, balancing))
    p.fit(data.iloc[:300], y[:300])
    model = M.prediction_model(p)
    assert np.allclose(model.predict_proba(data.iloc[300:]), p.predict_proba(data.iloc[300:]))
    blob = pickle.dumps(model)
    train = np.ascontiguousarray(p.steps[0][1].transform(data.iloc[:300]) if variant not in C.REGISTRY_INPUT
                                 else data.iloc[:300], dtype=float)
    assert train.tobytes()[:400] not in blob                  # no block of training rows inside the model
    card = M.model_card(model)
    assert 'intercept' in card and 'term_scores' in card
    if balancing == 'weights':
        assert card['offset'] != 0
    diag = M.oversampling_diagnostics(p, data.iloc[:300], y[:300])
    if balancing in ('smote', 'adasyn'):
        assert diag['n_synthetic'] > 0 and diag['features']
        assert any(f['synthetic_off_support_share'] > 0 for f in diag['features'])
    else:
        assert diag is None


def test_report_from_checkpoints_without_training(tmp_path):
    """The report (equations, term tables, figures) is built only from what a task saves."""
    pytest.importorskip('matplotlib')
    from development import report
    X, Xr, y = _core()
    res = []
    for variant, balancing, data in (('RabMV', 'smote', X), ('EBM', 'weights', X), ('RabMV_registry', 'none', Xr)):
        p = M.pipeline(variant, balancing, [1, 2, 3, 6]).set_params(
            **M.with_prefix({'learning_rate': 0.2, 'max_leaves': 3, 'max_rounds': 50}, balancing))
        p.fit(data.iloc[:300], y[:300])
        model = M.prediction_model(p)
        res.append(dict(variant=variant, balancing=balancing, optimizer='grid', fold=0, best_params={},
                        model=model, model_card=M.model_card(model), y_test=y[300:],
                        p_test=model.predict_proba(data.iloc[300:])[:, 1],
                        history=[{'score': 0.1}, {'score': 0.2}],
                        oversampling=M.oversampling_diagnostics(p, data.iloc[:300], y[:300])))
    folds = pd.DataFrame([dict(variant=r['variant'], balancing=r['balancing'], optimizer='grid', auc_pr=0.3)
                          for r in res])
    assert report.build(res, folds, tmp_path) == 3
    eq = (tmp_path / 'models' / 'RabMV__smote__grid' / 'fold_00_equation.txt').read_text(encoding='utf-8')
    assert 'logit p(x)' in eq and 'β₀' in eq
    t = pd.read_csv(tmp_path / 'models' / 'EBM__weights__grid' / 'fold_00_terms.csv')
    assert '(prior correction of the weights)' in set(t.term)
    for name in ('pr_curves.png', 'calibration.png', 'convergence.png', 'auc_pr_by_combination.png',
                 'oversampling.csv'):
        assert (tmp_path / name).exists()


def test_english_labels_of_variables_categories_and_resolution_groups(tmp_path):
    """labels_en.toml gives the English name of a variable, of a one-hot category and of each group of the
    common resolution of a registry concept."""
    from development import report
    from tests.test_schema import CATALOG, MAPPING_105, MAPPING_356
    (tmp_path / 'mappings').mkdir()
    (tmp_path / 'catalog.toml').write_text(CATALOG, encoding='utf-8')
    (tmp_path / 'mappings' / '356.toml').write_text(MAPPING_356, encoding='utf-8')
    (tmp_path / 'mappings' / '105.toml').write_text(MAPPING_105, encoding='utf-8')
    (tmp_path / 'labels_en.toml').write_text(
        '[variables.education]\nname = "Education level"\n'
        '[variables.tip_ss]\nname = "Health insurance scheme"\ncategories = { C = "Contributory" }\n'
        '[variables.sexo]\nname = "Sex"\ncategories = { 0 = "Female", 1 = "Male" }\n'
        '[resolutions.education]\nfive_levels = ["None", "Primary", "Secondary", "Technical", "University"]\n',
        encoding='utf-8')
    lab = report.Labels(tmp_path)
    assert lab.name('education') == 'Education level'
    assert lab.name('cat__tip_ss_C') == 'Health insurance scheme: Contributory'
    assert lab.name('num__sexo & education') == 'Sex × Education level'
    assert lab.bin('education', 1.5, 2.5) == 'Secondary'
    assert lab.bin('num__sexo', 0.5, np.inf) == 'Male'
    assert lab.bin('cat__tip_ss_C', 0.5, np.inf) == 'yes'
    assert report.Labels().name('num__edad') == 'num__edad'
