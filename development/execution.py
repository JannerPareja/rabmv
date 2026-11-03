"""One task = one combination (variant × balancing × optimizer) in one outer fold.

Reproduces c1/ejecucion.py: X_fit / X_thresh (stratified 20 % of the outer training set) / test,
optimization with inner CV on X_fit, refit with the best configuration, F1 threshold and recalibration on
X_thresh, metrics on the test fold (``evaluation.py``, translation of the Competencia 1 module).

Each task is saved in its own file (checkpoint) in the WORK folder, which must be outside any repository:
it contains per-case probabilities and indices and the fitted final model (``model``: preprocessing and
classifier; the oversampling step is left out because its fitted neighbour search keeps training rows, and it
does not act at prediction time). The fitted model holds no data rows: cuts, shape functions, medians, scales
and categories. Also saved: the model description in plain arrays (``model_card``: intercept, terms, cuts,
scores, standard deviations, importances; the equation of the model), the probabilities of X_fit, X_thresh and
the test fold (original, Platt and isotonic), the indices of the three partitions and, with SMOTE / ADASYN, the
diagnostics of the synthetic cases. Everything can be analysed later without training again.
"""
import os
import pickle
import platform
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config as C
from . import data as D
from . import evaluation as E
from . import models as M
from . import optimizers as O
from . import progress as P

_CACHE = {}


def prepare(data_path, schema_folder, check_fingerprint=True):
    """Load the data and the splits once per process."""
    key = (str(data_path), str(schema_folder))
    if key not in _CACHE:
        d = D.load(data_path, schema_folder, check_fingerprint=check_fingerprint)
        _CACHE[key] = (d, D.splits(d.y))
    return _CACHE[key]


def task_path(workdir, level, variant, balancing, optimizer, fold):
    """``<workdir>/<level>/tasks/<variant>__<balancing>__<optimizer>/fold_XX.pkl``."""
    return os.path.join(workdir, level, 'tasks', f'{variant}__{balancing}__{optimizer}', f'fold_{fold:02d}.pkl')


def _atomic_save(obj, path):
    """Save with pickle through a temporary file and ``os.replace`` (never left half-written)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def fold_indices(y, parts, fold, year=None):
    """Fit, threshold (X_thresh) and test indices of an outer fold (as in Competencia 1).

    Folds 0–24 are those of the 5×5. Phase 6 adds two special folds: ``TEMPORAL_FOLD`` (training set = cases
    notified up to ``TEMPORAL_SPLIT_YEAR``, test = the later ones) and ``FINAL_FOLD`` (every case is training;
    no test fold: the final model is validated with out-of-fold probabilities).
    """
    if fold == C.TEMPORAL_FOLD:
        year = np.asarray(year)
        tr, te = np.flatnonzero(year <= C.TEMPORAL_SPLIT_YEAR), np.flatnonzero(year > C.TEMPORAL_SPLIT_YEAR)
    elif fold == C.FINAL_FOLD:
        return np.arange(len(y)), np.array([], dtype=int), np.array([], dtype=int)
    else:
        tr, te = parts[fold]
    pos_fit, pos_thr = train_test_split(np.arange(len(tr)), test_size=C.THRESHOLD_FRACTION,
                                        stratify=y[tr], random_state=C.SEED)
    return tr[pos_fit], tr[pos_thr], te


def run_task(task):
    """Run (or recover) a task.

    Parameters
    ----------
    task : dict
        variant, balancing, optimizer, fold, level, workdir, data, schema and check_fingerprint.

    Returns
    -------
    dict
        The same content saved in the task file.
    """
    v, b, o, fold, level = task['variant'], task['balancing'], task['optimizer'], task['fold'], task['level']
    path = task_path(task['workdir'], level, v, b, o, fold)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            saved = pickle.load(f)
        if saved.get('design_version') != C.DESIGN_VERSION:
            raise RuntimeError(f'{path} was saved with design version {saved.get("design_version", 1)}, not '
                               f'{C.DESIGN_VERSION}: use a new work folder')
        return saved

    d, parts = prepare(task['data'], task['schema'], task.get('check_fingerprint', True))
    cfg = C.LEVELS[level]
    X = d.input(v)
    i_fit, i_thr, te = fold_indices(d.y, parts, fold, d.year)
    inner_cv = StratifiedKFold(n_splits=cfg['inner'], shuffle=True, random_state=C.SEED)
    res = {'variant': v, 'balancing': b, 'optimizer': o, 'fold': fold, 'level': level, 'seed': C.SEED,
           'design_version': C.DESIGN_VERSION,
           'data_fingerprint': d.fingerprint, 'budget': cfg['budget'], 'n_fit': len(i_fit),
           'n_thresh': len(i_thr), 'n_test': len(te), 'idx_test': te, 'y_test': d.y[te],
           'machine': platform.node(), 'error': None}

    extra = C.FINAL_OOF if fold == C.FINAL_FOLD else 0
    P.start(os.path.join(task['workdir'], level, 'progress', f'{v}__{b}__{o}__fold_{fold:02d}.json'),
            f'{v}·{b}·{o}·f{fold}', cfg['budget'] * cfg['inner'] + 1 + extra)
    try:
        if fold == C.FINAL_FOLD:
            return _run_final(task, res, d, X, i_fit, inner_cv, cfg, path)
        return _run_task(task, res, d, X, i_fit, i_thr, te, inner_cv, cfg, path)
    finally:
        P.finish()


def _run_task(task, res, d, X, i_fit, i_thr, te, inner_cv, cfg, path):
    """Optimization, refit, evaluation and saving of one task (``run_task`` handles progress)."""
    v, b, o = task['variant'], task['balancing'], task['optimizer']
    t0 = time.perf_counter()
    best, history, info = O.optimize(o, v, b, X.iloc[i_fit], d.y[i_fit], inner_cv, cfg['budget'],
                                     categorical=d.categorical)
    valid = [h['score'] for h in history if np.isfinite(h['score'])]
    res.update(seconds_optimization=time.perf_counter() - t0, history=history, optimizer_info=info,
               best_params=best, best_inner_score=max(valid, default=np.nan),
               evaluations=len(history), failed_evaluations=sum(h['error'] is not None for h in history))
    if best is None:
        res['error'] = 'every evaluation failed: ' + str(history[0]['error'] if history else '')
        _atomic_save(res, path)
        return res

    final = M.pipeline(v, b, d.categorical).set_params(**M.with_prefix(best, b))
    t1 = time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', ConvergenceWarning)
        final.fit(X.iloc[i_fit], d.y[i_fit])
    res['seconds_refit'] = time.perf_counter() - t1
    P.tick()
    res['refit_convergence_warnings'] = sum(issubclass(a.category, ConvergenceWarning) for a in w)
    p_test = final.predict_proba(X.iloc[te])[:, 1]
    p_thr = final.predict_proba(X.iloc[i_thr])[:, 1]
    p_fit = final.predict_proba(X.iloc[i_fit])[:, 1]
    threshold, rows = E.evaluate_fold(d.y[i_thr], p_thr, d.y[te], p_test, d.sex[i_thr], d.sex[te], by_sex=False)
    res['sex_test'] = d.sex[te]                       # for the equity analysis of phase 6 (aggregates only)
    recal = E.recalibrators(d.y[i_thr], p_thr)
    model = M.prediction_model(final)
    res.update(p_test=p_test, threshold=threshold, metrics=rows,
               idx_fit=i_fit, idx_thresh=i_thr, y_fit=d.y[i_fit], y_thresh=d.y[i_thr],
               p_fit=p_fit, p_thresh=p_thr,
               p_test_platt=recal['platt'](p_test), p_test_isotonic=recal['isotonic'](p_test),
               p_thresh_platt=recal['platt'](p_thr), p_thresh_isotonic=recal['isotonic'](p_thr),
               model=model, model_card=M.model_card(model),
               oversampling=M.oversampling_diagnostics(final, X.iloc[i_fit], d.y[i_fit]))
    _atomic_save(res, path)
    return res


def _run_final(task, res, d, X, i_all, inner_cv, cfg, path):
    """Final model of a variant (phase 6): search with inner CV and refit on every case of the window.

    There is no test fold. The operating threshold and an internal validation come from out-of-fold
    probabilities: the chosen configuration is refitted ``FINAL_OOF`` times (stratified folds, study seed) and each
    case is predicted by the fit that did not see it. Those probabilities are saved as ``p_test`` (with
    ``idx_test`` = every case) so that the report draws them like any other fold; the metrics under
    ``metrics['original']`` are out-of-fold (slightly optimistic: the configuration was chosen with every case,
    declared in the protocol). ``p_apparent`` are the probabilities of the final model on its own training cases.
    """
    v, b, o = task['variant'], task['balancing'], task['optimizer']
    t0 = time.perf_counter()
    best, history, info = O.optimize(o, v, b, X.iloc[i_all], d.y[i_all], inner_cv, cfg['budget'],
                                     categorical=d.categorical)
    valid = [h['score'] for h in history if np.isfinite(h['score'])]
    res.update(seconds_optimization=time.perf_counter() - t0, history=history, optimizer_info=info,
               best_params=best, best_inner_score=max(valid, default=np.nan),
               evaluations=len(history), failed_evaluations=sum(h['error'] is not None for h in history),
               n_fit=len(i_all), n_thresh=0, n_test=len(i_all), idx_test=i_all, y_test=d.y[i_all],
               sex_test=d.sex[i_all])
    if best is None:
        res['error'] = 'every evaluation failed: ' + str(history[0]['error'] if history else '')
        _atomic_save(res, path)
        return res
    final = M.pipeline(v, b, d.categorical).set_params(**M.with_prefix(best, b))
    t1 = time.perf_counter()
    final.fit(X.iloc[i_all], d.y[i_all])
    res['seconds_refit'] = time.perf_counter() - t1
    P.tick()
    p_oof = np.full(len(i_all), np.nan)
    oof = StratifiedKFold(n_splits=C.FINAL_OOF, shuffle=True, random_state=C.SEED)
    for k_fit, k_out in oof.split(np.zeros(len(i_all)), d.y[i_all]):
        m = M.pipeline(v, b, d.categorical).set_params(**M.with_prefix(best, b))
        m.fit(X.iloc[i_all[k_fit]], d.y[i_all[k_fit]])
        p_oof[k_out] = m.predict_proba(X.iloc[i_all[k_out]])[:, 1]
        P.tick()
    threshold = E.f1_threshold(d.y[i_all], p_oof)
    model = M.prediction_model(final)
    p_app = final.predict_proba(X.iloc[i_all])[:, 1]
    res.update(p_test=p_oof, threshold=threshold,
               metrics={'original': E.metrics(d.y[i_all], p_oof, threshold, d.sex[i_all]),
                        'apparent': E.metrics(d.y[i_all], p_app, threshold, d.sex[i_all])},
               idx_fit=i_all, y_fit=d.y[i_all], p_fit=p_app, p_apparent=p_app,
               idx_thresh=np.array([], dtype=int), y_thresh=np.array([], dtype=int), p_thresh=np.array([]),
               model=model, model_card=M.model_card(model), oversampling=None)
    _atomic_save(res, path)
    return res


def tasks(level, workdir, data_path, schema_folder, check_fingerprint=True, variants=None, balancing=None):
    """Every task of the level, in a fixed order (combination × fold), optionally only some variants or
    balancing methods (to run the level in blocks)."""
    out = []
    for v, b, o in C.combinations(level):
        if (variants and v not in variants) or (balancing and b not in balancing):
            continue
        for fold in C.LEVELS[level]['folds']:
            out.append(dict(variant=v, balancing=b, optimizer=o, fold=fold, level=level, workdir=str(workdir),
                            data=str(data_path), schema=str(schema_folder), check_fingerprint=check_fingerprint))
    return out
