"""Pre-specified analysis of phase 6 (``PROTOCOL_PHASE6.md``), from the saved checkpoints, without training.

Inputs: the work folder with the tasks of phase 5 (level ``development``; only no balancing · Bayesian search is
used) and of phase 6 (levels ``confirmation``, ``temporal`` and ``final``), plus the data and the schema (to
score the ablated models and to count). Outputs, only aggregates, in ``development/results/phase6/``:

* ``decisions.json`` and ``RESULTS_PHASE6.md`` — the decisions of the protocol (H1, H2, secondary family,
  ablation) with every number they rest on;
* ``tables/`` — metrics per fold, comparisons (folds 5–24 and 0–24; AUC-PR, AUC-ROC, Brier, F1), calibration,
  equity by sex, decision curve, stability of the interpretation, interactions, hyperparameters, cost, temporal
  validation, final models, ablation and supporting counts (1–4 shown as «<5»);
* ``figures/`` — forest plot of the comparisons, Bayesian posteriors, AUC-PR per fold, ablation, pooled
  precision-recall and calibration curves, decision curve, equity, importances and temporal validation;
* ``manifest.json`` — fingerprints of the protocol, the data and every output file.

Every step prints a line with the elapsed time (visible progress).
"""
import copy
import datetime as dt
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

from . import config as C
from . import data as D
from . import evaluation as E

COMPARISON_METRICS = ['auc_pr', 'auc_roc', 'brier', 'f1']
PRIMARY = ('RabMV', 'EBM')
SECONDARY = [('S1', 'RabMV_registry', 'EBM_registry'), ('S2', 'EBM_registry', 'EBM'),
             ('S3', 'RabMV_registry', 'RabMV'), ('S4', 'RabMV_common_education', 'RabMV'),
             ('S5', 'RabMV_registry', 'RabMV_common_education')]
ABLATED = 'EBM_registry (missing scores = 0)'
CONFIRMATION_FOLDS = list(range(5, 25))
ALL_FOLDS = list(range(25))
MIN_COUNT = 5                 # counts below this (1–4) are shown as «<5»
MIN_BIN = 5                   # minimum cases per calibration bin drawn


class Progress:
    """One printed line per step with the elapsed time."""

    def __init__(self, total):
        self.total, self.k, self.t0 = total, 0, time.perf_counter()

    def step(self, text):
        self.k += 1
        print(f'[analysis {self.k:2d}/{self.total}] {(time.perf_counter() - self.t0) / 60:5.1f} min  {text}', flush=True)

    def note(self, text):
        print(f'                 {(time.perf_counter() - self.t0) / 60:5.1f} min  {text}', flush=True)


# ----------------------------------------------------------------------------- statistics
def corrected_t(d, n_train, n_test, margin=0.0):
    """Corrected resampled t statistic for repeated k-fold CV (Nadeau and Bengio, 2003).

    Parameters
    ----------
    d : array-like
        Paired differences per fold.
    n_train, n_test : float
        Mean sizes of the outer training set and of the test fold.
    margin : float
        Null value: H0 is d̄ ≤ margin for the one-sided p (``p_greater``).

    Returns
    -------
    dict
        mean, sd, se (corrected), t, df, p_greater (one-sided, H1: d̄ > margin), p_two_sided (against 0),
        lower_95_one_sided, ci95 (two-sided).
    """
    d = np.asarray(d, float)
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt((1 / n + n_test / n_train) * sd ** 2))
    df = n - 1
    out = {'n': n, 'mean': mean, 'sd': sd, 'se_corrected': se, 'df': df}
    if se == 0 or df < 1:
        out.update(t=np.nan, p_greater=np.nan, p_two_sided=np.nan, lower_95_one_sided=mean, ci95=[mean, mean])
        return out
    q1, q2 = stats.t.ppf(0.95, df), stats.t.ppf(0.975, df)
    out.update(t=(mean - margin) / se, p_greater=float(stats.t.sf((mean - margin) / se, df)),
               p_two_sided=float(2 * stats.t.sf(abs(mean) / se, df)),
               lower_95_one_sided=mean - q1 * se, ci95=[mean - q2 * se, mean + q2 * se])
    return out


def bayes_correlated(d, n_train, n_test, rope=C.ROPE):
    """Bayesian correlated t-test (Corani and Benavoli, 2015): posterior of the mean difference.

    The posterior is a Student t with n − 1 degrees of freedom, location d̄ and scale
    √((1/n + ρ/(1 − ρ)) · s²), with ρ = n_test / (n_test + n_train).

    Returns
    -------
    dict
        p_left = P(d < −rope), p_rope = P(|d| ≤ rope), p_right = P(d > rope), and the posterior parameters.
    """
    d = np.asarray(d, float)
    n = len(d)
    rho = n_test / (n_test + n_train)
    mean, var = float(d.mean()), float(d.var(ddof=1)) if n > 1 else 0.0
    scale = float(np.sqrt((1 / n + rho / (1 - rho)) * var))
    if scale == 0:
        left, right = float(mean < -rope), float(mean > rope)
        return {'rho': rho, 'df': n - 1, 'loc': mean, 'scale': 0.0, 'p_left': left, 'p_rope': 1 - left - right,
                'p_right': right}
    cdf = lambda x: float(stats.t.cdf(x, n - 1, loc=mean, scale=scale))  # noqa: E731
    return {'rho': rho, 'df': n - 1, 'loc': mean, 'scale': scale, 'p_left': cdf(-rope),
            'p_rope': cdf(rope) - cdf(-rope), 'p_right': 1 - cdf(rope)}


def cliffs_delta(a, b):
    """Cliff's δ between the fold values of a and b (P(a > b) − P(a < b))."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).mean()
    lt = (a[:, None] < b[None, :]).mean()
    return float(gt - lt)


def wilcoxon_p(d):
    """Two-sided Wilcoxon signed-rank p-value of the paired differences (NaN if every difference is 0)."""
    d = np.asarray(d, float)
    if np.allclose(d, 0):
        return np.nan
    return float(stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided').pvalue)


def holm(pvalues):
    """Holm step-down adjusted p-values (same order as the input; NaN stays NaN)."""
    p = np.asarray(pvalues, float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[j]))
        out[j] = running
    return out


def kendalls_w(ranks):
    """Kendall's coefficient of concordance of a (raters × items) rank matrix (no tie correction)."""
    ranks = np.asarray(ranks, float)
    m, n = ranks.shape
    if m < 2 or n < 2:
        return np.nan
    r = ranks.sum(0)
    s = ((r - r.mean()) ** 2).sum()
    return float(12 * s / (m ** 2 * (n ** 3 - n)))


def suppress(n):
    """A count as text, with 1–4 shown as «<5»."""
    n = int(n)
    return '<5' if 0 < n < MIN_COUNT else str(n)


# ----------------------------------------------------------------------------- loading
def collect(workdir, load_tasks):
    """Tasks of phases 5 and 6 used by the analysis.

    Returns
    -------
    dict
        ``folds``: {(variant, fold): task} for folds 0–24 (phase 5 none · bayes and phase 6 confirmation);
        ``temporal``: {variant: task}; ``final``: {variant: task}.
    """
    folds, temporal, final = {}, {}, {}
    for r in load_tasks(workdir, 'development', missing_ok=True):
        if r['balancing'] == C.PHASE6_BALANCING and r['optimizer'] == C.PHASE6_OPTIMIZER and r['variant'] in C.VARIANTS:
            folds[(r['variant'], r['fold'])] = r
    for r in load_tasks(workdir, 'confirmation', missing_ok=True):
        folds[(r['variant'], r['fold'])] = r
    for r in load_tasks(workdir, 'temporal', missing_ok=True):
        temporal[r['variant']] = r
    for r in load_tasks(workdir, 'final', missing_ok=True):
        final[r['variant']] = r
    return {'folds': folds, 'temporal': temporal, 'final': final}


def fold_metrics(tasks):
    """Long table: one row per (variant, fold, probability kind) with every metric of the test fold."""
    rows = []
    for (v, f), r in sorted(tasks.items(), key=lambda kv: (C.VARIANTS.index(kv[0][0]), kv[0][1])):
        if r.get('error'):
            continue
        for kind, m in r['metrics'].items():
            row = {'variant': v, 'fold': f, 'repetition': f // C.OUTER + 1, 'probabilities': kind,
                   'n_train': r['n_fit'] + r['n_thresh'], 'n_test': r['n_test'], 'threshold': r['threshold']}
            row.update({k: val for k, val in m.items() if np.isscalar(val)})
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- comparisons
def compare(wide, a, b, folds, metric, sizes):
    """Paired comparison a − b on the given folds: corrected t, Bayesian, Cliff's δ, Wilcoxon, wins."""
    sub = wide.loc[wide.index.isin(folds)]
    if a not in sub or b not in sub:
        return None
    pair = sub[[a, b]].dropna()
    if len(pair) < 2:
        return None
    d = (pair[a] - pair[b]).to_numpy()
    n_tr, n_te = sizes.loc[pair.index, 'n_train'].mean(), sizes.loc[pair.index, 'n_test'].mean()
    t = corrected_t(d, n_tr, n_te)
    bay = bayes_correlated(d, n_tr, n_te, C.ROPE)
    return {'comparison': f'{a} − {b}', 'a': a, 'b': b, 'metric': metric, 'folds': len(d),
            'mean_a': float(pair[a].mean()), 'mean_b': float(pair[b].mean()), 'mean_difference': t['mean'],
            'sd_difference': t['sd'], 'se_corrected': t['se_corrected'], 't': t['t'], 'df': t['df'],
            'p_two_sided': t['p_two_sided'], 'ci95_low': t['ci95'][0], 'ci95_high': t['ci95'][1],
            'lower_95_one_sided': t['lower_95_one_sided'],
            'bayes_p_left': bay['p_left'], 'bayes_p_rope': bay['p_rope'], 'bayes_p_right': bay['p_right'],
            'bayes_scale': bay['scale'], 'cliffs_delta': cliffs_delta(pair[a], pair[b]),
            'wilcoxon_p': wilcoxon_p(d), 'folds_a_better': int((d > 0).sum()) if metric != 'brier' else int((d < 0).sum())}


def primary_decision(fm, sizes):
    """H1 (non-inferiority with weak calibration) and H2 (superiority, only if H1) of the protocol."""
    orig = fm[fm.probabilities == 'original']
    wide = orig.pivot(index='fold', columns='variant', values='auc_pr')
    a, b = PRIMARY
    sub = wide.loc[wide.index.isin(CONFIRMATION_FOLDS), [a, b]].dropna()
    out = {'question': 'RabMV − EBM, AUC-PR, folds 5–24', 'folds': int(len(sub))}
    if len(sub) < 2:
        out['status'] = 'not enough folds'
        return out
    d = (sub[a] - sub[b]).to_numpy()
    n_tr, n_te = sizes.loc[sub.index, 'n_train'].mean(), sizes.loc[sub.index, 'n_test'].mean()
    t_ni = corrected_t(d, n_tr, n_te, margin=-C.NI_MARGIN)
    t_sup = corrected_t(d, n_tr, n_te, margin=0.0)
    bay = bayes_correlated(d, n_tr, n_te, C.ROPE)
    cal = orig[(orig.variant == a) & orig.fold.isin(CONFIRMATION_FOLDS)]
    ci, cs = float(cal.calib_intercept.mean()), float(cal.calib_slope.mean())
    weak = C.CALIBRATION_INTERCEPT[0] <= ci <= C.CALIBRATION_INTERCEPT[1] and C.CALIBRATION_SLOPE[0] <= cs <= C.CALIBRATION_SLOPE[1]
    ni = t_ni['lower_95_one_sided'] > -C.NI_MARGIN
    h1 = bool(ni and weak)
    out.update(mean_rabmv=float(sub[a].mean()), mean_ebm=float(sub[b].mean()), mean_difference=t_ni['mean'],
               sd_difference=t_ni['sd'], se_corrected=t_ni['se_corrected'], df=t_ni['df'],
               n_train_mean=float(n_tr), n_test_mean=float(n_te),
               lower_95_one_sided=t_ni['lower_95_one_sided'], ni_margin=C.NI_MARGIN,
               p_noninferiority=t_ni['p_greater'], noninferior_auc_pr=bool(ni),
               rabmv_calibration_intercept_mean=ci, rabmv_calibration_slope_mean=cs,
               weak_calibration=bool(weak), H1=h1,
               bayes_p_left=bay['p_left'], bayes_p_rope=bay['p_rope'], bayes_p_right=bay['p_right'],
               bayes_p_noninferior=float(1 - bay['p_left']))
    if h1:
        h2 = bool(t_sup['p_greater'] < C.ALPHA and bay['p_right'] > C.BAYES_PROBABILITY)
        out.update(p_superiority=t_sup['p_greater'], H2=h2)
    else:
        out.update(p_superiority=None, H2='not tested (H1 does not hold)')
    return out


def comparisons_table(fm, sizes):
    """Every paired comparison of the protocol, on folds 5–24 and 0–24, for the four comparison metrics."""
    orig = fm[fm.probabilities == 'original']
    rows = []
    pairs = [('H1', *PRIMARY)] + SECONDARY
    for metric in COMPARISON_METRICS:
        wide = orig.pivot(index='fold', columns='variant', values=metric)
        for label, folds in (('5–24', CONFIRMATION_FOLDS), ('0–24', ALL_FOLDS)):
            for code, a, b in pairs:
                r = compare(wide, a, b, folds, metric, sizes)
                if r:
                    rows.append({'code': code, 'folds_used': label, **r})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t['p_holm'] = np.nan
    for (metric, label), g in t.groupby(['metric', 'folds_used']):
        m = g.code.str.startswith('S')
        t.loc[g.index[m], 'p_holm'] = holm(g.loc[m, 'p_two_sided'])
    return t


# ----------------------------------------------------------------------------- ablation
def ablate_missing(model):
    """Copy of a fitted EBM (inside the prediction pipeline) with every missing-value score set to 0.

    InterpretML keeps bin 0 of every feature for missing values; for a pair, row 0 and column 0. Nothing else
    changes (intercept, other scores). Returns the copy and the number of terms changed.
    """
    m = copy.deepcopy(model)
    ebm = m.steps[-1][1]
    ebm = getattr(ebm, 'estimator_', ebm)
    changed = 0
    for t, s in enumerate(ebm.term_scores_):
        s = np.array(s, float, copy=True)
        before = s.copy()
        for axis in range(s.ndim):
            idx = [slice(None)] * s.ndim
            idx[axis] = 0
            s[tuple(idx)] = 0.0
        changed += int(not np.array_equal(before, s))
        ebm.term_scores_[t] = s
    return m, changed


def ablation(tasks, temporal, data, progress):
    """Re-score every EBM_registry model with its missing scores set to 0 (folds 0–24 and temporal)."""
    X = data.input('EBM_registry')
    rows = []
    items = [((f), r) for (v, f), r in tasks.items() if v == 'EBM_registry' and not r.get('error')]
    if 'EBM_registry' in temporal and not temporal['EBM_registry'].get('error'):
        items.append((C.TEMPORAL_FOLD, temporal['EBM_registry']))
    for k, (f, r) in enumerate(sorted(items, key=lambda x: x[0]), 1):
        m, changed = ablate_missing(r['model'])
        te = np.asarray(r['idx_test'])
        p = m.predict_proba(X.iloc[te])[:, 1]
        y = data.y[te]
        miss = X.iloc[te].isna().any(axis=1).to_numpy()
        rows.append({'fold': f, 'auc_pr_ebm_registry': average_precision_score(y, r['p_test']),
                     'auc_pr_ablated': average_precision_score(y, p), 'brier_ablated': brier_score_loss(y, p),
                     'terms_changed': changed, 'test_cases_with_missing': int(miss.sum()),
                     'events_with_missing': int((miss & (y == 1)).sum()), 'p_ablated': p})
        if k % 5 == 0 or k == len(items):
            progress.note(f'ablation: {k}/{len(items)} models re-scored')
    return rows


# ----------------------------------------------------------------------------- complementary
def counts(data, labels):
    """Supporting counts per class (aggregates; 1–4 shown as «<5»)."""
    Xr, y, year = data.X_registry, data.y, data.year
    rows = []
    miss = Xr['escolaridad'].isna().to_numpy()
    for name, m in (('all years', np.ones(len(y), bool)), ('2016', year == 2016), ('2017–2022', year >= 2017)):
        for cls, cname in ((0, 'attempt (form 356)'), (1, 'completed suicide (form 105)')):
            sel = m & (y == cls)
            rows.append({'table': 'education missing', 'group': name, 'class': cname, 'category': 'missing',
                         'count': suppress((sel & miss).sum()), 'of': suppress(sel.sum())})
    reg = Xr['regimen'].to_numpy()
    for code in sorted(pd.unique(reg[np.isfinite(reg)])):
        for cls, cname in ((0, 'attempt (form 356)'), (1, 'completed suicide (form 105)')):
            sel = y == cls
            label = labels._group('regimen', code) if labels else str(code)
            rows.append({'table': 'insurance scheme', 'group': 'all years', 'class': cname,
                         'category': f'{int(code)}: {label}', 'count': suppress((sel & (reg == code)).sum()),
                         'of': suppress(sel.sum())})
    for cls, cname in ((0, 'attempt (form 356)'), (1, 'completed suicide (form 105)')):
        sel = y == cls
        rows.append({'table': 'insurance scheme', 'group': 'all years', 'class': cname, 'category': 'missing',
                     'count': suppress((sel & ~np.isfinite(reg)).sum()), 'of': suppress(sel.sum())})
    return pd.DataFrame(rows)


def calibration_table(fm):
    """Mean and SD of the calibration metrics per variant, probability kind and set of folds."""
    rows = []
    for label, folds in (('5–24', CONFIRMATION_FOLDS), ('0–24', ALL_FOLDS)):
        sub = fm[fm.fold.isin(folds)]
        for (v, kind), g in sub.groupby(['variant', 'probabilities'], sort=False):
            row = {'folds_used': label, 'variant': v, 'probabilities': kind, 'folds': len(g)}
            for m in ('auc_pr', 'auc_roc', 'brier', 'ece', 'calib_intercept', 'calib_slope', 'precision', 'recall',
                      'f1', 'specificity'):
                row[f'{m}_mean'], row[f'{m}_sd'] = float(g[m].mean()), float(g[m].std())
            rows.append(row)
    return pd.DataFrame(rows)


def equity(tasks, data):
    """Sensitivity by sex: mean over folds 5–24 (threshold of each fold) and per repetition (each case once,
    averaged over repetitions 2–5)."""
    rows = []
    for v in C.VARIANTS:
        per_fold, per_rep, events = [], [], None
        for rep in range(2, C.REPETITIONS + 1):
            pooled = {0: [0, 0], 1: [0, 0]}
            complete = True
            for f in range((rep - 1) * C.OUTER, rep * C.OUTER):
                r = tasks.get((v, f))
                if r is None or r.get('error'):
                    complete = False
                    continue
                m = r['metrics']['original']
                per_fold.append((m['tpr_women'], m['tpr_men'], m['tpr_gap']))
                te = np.asarray(r['idx_test'])
                sex, y = data.sex[te], data.y[te]
                pred = np.asarray(r['p_test']) >= r['threshold']
                for g in (0, 1):
                    sel = (sex == g) & (y == 1)
                    pooled[g][0] += int(pred[sel].sum())
                    pooled[g][1] += int(sel.sum())
            if complete and pooled[0][1] and pooled[1][1]:
                per_rep.append((pooled[0][0] / pooled[0][1], pooled[1][0] / pooled[1][1]))
                events = (pooled[0][1], pooled[1][1])
        if not per_fold:
            continue
        a = np.array(per_fold, float)
        b = np.array(per_rep, float) if per_rep else np.full((1, 2), np.nan)
        tw, tm = float(np.mean(b[:, 0])), float(np.mean(b[:, 1]))
        rows.append({'variant': v, 'folds': len(a), 'tpr_women_mean': np.nanmean(a[:, 0]),
                     'tpr_men_mean': np.nanmean(a[:, 1]), 'tpr_gap_mean': np.nanmean(a[:, 2]),
                     'repetitions': len(per_rep), 'tpr_women_by_repetition': tw, 'tpr_men_by_repetition': tm,
                     'tpr_gap_by_repetition': abs(tw - tm),
                     'events_women': suppress(events[0]) if events else '', 'events_men': suppress(events[1]) if events else ''})
    return pd.DataFrame(rows)


def net_benefit(y, p, thresholds):
    """Net benefit of the model and of «alert everyone» at each threshold probability."""
    y, p = np.asarray(y), np.asarray(p, float)
    n, prev = len(y), y.mean()
    out = []
    for pt in thresholds:
        alert = p >= pt
        tp, fp = int((alert & (y == 1)).sum()), int((alert & (y == 0)).sum())
        w = pt / (1 - pt)
        out.append((tp / n - fp / n * w, prev - (1 - prev) * w))
    return np.array(out)


def decision_curve(tasks):
    """Net benefit per variant: per repetition (each case once), averaged over repetitions 2–5."""
    lo, hi, step = C.DCA_THRESHOLDS
    thr = np.round(np.arange(lo, hi + step / 2, step), 4)
    rows = []
    for v in C.VARIANTS:
        reps = []
        for rep in range(2, C.REPETITIONS + 1):
            fs = [tasks.get((v, f)) for f in range((rep - 1) * C.OUTER, rep * C.OUTER)]
            if any(r is None or r.get('error') for r in fs):
                continue
            y = np.concatenate([r['y_test'] for r in fs])
            p = np.concatenate([r['p_test'] for r in fs])
            reps.append(net_benefit(y, p, thr))
        if not reps:
            continue
        a = np.array(reps)
        for i, pt in enumerate(thr):
            rows.append({'variant': v, 'threshold': pt, 'repetitions': len(a), 'net_benefit': a[:, i, 0].mean(),
                         'net_benefit_min': a[:, i, 0].min(), 'net_benefit_max': a[:, i, 0].max(),
                         'treat_all': a[:, i, 1].mean(), 'treat_none': 0.0})
    return pd.DataFrame(rows)


def _main_importances(card):
    """{feature name: importance} of the main effects of a model card; and the list of pair names."""
    names = card.get('input_features') or list(card.get('feature_names_in', []))
    imp = card.get('term_importances')
    out, pairs = {}, []
    if card['classifier'] == 'RabMVClassifier':
        p = len(card['bin_edges'])
        for t, name in enumerate(card['term_names']):
            if t < p:
                out[str(names[t]) if t < len(names) else name] = float(imp[t])
            else:
                a, b = card['interaction_pairs'][t - p]
                pairs.append(' & '.join(str(names[i]) if i < len(names) else str(i) for i in (a, b)))
    else:
        for t, (name, feats) in enumerate(zip(card['term_names'], card['term_features'])):
            if len(feats) == 1:
                out[str(names[feats[0]]) if feats[0] < len(names) else name] = float(imp[t])
            else:
                pairs.append(' & '.join(str(names[i]) if i < len(names) else str(i) for i in feats))
    return out, pairs


def stability(tasks, labels):
    """Importance of each main term across folds, Spearman between folds, Kendall's W and interactions."""
    imp_rows, summary, inter = [], [], []
    for v in C.VARIANTS:
        mats, n_folds = [], 0
        counts_pairs = {}
        for f in ALL_FOLDS:
            r = tasks.get((v, f))
            if r is None or r.get('error'):
                continue
            n_folds += 1
            imp, pairs = _main_importances(r['model_card'])
            mats.append(imp)
            for name, val in imp.items():
                imp_rows.append({'variant': v, 'fold': f, 'term': name,
                                 'term_en': labels.name(name) if labels else name, 'importance': val})
            for pr in pairs:
                counts_pairs[pr] = counts_pairs.get(pr, 0) + 1
        if not mats:
            continue
        terms = sorted(set().union(*mats))
        M = np.array([[m.get(t, 0.0) for t in terms] for m in mats])
        rho = []
        for i in range(len(M)):
            for j in range(i + 1, len(M)):
                if np.std(M[i]) > 0 and np.std(M[j]) > 0:
                    rho.append(stats.spearmanr(M[i], M[j]).statistic)
        ranks = np.array([stats.rankdata(-row) for row in M])
        summary.append({'variant': v, 'folds': n_folds, 'terms': len(terms),
                        'spearman_mean': float(np.mean(rho)) if rho else np.nan,
                        'spearman_min': float(np.min(rho)) if rho else np.nan, 'kendalls_w': kendalls_w(ranks)})
        for pr, k in sorted(counts_pairs.items(), key=lambda kv: -kv[1]):
            inter.append({'variant': v, 'pair': pr, 'pair_en': labels.name(pr) if labels else pr, 'folds_selected': k,
                          'of_folds': n_folds})
    return pd.DataFrame(imp_rows), pd.DataFrame(summary), pd.DataFrame(inter)


def hyperparameters(tasks):
    """Chosen hyperparameters per variant and fold."""
    rows = []
    for (v, f), r in sorted(tasks.items(), key=lambda kv: (C.VARIANTS.index(kv[0][0]), kv[0][1])):
        if r.get('best_params'):
            rows.append({'variant': v, 'fold': f, **r['best_params']})
    return pd.DataFrame(rows)


def cost(tasks, data, progress):
    """Search, refit and prediction time per variant (prediction measured here on each test fold)."""
    rows = []
    items = [(k, r) for k, r in tasks.items() if not r.get('error') and k[1] in ALL_FOLDS]
    for i, ((v, f), r) in enumerate(items, 1):
        X = data.input(v).iloc[np.asarray(r['idx_test'])]
        t0 = time.perf_counter()
        r['model'].predict_proba(X)
        ms = (time.perf_counter() - t0) * 1000 / max(len(X), 1) * 1000
        rows.append({'variant': v, 'fold': f, 'minutes_search': r['seconds_optimization'] / 60,
                     'seconds_refit': r.get('seconds_refit', np.nan), 'evaluations': r.get('evaluations'),
                     'ms_per_1000_predictions': ms})
        if i % 25 == 0 or i == len(items):
            progress.note(f'cost: {i}/{len(items)} models timed')
    t = pd.DataFrame(rows)
    return t, t.groupby('variant', sort=False)[['minutes_search', 'seconds_refit', 'ms_per_1000_predictions']].agg(
        ['mean', 'std']).reset_index()


def temporal_table(temporal):
    """Metrics of the temporal validation and the paired differences of the protocol (descriptive)."""
    rows = []
    for v in C.VARIANTS:
        r = temporal.get(v)
        if r is None or r.get('error'):
            continue
        m = r['metrics']['original']
        rows.append({'variant': v, 'n_train': r['n_fit'] + r['n_thresh'], 'n_test': r['n_test'],
                     'events_test': suppress(int(np.sum(r['y_test']))), 'threshold': r['threshold'],
                     **{k: m[k] for k in ('auc_pr', 'auc_roc', 'brier', 'ece', 'calib_intercept', 'calib_slope',
                                          'precision', 'recall', 'f1', 'specificity')}})
    t = pd.DataFrame(rows)
    diffs = []
    if not t.empty:
        s = t.set_index('variant')
        for code, a, b in [('H1', *PRIMARY)] + SECONDARY:
            if a in s.index and b in s.index:
                diffs.append({'code': code, 'comparison': f'{a} − {b}',
                              **{f'd_{k}': s.loc[a, k] - s.loc[b, k] for k in ('auc_pr', 'auc_roc', 'brier', 'f1')}})
    return t, pd.DataFrame(diffs)


def final_table(final):
    """Out-of-fold and apparent metrics, threshold and hyperparameters of the final models."""
    rows = []
    for v in C.VARIANTS:
        r = final.get(v)
        if r is None or r.get('error'):
            continue
        for kind in ('original', 'apparent'):
            m = r['metrics'][kind]
            rows.append({'variant': v, 'probabilities': 'out-of-fold' if kind == 'original' else 'apparent',
                         'n': r['n_test'], 'threshold': r['threshold'], 'hyperparameters': json.dumps(r['best_params']),
                         **{k: m[k] for k in ('auc_pr', 'auc_roc', 'brier', 'ece', 'calib_intercept', 'calib_slope',
                                              'precision', 'recall', 'f1', 'specificity')}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


VARIANT_COLOURS = {'EBM': '#7f7f7f', 'RabMV': '#1f4e79', 'RabMV_registry': '#2e75b6',
                   'RabMV_common_education': '#9dc3e6', 'EBM_registry': '#c55a11', ABLATED: '#f4b183'}


def fig_forest(comp, path):
    """Mean paired difference of AUC-PR with its corrected 95 % CI (folds 5–24 filled, 0–24 hollow)."""
    plt = _plt()
    sub = comp[comp.metric == 'auc_pr']
    labels = list(dict.fromkeys(sub.code + '  ' + sub.comparison))
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(labels) + 1.5))
    for i, lab in enumerate(labels):
        for folds_used, style, off in (('5–24', 'o', 0.12), ('0–24', 'o', -0.12)):
            r = sub[(sub.code + '  ' + sub.comparison == lab) & (sub.folds_used == folds_used)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.plot([r.ci95_low, r.ci95_high], [i + off] * 2, color='#1f4e79', linewidth=1.2)
            ax.plot(r.mean_difference, i + off, style, color='#1f4e79',
                    markerfacecolor='#1f4e79' if folds_used == '5–24' else 'white', markersize=6)
    ax.axvline(0, color='grey', linewidth=0.8)
    ax.axvline(-C.NI_MARGIN, color='#c00000', linewidth=0.8, linestyle='--')
    ax.text(-C.NI_MARGIN, 1.0, 'non-inferiority margin ', color='#c00000', fontsize=7, ha='right', va='bottom',
            transform=ax.get_xaxis_transform())
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Difference in AUC-PR (a − b), mean and corrected 95 % CI')
    ax.set_title('Paired comparisons: filled = folds 5–24 (confirmatory), hollow = folds 0–24', fontsize=9, pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_posteriors(comp, path):
    """Posterior density of the mean AUC-PR difference (Bayesian correlated t-test) with the ROPE."""
    plt = _plt()
    sub = comp[(comp.metric == 'auc_pr') & (comp.folds_used == '5–24')]
    if sub.empty:
        return
    n = len(sub)
    fig, axes = plt.subplots(int(np.ceil(n / 3)), min(3, n), figsize=(4.2 * min(3, n), 3.2 * int(np.ceil(n / 3))),
                             squeeze=False)
    for ax, (_, r) in zip(axes.ravel(), sub.iterrows()):
        if r.bayes_scale > 0:
            x = np.linspace(r.mean_difference - 5 * r.bayes_scale, r.mean_difference + 5 * r.bayes_scale, 400)
            ax.plot(x, stats.t.pdf(x, r.df, loc=r.mean_difference, scale=r.bayes_scale), color='#1f4e79')
        ax.axvspan(-C.ROPE, C.ROPE, color='#d9d9d9', alpha=0.6, linewidth=0)
        ax.axvline(0, color='grey', linewidth=0.6)
        ax.set_title(f'{r.code}: {r.comparison}', fontsize=8)
        ax.set_xlabel('Difference in AUC-PR', fontsize=7)
        ax.text(0.02, 0.95, f'P(<−ROPE) {r.bayes_p_left:.3f}\nP(ROPE) {r.bayes_p_rope:.3f}\nP(>ROPE) {r.bayes_p_right:.3f}',
                transform=ax.transAxes, va='top', fontsize=7)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    fig.suptitle(f'Bayesian correlated t-test, folds 5–24 (ROPE ±{C.ROPE})', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_folds(fm, path):
    """AUC-PR of every test fold per variant, coloured by repetition; mean as a bar."""
    plt = _plt()
    orig = fm[fm.probabilities == 'original']
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cmap = plt.get_cmap('viridis')
    for i, v in enumerate(C.VARIANTS):
        g = orig[orig.variant == v]
        if g.empty:
            continue
        ax.bar(i, g[g.fold.isin(CONFIRMATION_FOLDS)].auc_pr.mean(), color='#eaf1fb', edgecolor='#1f4e79')
        for rep in sorted(g.repetition.unique()):
            h = g[g.repetition == rep]
            ax.scatter(np.full(len(h), i + (rep - 3) * 0.08), h.auc_pr, s=10, color=cmap((rep - 1) / 4), label=f'repetition {rep}' if i == 0 else None)
    ax.set_xticks(range(len(C.VARIANTS)))
    ax.set_xticklabels(C.VARIANTS, fontsize=8, rotation=15)
    ax.set_ylabel('AUC-PR (test fold)')
    ax.set_title('AUC-PR per fold (points) and mean over folds 5–24 (bar)', fontsize=9)
    ax.legend(fontsize=7, ncol=5, loc='lower center')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_ablation(abl, tasks, path):
    """Per fold: EBM_registry, the same model without its missing scores, and RabMV_registry."""
    plt = _plt()
    folds = [r['fold'] for r in abl if r['fold'] in ALL_FOLDS]
    if not folds:
        return
    a = np.array([[r['auc_pr_ebm_registry'], r['auc_pr_ablated'],
                   tasks[('RabMV_registry', r['fold'])]['metrics']['original']['auc_pr']
                   if ('RabMV_registry', r['fold']) in tasks else np.nan] for r in abl if r['fold'] in ALL_FOLDS])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    names = ['EBM_registry', ABLATED, 'RabMV_registry']
    for row in a:
        ax.plot(range(3), row, color='#bfbfbf', linewidth=0.6)
    for j, n in enumerate(names):
        ax.scatter(np.full(len(a), j), a[:, j], s=12, color=VARIANT_COLOURS.get(n, '#1f4e79'), zorder=3)
        ax.plot([j - 0.2, j + 0.2], [np.nanmean(a[:, j])] * 2, color='black', linewidth=2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['EBM_registry', 'EBM_registry\nmissing scores = 0', 'RabMV_registry'], fontsize=8)
    ax.set_ylabel('AUC-PR (test fold)')
    ax.set_title('Shortcut ablation, folds 0–24 (lines join the same fold; bar = mean)', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_pooled_curves(tasks, folder):
    """Pooled precision-recall and binned calibration curves per variant (folds 5–24)."""
    plt = _plt()
    for kind in ('pr', 'calibration'):
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for v in C.VARIANTS:
            rs = [tasks.get((v, f)) for f in CONFIRMATION_FOLDS]
            rs = [r for r in rs if r is not None and not r.get('error')]
            if not rs:
                continue
            y = np.concatenate([r['y_test'] for r in rs])
            p = np.concatenate([r['p_test'] for r in rs])
            if kind == 'pr':
                prec, rec, _ = precision_recall_curve(y, p)
                ax.plot(rec, prec, label=v, linewidth=1, color=VARIANT_COLOURS[v])
            else:
                q = np.quantile(p, np.linspace(0, 1, 11))
                idx = np.clip(np.searchsorted(q[1:-1], p, side='right'), 0, 9)
                pts = [(p[idx == k].mean(), y[idx == k].mean()) for k in range(10) if (idx == k).sum() >= MIN_BIN]
                if pts:
                    ax.plot(*zip(*pts), marker='o', markersize=3, label=v, linewidth=1, color=VARIANT_COLOURS[v])
        if kind == 'pr':
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-recall, test folds 5–24 pooled', fontsize=9)
        else:
            lim = ax.get_xlim()[1]
            ax.plot([0, lim], [0, lim], color='grey', linewidth=0.6)
            ax.set_xlabel('Mean predicted probability (decile)')
            ax.set_ylabel('Observed proportion')
            ax.set_title('Calibration, test folds 5–24 pooled', fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(Path(folder) / ('pr_curves_pooled.png' if kind == 'pr' else 'calibration_pooled.png'), dpi=200)
        plt.close(fig)


def fig_decision_curve(dca, path):
    """Net benefit against the threshold probability (mean over repetitions 2–5)."""
    if dca.empty:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for v, g in dca.groupby('variant', sort=False):
        ax.plot(g.threshold, g.net_benefit, label=v, color=VARIANT_COLOURS[v], linewidth=1.2)
    g = dca[dca.variant == dca.variant.iloc[0]]
    ax.plot(g.threshold, g.treat_all, label='Alert everyone', color='black', linestyle='--', linewidth=0.8)
    ax.axhline(0, color='black', linewidth=0.8, label='Alert no one')
    ax.set_ylim(min(-0.01, dca.net_benefit.min() - 0.005), dca[['net_benefit', 'treat_all']].max().max() + 0.01)
    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit')
    ax.set_title('Decision curve, mean over repetitions 2–5', fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_equity(eq, path):
    """Sensitivity in women and men per variant (mean over folds 5–24)."""
    if eq.empty:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(eq))
    ax.bar(x - 0.18, eq.tpr_women_mean, 0.36, label='Women', color='#c55a11')
    ax.bar(x + 0.18, eq.tpr_men_mean, 0.36, label='Men', color='#1f4e79')
    ax.set_xticks(x)
    ax.set_xticklabels(eq.variant, fontsize=8, rotation=15)
    ax.set_ylabel('Sensitivity (TPR) at the fold threshold')
    ax.set_title('Equity by sex, mean over folds 5–24', fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_importances(imp, path):
    """Importance of each main term across folds 0–24, per variant (box plots)."""
    if imp.empty:
        return
    plt = _plt()
    vs = [v for v in C.VARIANTS if v in set(imp.variant)]
    fig, axes = plt.subplots(len(vs), 1, figsize=(9, 2.6 * len(vs)), squeeze=False)
    for ax, v in zip(axes.ravel(), vs):
        g = imp[imp.variant == v]
        order = g.groupby('term_en').importance.median().sort_values(ascending=False).index
        ax.boxplot([g[g.term_en == t].importance for t in order], showfliers=False)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, fontsize=6, rotation=40, ha='right')
        ax.set_title(v, fontsize=8)
        ax.set_ylabel('Importance', fontsize=7)
    fig.suptitle('Term importance (mean |f| on the training data) across folds 0–24', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_temporal(tt, path):
    """AUC-PR of the temporal validation (fit 2016–2020, test 2021–2022)."""
    if tt.empty:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(tt)), tt.auc_pr, color=[VARIANT_COLOURS[v] for v in tt.variant])
    ax.set_xticks(range(len(tt)))
    ax.set_xticklabels(tt.variant, fontsize=8, rotation=15)
    ax.set_ylabel('AUC-PR, 2021–2022')
    ax.set_title(f'Temporal validation: fit 2016–{C.TEMPORAL_SPLIT_YEAR}, test {C.TEMPORAL_SPLIT_YEAR + 1}–2022', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- results text
def _f(x, nd=4):
    return 'NA' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}f}'


def results_text(dec, comp, abl_summary, temporal_t, final_t):
    """The decisions of the protocol written out (RESULTS_PHASE6.md, English, numbers only from the tables)."""
    p = dec['primary']
    lines = ['# Results of phase 6 (pre-specified protocol PROTOCOL_PHASE6.md)', '',
             f"Generated {dec['generated']} from the checkpoints; protocol SHA-256 {dec['protocol_sha256'][:16].upper()}.", '',
             '## Primary question (H1, H2)', '']
    if 'mean_difference' in p:
        lines += [f"* Folds 5–24 ({p['folds']}): mean AUC-PR RabMV {_f(p['mean_rabmv'])}, EBM {_f(p['mean_ebm'])}; "
                  f"difference {_f(p['mean_difference'])} (corrected SE {_f(p['se_corrected'])}).",
                  f"* One-sided 95 % lower bound {_f(p['lower_95_one_sided'])} against the margin −{C.NI_MARGIN}: "
                  f"{'non-inferior' if p['noninferior_auc_pr'] else 'non-inferiority not shown'} "
                  f"(p = {_f(p['p_noninferiority'])}).",
                  f"* RabMV calibration on folds 5–24: mean intercept {_f(p['rabmv_calibration_intercept_mean'], 3)}, "
                  f"mean slope {_f(p['rabmv_calibration_slope_mean'], 3)} → weak calibration "
                  f"{'met' if p['weak_calibration'] else 'not met'}.",
                  f"* **H1: {'holds' if p['H1'] else 'does not hold'}.** H2 (superiority): {p['H2']}.",
                  f"* Bayesian correlated t-test (ROPE ±{C.ROPE}): P(RabMV worse) {_f(p['bayes_p_left'], 3)}, "
                  f"P(practically equivalent) {_f(p['bayes_p_rope'], 3)}, P(RabMV better) {_f(p['bayes_p_right'], 3)}."]
    else:
        lines.append(f"* {p.get('status')}")
    lines += ['', '## Secondary comparisons (AUC-PR, folds 5–24, Holm)', '',
              '| Code | Comparison | Mean difference | Corrected 95 % CI | p (Holm) | P(<−ROPE) / P(ROPE) / P(>ROPE) | Folds a better |',
              '|---|---|---|---|---|---|---|']
    sub = comp[(comp.metric == 'auc_pr') & (comp.folds_used == '5–24')] if not comp.empty else comp
    for _, r in sub.iterrows():
        lines.append(f"| {r.code} | {r.comparison} | {_f(r.mean_difference)} | [{_f(r.ci95_low)}, {_f(r.ci95_high)}] | "
                     f"{_f(r.p_holm if r.code.startswith('S') else r.p_two_sided, 4)} | {_f(r.bayes_p_left, 3)} / "
                     f"{_f(r.bayes_p_rope, 3)} / {_f(r.bayes_p_right, 3)} | {r.folds_a_better}/{r.folds} |")
    lines += ['', '## Shortcut ablation of EBM_registry', '']
    if abl_summary:
        a = abl_summary
        lines += [f"* Folds 5–24: EBM_registry {_f(a['mean_ebm_registry'])}, without missing scores {_f(a['mean_ablated'])}, "
                  f"RabMV_registry {_f(a['mean_rabmv_registry'])} (gap EBM_registry − RabMV_registry "
                  f"{_f(a['gap_ebm_registry_minus_rabmv_registry'])}).",
                  f"* Share of the S1 gap removed by the ablation: {_f(a['share_of_gap_removed'], 3)} "
                  f"(pre-specified reading: ≥ 0.5 and negative Δ → the advantage is attributed mainly to the missing-value "
                  f"score): **{a['reading']}**.",
                  f"* Ablated EBM_registry − RabMV_registry: {_f(a['A_mean'])} (corrected 95 % CI [{_f(a['A_ci95'][0])}, "
                  f"{_f(a['A_ci95'][1])}]; Bayesian P(ROPE) {_f(a['A_bayes_p_rope'], 3)})."]
    else:
        lines.append('* Not available (no EBM_registry models).')
    lines += ['', '## Temporal validation (descriptive)', '']
    if not temporal_t.empty:
        lines += ['| Variant | AUC-PR | AUC-ROC | Brier | Calibration slope | F1 |', '|---|---|---|---|---|---|']
        lines += [f"| {r.variant} | {_f(r.auc_pr)} | {_f(r.auc_roc)} | {_f(r.brier)} | {_f(r.calib_slope, 3)} | {_f(r.f1)} |"
                  for _, r in temporal_t.iterrows()]
    else:
        lines.append('* Not available.')
    lines += ['', '## Final models (out-of-fold internal validation)', '']
    if not final_t.empty:
        oof = final_t[final_t.probabilities == 'out-of-fold']
        lines += ['| Variant | AUC-PR | Brier | Calibration slope | Threshold |', '|---|---|---|---|---|']
        lines += [f"| {r.variant} | {_f(r.auc_pr)} | {_f(r.brier)} | {_f(r.calib_slope, 3)} | {_f(r.threshold)} |"
                  for _, r in oof.iterrows()]
    else:
        lines.append('* Not available.')
    lines += ['', 'All tables are in `tables/` and the figures in `figures/`. Only aggregates; counts 1–4 shown as «<5».', '']
    return '\n'.join(lines)


# ----------------------------------------------------------------------------- entry point
def sha16(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()[:16].upper()


def protocol_fingerprint():
    """SHA-256 of PROTOCOL_PHASE6.md; the analysis refuses to run if it is not the registered one."""
    path = Path(__file__).resolve().parent / 'PROTOCOL_PHASE6.md'
    h = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()     # same on Windows and Linux
    return h, h == C.PROTOCOL_SHA256


def run(workdir, data_path, schema, out, load_tasks, check_fingerprint=True):
    """Run the whole analysis of the protocol and write every output into ``out``."""
    from .report import Labels
    P = Progress(13)
    proto, ok = protocol_fingerprint()
    if not ok:
        raise SystemExit(f'PROTOCOL_PHASE6.md changed (SHA-256 {proto[:16].upper()}, registered '
                         f'{C.PROTOCOL_SHA256[:16].upper()}): the analysis follows the registered protocol only')
    out = Path(out)
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    tab = lambda df, name: df.to_csv(out / 'tables' / name, index=False, lineterminator='\n')  # noqa: E731

    P.step('loading checkpoints of phases 5 and 6')
    got = collect(workdir, load_tasks)
    tasks, temporal, final = got['folds'], got['temporal'], got['final']
    have = {v: sorted(f for (w, f) in tasks if w == v) for v in C.VARIANTS}
    P.note('folds per variant: ' + '; '.join(f'{v} {len(f)}' for v, f in have.items()) +
           f'; temporal {len(temporal)}; final {len(final)}')
    P.step('loading the data and the schema (scoring of the ablation, counts, equity)')
    data = D.load(data_path, schema, check_fingerprint=check_fingerprint)
    labels = Labels(schema) if (Path(schema) / 'labels_en.toml').exists() else None
    for (v, f), r in tasks.items():
        if r.get('data_fingerprint') != data.fingerprint:
            raise SystemExit(f'task {v} fold {f} was run with other data ({str(r.get("data_fingerprint"))[:16].upper()})')

    P.step('metrics per fold')
    fm = fold_metrics(tasks)
    tab(fm, 'metrics_by_fold.csv')
    sizes = fm[fm.probabilities == 'original'].groupby('fold')[['n_train', 'n_test']].mean()

    P.step('primary decision (H1, H2) and secondary comparisons')
    primary = primary_decision(fm, sizes)
    comp = comparisons_table(fm, sizes)
    tab(comp, 'comparisons.csv')

    P.step('shortcut ablation of EBM_registry (re-scoring without training)')
    abl = ablation(tasks, temporal, data, P)
    abl_t = pd.DataFrame([{k: v for k, v in r.items() if k != 'p_ablated'} for r in abl])
    abl_summary = None
    if not abl_t.empty:
        abl_t['auc_pr_rabmv_registry'] = [tasks[('RabMV_registry', f)]['metrics']['original']['auc_pr']
                                          if ('RabMV_registry', f) in tasks else
                                          (temporal['RabMV_registry']['metrics']['original']['auc_pr']
                                           if f == C.TEMPORAL_FOLD and 'RabMV_registry' in temporal else np.nan)
                                          for f in abl_t.fold]
        abl_t['delta_ablation'] = abl_t.auc_pr_ablated - abl_t.auc_pr_ebm_registry
        abl_t['A_ablated_minus_rabmv_registry'] = abl_t.auc_pr_ablated - abl_t.auc_pr_rabmv_registry
        out_t = abl_t.copy()
        out_t['test_cases_with_missing'] = out_t.test_cases_with_missing.map(suppress)
        out_t['events_with_missing'] = out_t.events_with_missing.map(suppress)
        tab(out_t, 'ablation_by_fold.csv')
        c = abl_t[abl_t.fold.isin(CONFIRMATION_FOLDS)].dropna(subset=['auc_pr_rabmv_registry'])
        if len(c) >= 2:
            gap = c.auc_pr_ebm_registry.mean() - c.auc_pr_rabmv_registry.mean()
            removed = c.auc_pr_ebm_registry.mean() - c.auc_pr_ablated.mean()
            share = removed / gap if gap > 0 else np.nan          # not applicable when EBM_registry is not ahead
            n_tr, n_te = sizes.loc[c.fold, 'n_train'].mean(), sizes.loc[c.fold, 'n_test'].mean()
            tA = corrected_t(c.A_ablated_minus_rabmv_registry, n_tr, n_te)
            bA = bayes_correlated(c.A_ablated_minus_rabmv_registry, n_tr, n_te)
            tD = corrected_t(c.delta_ablation, n_tr, n_te)
            reading = ('advantage attributed mainly to the missing-value score' if (np.isfinite(share) and share >= 0.5
                                                                                     and c.delta_ablation.mean() < 0)
                       else 'advantage not attributed mainly to the missing-value score')
            abl_summary = {'folds': int(len(c)), 'gap_ebm_registry_minus_rabmv_registry': float(gap),
                           'mean_ebm_registry': float(c.auc_pr_ebm_registry.mean()),
                           'mean_ablated': float(c.auc_pr_ablated.mean()),
                           'mean_rabmv_registry': float(c.auc_pr_rabmv_registry.mean()),
                           'delta_ablation_mean': tD['mean'], 'delta_ablation_ci95': tD['ci95'],
                           'share_of_gap_removed': float(share), 'reading': reading,
                           'A_mean': tA['mean'], 'A_ci95': tA['ci95'], 'A_p_two_sided': tA['p_two_sided'],
                           'A_bayes_p_left': bA['p_left'], 'A_bayes_p_rope': bA['p_rope'], 'A_bayes_p_right': bA['p_right']}

    P.step('supporting counts (aggregates, 1–4 suppressed)')
    tab(counts(data, labels), 'supporting_counts.csv')

    P.step('calibration, equity by sex and decision curve')
    tab(calibration_table(fm), 'calibration_summary.csv')
    eq = equity(tasks, data)
    tab(eq, 'equity_by_sex.csv')
    dca = decision_curve(tasks)
    tab(dca, 'decision_curve.csv')

    P.step('stability of the interpretation, interactions and hyperparameters')
    imp, stab, inter = stability(tasks, labels)
    tab(imp, 'importances_by_fold.csv')
    tab(stab, 'stability_summary.csv')
    tab(inter, 'interactions_selected.csv')
    tab(hyperparameters(tasks), 'hyperparameters_by_fold.csv')

    P.step('computational cost (prediction timed on each test fold)')
    cost_t, cost_s = cost(tasks, data, P)
    tab(cost_t, 'cost_by_fold.csv')
    cost_s.columns = ['_'.join(c).strip('_') for c in cost_s.columns]
    tab(cost_s, 'cost_summary.csv')

    P.step('temporal validation and final models')
    tt, td = temporal_table(temporal)
    tab(tt, 'temporal_validation.csv')
    tab(td, 'temporal_differences.csv')
    ft = final_table(final)
    tab(ft, 'final_models.csv')

    P.step('figures')
    if not comp.empty:
        fig_forest(comp, out / 'figures' / 'forest_comparisons.png')
        fig_posteriors(comp, out / 'figures' / 'bayesian_posteriors.png')
    fig_folds(fm, out / 'figures' / 'auc_pr_by_fold.png')
    fig_ablation(abl, tasks, out / 'figures' / 'shortcut_ablation.png')
    fig_pooled_curves(tasks, out / 'figures')
    fig_decision_curve(dca, out / 'figures' / 'decision_curve.png')
    fig_equity(eq, out / 'figures' / 'equity_by_sex.png')
    fig_importances(imp, out / 'figures' / 'importances.png')
    fig_temporal(tt, out / 'figures' / 'temporal_validation.png')

    P.step('decisions and results text')
    expected = {v: CONFIRMATION_FOLDS for v in C.VARIANTS}
    complete = all(set(expected[v]) <= set(have[v]) for v in C.VARIANTS)
    dec = {'generated': dt.datetime.now().isoformat(timespec='seconds'), 'protocol_sha256': proto,
           'data_fingerprint': data.fingerprint, 'confirmation_complete': complete,
           'folds_available': {v: len(f) for v, f in have.items()},
           'primary': primary, 'ablation': abl_summary,
           'secondary': (comp[(comp.metric == 'auc_pr') & (comp.folds_used == '5–24') & comp.code.str.startswith('S')]
                         [['code', 'comparison', 'mean_difference', 'ci95_low', 'ci95_high', 'p_two_sided', 'p_holm',
                           'bayes_p_left', 'bayes_p_rope', 'bayes_p_right', 'cliffs_delta', 'folds_a_better']]
                         .to_dict('records') if not comp.empty else [])}
    if not complete:
        dec['warning'] = 'confirmation incomplete: the decisions are provisional'
    (out / 'decisions.json').write_text(json.dumps(dec, ensure_ascii=False, indent=2, default=float) + '\n',
                                        encoding='utf-8')
    (out / 'RESULTS_PHASE6.md').write_text(results_text(dec, comp, abl_summary, tt, ft), encoding='utf-8')

    P.step('manifest with the fingerprints of every output')
    files = sorted(p for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json')
    manifest = {'date': dt.date.today().isoformat(), 'protocol_sha256': proto, 'data_fingerprint': data.fingerprint,
                'python': platform.python_version(), 'workdir_tasks': {'folds': len(tasks), 'temporal': len(temporal),
                                                                        'final': len(final)},
                'files': {str(p.relative_to(out)).replace('\\', '/'): sha16(p) for p in files}}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'\nAnalysis of phase 6 in: {out}')
    if 'mean_difference' in primary:
        print(f"H1 (non-inferiority with weak calibration): {primary['H1']}; H2: {primary['H2']}")
    if not complete:
        print('WARNING: confirmation incomplete; the decisions are provisional')
    return dec
