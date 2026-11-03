"""Exploratory analyses of phase 6, decided AFTER seeing its results (``PLAN_PHASE6_v2.md``), without training.

Two questions left open by the pre-specified analysis:

1. **Which shortcut explains the rest of the gap between EBM_registry and RabMV_registry?** The pre-specified
   ablation (missing-value scores set to 0) removed 32 % of it. Here, on the saved models of folds 0–24 and the
   temporal fold:

   * ``missing = mean``: EBM_registry with each missing-value score replaced by the mean score of the term over
     the *measured* training cases — exactly RabMV's rule (an unmeasured concept contributes the average of its
     term over the measured cases; RabMV's terms are centred on them, so that average is 0). Setting a score to
     0 or re-centring a whole term only adds a constant to every case and cannot change AUC-PR; replacing the
     missing score by the measured mean can, because it moves only the cases with missing values.
   * ``form-exclusive categories``: categories of a nominal concept that practically only one form produces
     (fewer than ``MIN_OTHER_FORM`` cases of the other class in the whole window, at least ``MIN_COUNT`` of its
     own) are a second way of recognising the form. At scoring time they are set to NaN (not measured): RabMV
     marginalizes them (contribution 0) and, in the ablated EBM, the missing score is neutral as above. Applied to
     both models, so the comparison stays paired.

2. **What difference can these data resolve?** For each metric: SD of the paired differences, corrected standard
   error with the 20 folds and its floor with infinitely many repetitions (√(n_test/n_train) · s), and the
   smallest difference a one-sided 95 % bound could separate from 0 (``t₀.₉₅ · SE``).

Outputs in ``development/results/phase6/exploratory/`` (aggregates only, counts 1–4 shown as «<5»), and
``RESULTS_PHASE6_v2.md`` next to the pre-specified results, with the estimation framing of the plan version 2.
"""
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from . import analysis as A
from . import config as C
from . import data as D

MIN_COUNT = 5                 # cases of its own form for a category to be considered
MIN_OTHER_FORM = 5            # a category with fewer cases than this in the other class is form-exclusive
VARIANTS_ABLATION = ['EBM_registry', 'EBM_registry: missing = 0', 'EBM_registry: missing = mean',
                     'EBM_registry: missing = mean, exclusive = NaN', 'RabMV_registry',
                     'RabMV_registry: exclusive = NaN']


def _ebm(model):
    """The InterpretML EBM inside a prediction pipeline (unwrapping the class-weights wrapper)."""
    est = model.steps[-1][1]
    return getattr(est, 'estimator_', est)


def missing_to_mean(model, X_fit):
    """Copy of a fitted EBM whose missing-value scores follow RabMV's rule.

    Main effect: the missing bin takes the mean contribution of the term over the measured training cases.
    Pair: the missing row and column take 0 (a pair with an unmeasured concept contributes nothing; EBM pairs are
    centred). Other scores and the intercept are unchanged.
    """
    m = copy.deepcopy(model)
    ebm = _ebm(m)
    contrib = ebm.eval_terms(X_fit)
    Xv = np.asarray(X_fit, dtype=float)
    for t, feats in enumerate(ebm.term_features_):
        s = np.array(ebm.term_scores_[t], float, copy=True)
        if len(feats) == 1:
            measured = ~np.isnan(Xv[:, feats[0]])
            s[0] = float(contrib[measured, t].mean()) if measured.any() else 0.0
        else:
            for axis in range(s.ndim):
                idx = [slice(None)] * s.ndim
                idx[axis] = 0
                s[tuple(idx)] = 0.0
        ebm.term_scores_[t] = s
    return m


def exclusive_categories(data, labels):
    """Nominal categories that practically only one form produces (whole window, aggregate counts)."""
    Xr, y = data.X_registry, data.y
    rows, mask = [], {}
    for j in data.categorical:
        col = Xr.columns[j]
        v = Xr[col].to_numpy(float)
        for code in sorted(pd.unique(v[np.isfinite(v)])):
            n0, n1 = int(((v == code) & (y == 0)).sum()), int(((v == code) & (y == 1)).sum())
            exclusive = (n0 >= MIN_COUNT and n1 < MIN_OTHER_FORM) or (n1 >= MIN_COUNT and n0 < MIN_OTHER_FORM)
            label = labels._group(col, code) if labels else str(code)
            rows.append({'concept': col, 'code': int(code), 'category': label,
                         'attempts_form_356': A.suppress(n0), 'completed_form_105': A.suppress(n1),
                         'form_exclusive': exclusive})
            if exclusive:
                mask.setdefault(col, []).append(code)
    return pd.DataFrame(rows), mask


def blank_exclusive(X, mask):
    """Copy of the registry-aware input with the form-exclusive categories set to NaN (not measured)."""
    X = X.copy()
    for col, codes in mask.items():
        X.loc[X[col].isin(codes), col] = np.nan
    return X


def ablation_variants(tasks, temporal, data, mask, progress):
    """AUC-PR and AUC-ROC of the six ablation variants per fold (0–24 and temporal)."""
    X = data.input('EBM_registry')
    Xb = blank_exclusive(X, mask)
    folds = sorted({f for (v, f) in tasks if v == 'EBM_registry'})
    items = [(f, tasks.get(('EBM_registry', f)), tasks.get(('RabMV_registry', f))) for f in folds]
    if 'EBM_registry' in temporal and 'RabMV_registry' in temporal:
        items.append((C.TEMPORAL_FOLD, temporal['EBM_registry'], temporal['RabMV_registry']))
    rows = []
    for k, (f, re, rr) in enumerate(items, 1):
        if re is None or rr is None or re.get('error') or rr.get('error'):
            continue
        te, fit = np.asarray(re['idx_test']), np.asarray(re['idx_fit'])
        y = data.y[te]
        zero, _ = A.ablate_missing(re['model'])
        mean = missing_to_mean(re['model'], X.iloc[fit])
        probs = {'EBM_registry': np.asarray(re['p_test']),
                 'EBM_registry: missing = 0': zero.predict_proba(X.iloc[te])[:, 1],
                 'EBM_registry: missing = mean': mean.predict_proba(X.iloc[te])[:, 1],
                 'EBM_registry: missing = mean, exclusive = NaN': mean.predict_proba(Xb.iloc[te])[:, 1],
                 'RabMV_registry': np.asarray(rr['p_test']),
                 'RabMV_registry: exclusive = NaN': rr['model'].predict_proba(Xb.iloc[te])[:, 1]}
        for v, p in probs.items():
            rows.append({'fold': f, 'variant': v, 'auc_pr': average_precision_score(y, p),
                         'auc_roc': roc_auc_score(y, p)})
        if k % 5 == 0 or k == len(items):
            progress.note(f'ablation variants: {k}/{len(items)} folds re-scored')
    return pd.DataFrame(rows)


def ablation_summary(abl, sizes):
    """Mean per variant (folds 5–24) and paired comparisons of each against RabMV_registry (same rule)."""
    c = abl[abl.fold.isin(A.CONFIRMATION_FOLDS)]
    wide = {m: c.pivot(index='fold', columns='variant', values=m) for m in ('auc_pr', 'auc_roc')}
    gap = wide['auc_pr']['EBM_registry'].mean() - wide['auc_pr']['RabMV_registry'].mean()
    rows = []
    for v in VARIANTS_ABLATION:
        if v not in wide['auc_pr']:
            continue
        ref = 'RabMV_registry: exclusive = NaN' if 'exclusive' in v else 'RabMV_registry'
        row = {'variant': v, 'auc_pr_mean': wide['auc_pr'][v].mean(), 'auc_roc_mean': wide['auc_roc'][v].mean()}
        if v.startswith('EBM'):
            row['share_of_S1_gap_removed'] = ((wide['auc_pr']['EBM_registry'].mean() - wide['auc_pr'][v].mean()) / gap
                                              if gap > 0 else np.nan)
            for m in ('auc_pr', 'auc_roc'):
                d = (wide[m][v] - wide[m][ref]).dropna()
                n_tr, n_te = sizes.loc[d.index, 'n_train'].mean(), sizes.loc[d.index, 'n_test'].mean()
                t = A.corrected_t(d, n_tr, n_te)
                b = A.bayes_correlated(d, n_tr, n_te)
                row.update({f'{m}_minus_{ref}': t['mean'], f'{m}_ci95_low': t['ci95'][0], f'{m}_ci95_high': t['ci95'][1],
                            f'{m}_bayes_p_left': b['p_left'], f'{m}_bayes_p_rope': b['p_rope'],
                            f'{m}_bayes_p_right': b['p_right'], f'{m}_folds_ebm_better': int((d > 0).sum())})
        rows.append(row)
    return pd.DataFrame(rows), float(gap)


def resolution(fm, sizes):
    """Smallest paired difference these data can resolve, per metric and comparison (folds 5–24)."""
    orig = fm[fm.probabilities == 'original']
    rows = []
    for metric in ('auc_pr', 'auc_roc', 'brier'):
        wide = orig.pivot(index='fold', columns='variant', values=metric)
        for code, a, b in [('H1', *A.PRIMARY)] + A.SECONDARY:
            sub = wide.loc[wide.index.isin(A.CONFIRMATION_FOLDS), [a, b]].dropna()
            if len(sub) < 2:
                continue
            d = (sub[a] - sub[b]).to_numpy()
            n = len(d)
            n_tr, n_te = sizes.loc[sub.index, 'n_train'].mean(), sizes.loc[sub.index, 'n_test'].mean()
            s = d.std(ddof=1)
            se = s * np.sqrt(1 / n + n_te / n_tr)
            floor = s * np.sqrt(n_te / n_tr)
            rows.append({'metric': metric, 'code': code, 'comparison': f'{a} − {b}', 'folds': n,
                         'sd_of_differences': s, 'se_corrected': se, 'se_floor_infinite_repetitions': floor,
                         'smallest_separable_difference': stats.t.ppf(0.95, n - 1) * se,
                         'smallest_separable_difference_infinite_repetitions': stats.norm.ppf(0.95) * floor})
    return pd.DataFrame(rows)


def fig_ablation_variants(abl, path):
    """Mean AUC-PR (folds 5–24) of the six ablation variants, with each fold as a point."""
    plt = A._plt()
    c = abl[abl.fold.isin(A.CONFIRMATION_FOLDS)]
    vs = [v for v in VARIANTS_ABLATION if v in set(c.variant)]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colours = ['#c55a11', '#e07b39', '#f4b183', '#f8cbad', '#2e75b6', '#9dc3e6']
    for i, v in enumerate(vs):
        g = c[c.variant == v]
        ax.bar(i, g.auc_pr.mean(), color=colours[i], edgecolor='black', linewidth=0.5)
        ax.scatter(np.full(len(g), i), g.auc_pr, s=8, color='black', zorder=3)
    ax.set_xticks(range(len(vs)))
    ax.set_xticklabels([v.replace(': ', ':\n').replace(', ', ',\n') for v in vs], fontsize=7)
    ax.set_ylabel('AUC-PR (test fold)')
    ax.set_title('Shortcut ablations (exploratory), folds 5–24: mean (bar) and folds (points)', fontsize=9)
    lo = c.auc_pr.min()
    ax.set_ylim(max(0, lo - 0.05), c.auc_pr.max() + 0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_resolution(res, path):
    """Smallest separable AUC-PR and AUC-ROC difference against the number of folds (RabMV − EBM)."""
    plt = A._plt()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    k = np.arange(5, 201)
    for metric, colour in (('auc_pr', '#1f4e79'), ('auc_roc', '#c55a11')):
        r = res[(res.metric == metric) & (res.code == 'H1')]
        if r.empty:
            continue
        r = r.iloc[0]
        ratio = (r.se_floor_infinite_repetitions / r.sd_of_differences) ** 2
        mdd = stats.t.ppf(0.95, k - 1) * r.sd_of_differences * np.sqrt(1 / k + ratio)
        ax.plot(k, mdd, color=colour, label=metric.upper().replace('_', '-'))
        ax.axhline(r.smallest_separable_difference_infinite_repetitions, color=colour, linestyle=':', linewidth=0.8)
        ax.scatter([r.folds], [r.smallest_separable_difference], color=colour, zorder=3)
    ax.axhline(C.NI_MARGIN, color='#c00000', linestyle='--', linewidth=0.8, label='margin 0.01')
    ax.set_xlabel('Number of test folds (repeated 5-fold CV)')
    ax.set_ylabel('Smallest separable difference (one-sided 95 %)')
    ax.set_title('Resolution of the data: RabMV − EBM (points = folds 5–24; dotted = infinite repetitions)',
                 fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _f(x, nd=4):
    return 'NA' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}f}'


def results_v2(dec, comp, summ, gap, res, excl, events):
    """RESULTS_PHASE6_v2.md: estimation framing (plan version 2) with the exploratory ablations."""
    def row(code, metric):
        r = comp[(comp.code == code) & (comp.metric == metric) & (comp.folds_used == '5–24')]
        return None if r.empty else r.iloc[0]

    lines = ['# Results of phase 6, version 2 (estimation framing; PLAN_PHASE6_v2.md)', '',
             f'Generated {dt.datetime.now().isoformat(timespec="seconds")} from the checkpoints and from the outputs of '
             'the pre-specified analysis (`RESULTS_PHASE6.md`, kept unchanged).', '',
             '## RabMV against EBM on the same input (folds 5–24)', '',
             '| Metric | RabMV | EBM | Difference | Corrected 95 % CI | P(RabMV worse) / P(equivalent, ±0.01) / P(RabMV better) |',
             '|---|---|---|---|---|---|']
    for metric, name in (('auc_pr', 'AUC-PR'), ('auc_roc', 'AUC-ROC'), ('brier', 'Brier (lower is better)')):
        r = row('H1', metric)
        if r is None:
            continue
        lines.append(f'| {name} | {_f(r.mean_a)} | {_f(r.mean_b)} | {_f(r.mean_difference)} | '
                     f'[{_f(r.ci95_low)}, {_f(r.ci95_high)}] | {_f(r.bayes_p_left, 3)} / {_f(r.bayes_p_rope, 3)} / '
                     f'{_f(r.bayes_p_right, 3)} |')
    p = dec['primary']
    lines += ['', f"* Calibration of RabMV: mean intercept {_f(p['rabmv_calibration_intercept_mean'], 3)}, mean slope "
              f"{_f(p['rabmv_calibration_slope_mean'], 3)}.",
              '* For Brier the Bayesian columns read «RabMV higher / equivalent / RabMV lower» (lower is better).']
    h = res[(res.metric == 'auc_pr') & (res.code == 'H1')]
    if not h.empty:
        h = h.iloc[0]
        others = res[(res.code == 'H1') & (res.metric != 'auc_pr')]
        resolves = [m.upper().replace('_', '-') for m, v in zip(others.metric, others.smallest_separable_difference)
                    if v <= C.ROPE]
        lines += [f"* Resolution of the data (AUC-PR): SD of the paired differences {_f(h.sd_of_differences)}; with the "
                  f"{int(h.folds)} folds the smallest separable difference is {_f(h.smallest_separable_difference, 3)}, "
                  f"and {_f(h.smallest_separable_difference_infinite_repetitions, 3)} with infinitely many repetitions "
                  f"({events} events). "
                  + (f'Differences of {C.ROPE} in AUC-PR cannot be resolved by these data'
                     if h.smallest_separable_difference_infinite_repetitions > C.ROPE else
                     f'Differences of {C.ROPE} in AUC-PR can be resolved with enough repetitions')
                  + ('; ' + ' and '.join(resolves).replace('BRIER', 'Brier')
                     + (' resolve them' if len(resolves) > 1 else ' resolves them') if resolves else '')
                  + ' (table `resolution.csv`).']
    lines += ['', '## Registry-aware input and the form shortcut (exploratory ablations, folds 5–24)', '',
              f'Gap EBM_registry − RabMV_registry: {_f(gap)} of AUC-PR.', '',
              '| Variant | AUC-PR | AUC-ROC | Share of the gap removed | EBM − RabMV (same rule), AUC-PR [95 % CI] | P(EBM better) |',
              '|---|---|---|---|---|---|']
    for _, r in summ.iterrows():
        if r.variant.startswith('EBM'):
            ref = 'RabMV_registry: exclusive = NaN' if 'exclusive' in r.variant else 'RabMV_registry'
            d = r.get(f'auc_pr_minus_{ref}')
            lines.append(f"| {r.variant} | {_f(r.auc_pr_mean)} | {_f(r.auc_roc_mean)} | {_f(r.share_of_S1_gap_removed, 3)} | "
                         f"{_f(d)} [{_f(r.auc_pr_ci95_low)}, {_f(r.auc_pr_ci95_high)}] | {_f(r.auc_pr_bayes_p_right, 3)} |")
        else:
            lines.append(f'| {r.variant} | {_f(r.auc_pr_mean)} | {_f(r.auc_roc_mean)} | | | |')
    ex = excl[excl.form_exclusive]
    lines += ['', 'Form-exclusive categories (whole window): ' +
              ('; '.join(f'{r.concept} = {r.category} (356: {r.attempts_form_356}, 105: {r.completed_form_105})'
                         for _, r in ex.iterrows()) if not ex.empty else 'none') + '.', '',
              'These ablations were decided after seeing the results of phase 6 (exploratory, declared as such).', '']
    return '\n'.join(lines)


def run(workdir, data_path, schema, out_phase6, load_tasks, check_fingerprint=True):
    """Run the exploratory analyses; needs the outputs of ``analyze`` in ``out_phase6``."""
    from .report import Labels
    P = A.Progress(6)
    out_phase6 = Path(out_phase6)
    out = out_phase6 / 'exploratory'
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    tab = lambda df, name: df.to_csv(out / 'tables' / name, index=False, lineterminator='\n')  # noqa: E731
    if not (out_phase6 / 'decisions.json').exists():
        raise SystemExit(f'run «analyze» first: {out_phase6 / "decisions.json"} does not exist')

    P.step('loading checkpoints, data and the outputs of the pre-specified analysis')
    got = A.collect(workdir, load_tasks)
    tasks, temporal = got['folds'], got['temporal']
    data = D.load(data_path, schema, check_fingerprint=check_fingerprint)
    labels = Labels(schema) if (Path(schema) / 'labels_en.toml').exists() else None
    dec = json.loads((out_phase6 / 'decisions.json').read_text(encoding='utf-8'))
    comp = pd.read_csv(out_phase6 / 'tables' / 'comparisons.csv')
    fm = A.fold_metrics(tasks)
    sizes = fm[fm.probabilities == 'original'].groupby('fold')[['n_train', 'n_test']].mean()

    P.step('form-exclusive categories (aggregate counts, 1–4 suppressed)')
    excl, mask = exclusive_categories(data, labels)
    tab(excl, 'form_exclusive_categories.csv')
    P.note('form-exclusive: ' + ('; '.join(f'{c} {v}' for c, v in mask.items()) if mask else 'none'))

    P.step('ablation variants of EBM_registry and RabMV_registry (re-scoring without training)')
    abl = ablation_variants(tasks, temporal, data, mask, P)
    tab(abl, 'ablation_variants_by_fold.csv')
    summ, gap = ablation_summary(abl, sizes)
    tab(summ, 'ablation_variants_summary.csv')

    P.step('resolution of the data')
    res = resolution(fm, sizes)
    tab(res, 'resolution.csv')

    P.step('figures and RESULTS_PHASE6_v2.md')
    fig_ablation_variants(abl, out / 'figures' / 'ablation_variants.png')
    fig_resolution(res, out / 'figures' / 'resolution.png')
    (out_phase6 / 'RESULTS_PHASE6_v2.md').write_text(results_v2(dec, comp, summ, gap, res, excl, int(data.y.sum())), encoding='utf-8')

    P.step('manifest of the exploratory outputs')
    plan = Path(__file__).resolve().parent / 'PLAN_PHASE6_v2.md'
    files = sorted(p for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json')
    files.append(out_phase6 / 'RESULTS_PHASE6_v2.md')
    manifest = {'date': dt.date.today().isoformat(), 'data_fingerprint': data.fingerprint,
                'plan_v2_sha256': hashlib.sha256(plan.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                if plan.exists() else None,
                'files': {str(p.relative_to(out_phase6)).replace('\\', '/'): A.sha16(p) for p in files}}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'\nExploratory analyses in: {out}\nResults (version 2): {out_phase6 / "RESULTS_PHASE6_v2.md"}')
    return summ
