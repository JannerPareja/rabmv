"""Summary of simulation v2: Monte Carlo means with their errors, pre-specified criteria, figures and RESULTS.md.

Reads only ``results/<level>/per_seed.jsonl`` (no model is refitted). Monte Carlo standard error of a mean:
SD / √n_sim (Morris, White and Crowther 2019); confidence intervals of a mean difference use the t
distribution with n_sim − 1 degrees of freedom; proportions use the 95 % Wilson interval.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from simulation_v2.generator import CONDITIONS, GAPS, TARGETS
from simulation_v2.methods import METHODS as _METHODS
from simulation_v2.run import RESULTS

METHODS = _METHODS[:-2] + ('rabmv_sensitivity',) + _METHODS[-2:]
METRICS = ('aucpr', 'aucroc', 'brier', 'logloss', 'excess_logloss', 'cal_intercept', 'cal_slope')
DIFF_METRICS = ('excess_logloss', 'aucpr', 'aucroc', 'cal_slope')
C1_COMPARATORS = ('ebm_native', 'rabmv_indicator', 'ebm_indicator', 'trees_native')
C2_PAIRS = (('rabmv', 'rabmv_mice'), ('ebm_marginalize', 'ebm_native'))
C2_MARGIN = 0.01
C3_SENSITIVITY, C3_SPECIFICITY, C4_LOCALIZATION = 0.90, 0.99, 0.90
MAIN = ('rabmv', 'rabmv_indicator', 'rabmv_mice', 'rabmv_pattern', 'ebm_native', 'ebm_marginalize', 'ebm_mice',
        'trees_native', 'drop')
N_CONCEPTS = 16


def wilson(k, n, z=norm.ppf(0.975)):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, c - h, c + h


def mean_ci(x, level):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return dict(mean=float(np.mean(x)) if n else np.nan, mcse=np.nan, lower=np.nan, upper=np.nan, n=n)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    q = t.ppf(0.5 + level / 2, n - 1)
    return dict(mean=float(m), mcse=float(se), lower=float(m - q * se), upper=float(m + q * se), n=n)


def load(level):
    path = RESULTS / level / 'per_seed.jsonl'
    return pd.DataFrame([json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()])


def methods_table(df):
    rows = []
    for (cond, ev), g in df.groupby(['condition', 'events']):
        for tg in TARGETS:
            for m in METHODS:
                for k in METRICS:
                    col = f'{tg}_{m}_{k}'
                    if col in g and g[col].notna().any():
                        rows.append(dict(condition=cond, events=ev, target=tg, method=m, metric=k,
                                         **mean_ci(g[col], 0.95)))
    return pd.DataFrame(rows)


def differences(df, base='rabmv'):
    rows = []
    for (cond, ev), g in df.groupby(['condition', 'events']):
        for tg in TARGETS:
            for m in METHODS:
                if m == base:
                    continue
                for k in DIFF_METRICS:
                    a, b = f'{tg}_{base}_{k}', f'{tg}_{m}_{k}'
                    if a in g and b in g and g[b].notna().any():
                        d = g[a] - g[b]
                        r95, r90 = mean_ci(d, 0.95), mean_ci(d, 0.90)
                        rows.append(dict(condition=cond, events=ev, target=tg, comparator=m, metric=k,
                                         mean=r95['mean'], mcse=r95['mcse'], lower95=r95['lower'],
                                         upper95=r95['upper'], lower90=r90['lower'], upper90=r90['upper'], n=r95['n']))
    return pd.DataFrame(rows)


def shortcut_table(df):
    rows = []
    for (cond, ev), g in df.groupby(['condition', 'events']):
        for m in METHODS:
            r = dict(condition=cond, events=ev, method=m)
            for k in ('shortcut_abs', 'shortcut_aucpr'):
                col = f'T0_{m}_{k}'
                if col in g:
                    s = mean_ci(g[col], 0.95)
                    r[f'{k}_mean'], r[f'{k}_mcse'] = s['mean'], s['mcse']
            col = f'{m}_shape_rmse'
            if col in g:
                s = mean_ci(g[col], 0.95)
                r['shape_rmse_mean'], r['shape_rmse_mcse'] = s['mean'], s['mcse']
            if len(r) > 3:
                rows.append(r)
    return pd.DataFrame(rows)


def audits_table(df):
    rows = []
    for (cond, ev), g in df.groupby(['condition', 'events']):
        n = len(g)
        gap = CONDITIONS[cond]['gap']
        r = dict(condition=cond, events=ev, n=n)
        for v in ('FAIL', 'CAUTION', 'PASS'):
            r[f'shukla_overall_{v}'] = float((g['shukla_overall'] == v).mean())
        for k in ('shukla_A_form', 'shukla_A_year', 'shukla_B', 'shukla_C'):
            r[f'{k}_FAIL'] = float((g[k] == 'FAIL').mean())
        for k in ('shukla_A_form_V', 'shukla_A_year_V', 'shukla_B_gain', 'attr_auroc_iid', 'attr_auroc_year_blocked',
                  'attr_auroc_residualized'):
            r[k] = float(g[k].mean())
        r['attr_auroc_form_blocked'] = float(g['attr_auroc_form_blocked'].mean())
        r['attr_form_blocked_undefined'] = float(g['attr_auroc_form_blocked'].isna().mean())
        for a in ('missing_test', 'checks_undeclared', 'checks_declared'):
            if gap:
                r[f'{a}_detects_gap'] = float(g[f'{a}_gap'].astype(float).mean())
            r[f'{a}_any_honest_alert'] = float((g[f'{a}_honest_alerts'] > 0).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def criteria(df, diff):
    out = {}
    d180 = diff[diff['events'] == 180]
    c1 = {}
    for gap in GAPS:
        for comp in C1_COMPARATORS:
            q = d180[(d180['condition'] == f'E_pi100_{gap}') & (d180['target'] == 'T3') & (d180['comparator'] == comp)
                     & (d180['metric'] == 'excess_logloss')]
            if len(q):
                q = q.iloc[0]
                c1[f'{gap}_vs_{comp}'] = dict(mean=q['mean'], upper95=q['upper95'], met=bool(q['upper95'] < 0))
    out['C1_transport'] = dict(rule='E_pi100, target T3: upper 95 % limit of mean Δ excess log-loss '
                                    '(RabMV − comparator) < 0 against each comparator', results=c1,
                               met=bool(c1) and all(v['met'] for v in c1.values()))
    c2 = {}
    g180 = df[df['events'] == 180]
    for gap in GAPS:
        g = g180[g180['condition'] == f'MAR_{gap}']
        for a, b in C2_PAIRS:
            if len(g) and f'T0_{a}_aucpr' in g:
                r = mean_ci(g[f'T0_{a}_aucpr'] - g[f'T0_{b}_aucpr'], 0.90)
                c2[f'{gap}_{a}_vs_{b}'] = dict(mean=r['mean'], lower90=r['lower'], upper90=r['upper'],
                                               met=bool(r['lower'] > -C2_MARGIN and r['upper'] < C2_MARGIN))
    out['C2_no_cost_of_the_rule'] = dict(rule=f'MAR (blanks made by people), T0: 90 % CI of Δ AUC-PR within '
                                              f'±{C2_MARGIN} for the rule against the same engine without it',
                                         results=c2,
                                         met=bool(c2) and all(v['met'] for v in c2.values()))
    e = g180[g180['condition'].isin([f'E_pi100_{gap}' for gap in GAPS])]
    k = int(e['checks_undeclared_gap'].astype(bool).sum()) if len(e) else 0
    sens = wilson(k, len(e))
    nul = g180[g180['condition'] == 'no_difference']
    alerts = int(nul['checks_undeclared'].fillna('').map(lambda s: len(s.split())).sum()) if len(nul) else 0
    total = len(nul) * N_CONCEPTS
    spec = wilson(total - alerts, total)
    out['C3_detection'] = dict(rule='Wilson lower limit of the sensitivity (E_pi100, undeclared schema) ≥ '
                                    f'{C3_SENSITIVITY}'
                                    f' and of the per-concept specificity (no_difference) ≥ {C3_SPECIFICITY}',
                               sensitivity=sens, specificity=spec,
                               met=bool(len(e) and len(nul) and sens[1] >= C3_SENSITIVITY
                                        and spec[1] >= C3_SPECIFICITY))
    if len(e):
        loc = e['checks_undeclared_gap'].astype(bool) & (e['checks_undeclared_honest_alerts'] == 0)
        lw = wilson(int(loc.sum()), len(e))
    else:
        lw = (np.nan,) * 3
    out['C4_localization'] = dict(rule=f'Wilson lower limit of P(gap concept alerted and no honest concept alerted) ≥ '
                                       f'{C4_LOCALIZATION} (E_pi100, undeclared schema)', estimate=lw,
                                  met=bool(len(e) and lw[1] >= C4_LOCALIZATION))
    out['method_validity'] = all(out[k]['met'] for k in out)
    return out


def figures(level, tab, short, aud):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = RESULTS / level / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    t180 = tab[(tab['events'] == 180) & (tab['metric'] == 'excess_logloss')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, gap in zip(axes, GAPS):
        q = t180[t180['condition'] == f'E_pi100_{gap}']
        for m in MAIN:
            s = q[q['method'] == m].set_index('target').reindex(TARGETS)
            ax.errorbar(range(len(TARGETS)), s['mean'], yerr=1.96 * s['mcse'], marker='o', capsize=3, label=m)
        ax.set_xticks(range(len(TARGETS)), TARGETS)
        ax.set_title(f'Form = class (π = 1), gap in {gap}')
        ax.set_xlabel('Target registry')
        ax.set_yscale('log')
    axes[0].set_ylabel('Excess log-loss over the oracle (log scale)')
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / 'transport.png', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, gap in zip(axes, GAPS):
        conds = [f'E_pi100_{gap}', f'E_pi85_{gap}', f'E_pi50_{gap}', f'MAR_{gap}', f'MNAR_{gap}']
        s = short[(short['events'] == 180) & short['condition'].isin(conds)]
        width = 0.8 / len(MAIN)
        for i, m in enumerate([m for m in MAIN if m != 'drop']):
            v = s[s['method'] == m].set_index('condition').reindex(conds)
            ax.bar(np.arange(len(conds)) + i * width, v['shortcut_aucpr_mean'], width,
                   yerr=1.96 * v['shortcut_aucpr_mcse'], label=m)
        ax.set_xticks(np.arange(len(conds)) + 0.4, [c.replace(f'_{gap}', '') for c in conds])
        ax.axhline(0, color='black', lw=0.8)
        ax.set_title(f'Gap in {gap}')
    axes[0].set_ylabel('AUC-PR gained from the missingness itself (T0)')
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / 'shortcut.png', dpi=200)
    plt.close(fig)

    a = aud[aud['events'] == 180].set_index('condition')
    cols = ['checks_undeclared_detects_gap', 'missing_test_detects_gap', 'shukla_overall_FAIL',
            'checks_undeclared_any_honest_alert', 'missing_test_any_honest_alert']
    labels = ['RabMV checks: gap detected', 'Missingness test: gap detected', 'Shukla et al.: overall FAIL',
              'RabMV checks: false alert', 'Missingness test: false alert']
    order = [c for c in CONDITIONS if c in a.index]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    width = 0.8 / len(cols)
    for i, (c, lab) in enumerate(zip(cols, labels)):
        if c in a:
            ax.bar(np.arange(len(order)) + i * width, a.reindex(order)[c], width, label=lab)
    ax.set_xticks(np.arange(len(order)) + 0.4, order, rotation=45, ha='right')
    ax.set_ylabel('Proportion of replicates')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'detection.png', dpi=200)
    plt.close(fig)


def results_md(level, crit, tab, diff, short, aud):
    lines = [f'# Simulation v2 of RabMV: results ({level} level)', '',
             'Synthetic data only. Pre-specified criteria (simulation_v2/PLAN.md):', '']
    for k, v in crit.items():
        if k != 'method_validity':
            lines.append(f'- **{k}**: {"met" if v["met"] else "NOT met"}. {v["rule"]}.')
    lines += ['', f'Method validity (all criteria): **{crit["method_validity"]}**', '',
              '## Excess log-loss over the oracle, 180 events (mean, MCSE)', '']
    t180 = tab[(tab['events'] == 180) & (tab['metric'] == 'excess_logloss')]
    for cond in CONDITIONS:
        q = t180[t180['condition'] == cond]
        if q.empty:
            continue
        p = q.pivot(index='method', columns='target', values='mean').reindex([m for m in METHODS if m in set(q.method)])
        lines += [f'### {cond}', '', p.round(4).to_markdown(), '']
    lines += ['## Shortcut and shape recovery (T0 and T1)', '',
              short[short['events'] == 180].round(4).to_markdown(index=False),
              '', '## Audits', '', aud.round(3).to_markdown(index=False), '']
    (RESULTS / level / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def summarize(level):
    df = load(level)
    out = RESULTS / level
    tab, diff, short, aud = methods_table(df), differences(df), shortcut_table(df), audits_table(df)
    tab.to_csv(out / 'methods.csv', index=False)
    diff.to_csv(out / 'differences.csv', index=False)
    short.to_csv(out / 'shortcut_shape.csv', index=False)
    aud.to_csv(out / 'audits.csv', index=False)
    crit = criteria(df, diff)
    (out / 'criteria.json').write_text(json.dumps(crit, indent=1, default=float) + '\n', encoding='utf-8')
    figures(level, tab, short, aud)
    results_md(level, crit, tab, diff, short, aud)
    print(json.dumps({k: v['met'] if isinstance(v, dict) else v for k, v in crit.items()}, indent=1))
    print(f'written {out}')
