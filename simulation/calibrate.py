"""Calibration of the audit checks with seeds 1–50 (phase 3, first stage).

Pre-specified rule (written before seeing the results): choose the LARGEST alpha of the grid
{1e-2, 1e-3, 1e-4, 1e-5} that meets both:

1. Scenario S4 (no differential measurement): per-check specificity (A, B and C) ≥ 0.995, with the lower
   limit of the 95 % Wilson interval ≥ 0.99.
2. Scenario S3 (schema that declares everything exact) with 180 events: sensitivity (point estimate) ≥ 0.95
   for A and ≥ 0.90 for B, C and E. With 50 seeds the Wilson limit does not reach those values even without
   failures; the lower limits are required in the evaluation (seeds 101–300).

If no alpha meets 2, the largest one meeting 1 is chosen and the missing sensitivity is reported. Check E
does not depend on alpha.

Usage (from the root of the rabmv repository):
    python -m simulation.calibrate
Writes simulation/results/calibration.csv and calibration_summary.json (synthetic aggregates only).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from rabmv import evaluate_eligibility, harmonize, run_checks
from simulation.generator import population, record, schema, subsample

SEEDS = range(1, 51)
EVENTS = (180, 120, 60)
ALPHAS = (1e-2, 1e-3, 1e-4, 1e-5)
EXPECTED_ELIGIBLE = ['age', 'sex', 'area', 'insurance', 'occupation', 'stratum', 'marital_status',
                     'education', 'f7', 'f8']
HONEST = {'age', 'sex', 'area', 'insurance', 'occupation', 'stratum', 'marital_status', 'f7', 'f8'}
B_PAIRS = {'f3 & f4', 'f3 & f5', 'f4 & f5'}
OUTPUT = Path(__file__).parent / 'results'


def wilson(successes, n, z=norm.ppf(0.975)):
    """95 % Wilson interval of a proportion: (estimate, lower, upper)."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = successes / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, centre - half, centre + half


def alert(row, alpha):
    """Is there an alert at threshold alpha? (the rows come from a run with alpha = 1)."""
    if row['result'] != 'alert':
        return False
    return row['check'] == 'E' or (row['p'] < alpha)


def one_seed(seed, events):
    """Check tables of S3 (with F) and S4, and the S1 eligibility, for one seed."""
    pop = subsample(population(seed), events, seed)
    y = pop['y']
    s_ok, s_ex, s_null = schema('correct'), schema('all_exact'), schema('null')
    d1 = record(pop, 'valledupar', seed)
    eligible = evaluate_eligibility(harmonize(d1, s_ok), y, s_ok.catalog).eligible
    d3 = record(pop, 'F', seed)                           # S3: every type + shifted age (F)
    c3 = run_checks(harmonize(d3, s_ex), y, s_ex, data=d3, alpha=1.0).table
    d4 = record(pop, 'null', seed)
    c4 = run_checks(harmonize(d4, s_null), y, s_null, data=d4, alpha=1.0).table
    return eligible, c3, c4, int(y.sum())


def main():
    """Run the calibration and write the aggregated results."""
    t0 = time.time()
    rows, real_events = [], {}
    count = {}
    for ev in EVENTS:
        for s in SEEDS:
            eligible, c3, c4, n1 = one_seed(s, ev)
            real_events.setdefault(ev, []).append(n1)
            count.setdefault((ev, 'S1_eligibility'), [0, 0])
            count[(ev, 'S1_eligibility')][0] += eligible == EXPECTED_ELIGIBLE
            count[(ev, 'S1_eligibility')][1] += 1
            for alpha in ALPHAS:
                al3 = c3[c3.apply(alert, axis=1, alpha=alpha)]
                al4 = c4[c4.apply(alert, axis=1, alpha=alpha)]
                pairs = {(r.check, r.concept) for r in al3.itertuples()}
                metrics = {
                    'sens_A': (sum(('A', f) in pairs for f in ('f1', 'f2')), 2),
                    'sens_B': (int(any(('B', p) in pairs for p in B_PAIRS)), 1),
                    'sens_C': (int(('C', 'f6') in pairs), 1),
                    'sens_E': (int(('E', 'education') in pairs), 1),
                    'detection_F': (int(any('age' in c for _, c in pairs)), 1),
                }
                evaluable4 = c4[(c4.result != 'not evaluable') & c4.check.isin(['A', 'B', 'C'])]
                false4 = int(al4.check.isin(['A', 'B', 'C']).sum())
                metrics['spec_S4'] = (len(evaluable4) - false4, len(evaluable4))
                n_e = int((c4.check == 'E').sum())
                metrics['spec_S4_E'] = (n_e - int((al4.check == 'E').sum()), n_e)
                hon = c3[(c3.result != 'not evaluable') & c3.apply(
                    lambda r: r.concept in HONEST or (r.check == 'B' and set(r.concept.split(' & ')) <= HONEST),
                    axis=1)]
                al_hon = hon[hon.apply(alert, axis=1, alpha=alpha)]
                metrics['spec_S3_honest'] = (len(hon) - len(al_hon), len(hon))
                metrics['dataset_S4_without_false'] = (int(len(al4) == 0), 1)
                for m, (ex, n) in metrics.items():
                    c = count.setdefault((ev, alpha, m), [0, 0])
                    c[0] += ex
                    c[1] += n
    for key, (ex, n) in count.items():
        ev, alpha, m = (key[0], np.nan, key[1]) if len(key) == 2 else key
        est, low, high = wilson(ex, n)
        rows.append(dict(events=ev, alpha=alpha, metric=m, successes=ex, n=n, estimate=est, wilson_low=low,
                         wilson_high=high))
    table = pd.DataFrame(rows).sort_values(['metric', 'events', 'alpha']).reset_index(drop=True)

    def val(m, alpha, ev=180, col='estimate'):
        return float(table[(table.metric == m) & (table.alpha == alpha) & (table.events == ev)][col].iloc[0])

    meets_spec = [a for a in ALPHAS if val('spec_S4', a) >= 0.995 and val('spec_S4', a, col='wilson_low') >= 0.99]
    meets_all = [a for a in meets_spec if val('sens_A', a) >= 0.95 and all(
        val(f'sens_{t}', a) >= 0.90 for t in 'BCE')]
    chosen = max(meets_all) if meets_all else (max(meets_spec) if meets_spec else None)
    summary = dict(
        rule='largest alpha with S4 specificity ≥ 0.995 (Wilson lower ≥ 0.99) and sensitivity at 180 events '
             '≥ 0.95 (A) and ≥ 0.90 (B, C, E); if none meets the sensitivity, the largest that meets the '
             'specificity',
        chosen_alpha=chosen, meets_sensitivity=bool(meets_all),
        real_events={str(k): dict(mean=float(np.mean(v)), min=int(min(v)), max=int(max(v)))
                     for k, v in real_events.items()},
        seconds=round(time.time() - t0, 1), seeds=[min(SEEDS), max(SEEDS)])
    OUTPUT.mkdir(exist_ok=True)
    table.to_csv(OUTPUT / 'calibration.csv', index=False, lineterminator='\n')
    (OUTPUT / 'calibration_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                     encoding='utf-8')
    return table, summary


if __name__ == '__main__':
    t, r = main()
    print(json.dumps(r, ensure_ascii=False, indent=2))
