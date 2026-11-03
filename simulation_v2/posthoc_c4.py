"""Post-hoc diagnosis of criterion C4 (localization), declared as such: written after the evaluation.

C4 was not met (180/200, Wilson lower limit 0.851 < 0.90). This script:

1. recomputes RabMV's checks (undeclared schema) for every replicate of the conditions with differential
   measurement and of ``no_difference``, and verifies that they reproduce the frozen evaluation exactly;
2. lists which check produced each false alert on an honest concept;
3. applies ``rabmv.drop_explained_pairs`` (single-choice pairs explained by an A or C alert) and reports
   detection and localization with and without it.

It trains no model (checks only; about 15 minutes on a laptop). The evaluation result of C4 is not changed by this
analysis. From the repository root, after ``python -m simulation_v2.run evaluation``:

    python -m simulation_v2.posthoc_c4

Writes ``simulation_v2/results/evaluation/posthoc_c4/`` (synthetic aggregates only).
"""
import json
import time

import numpy as np
import pandas as pd

import simulation_v2  # noqa: F401  (memoized normalization)
from rabmv import drop_explained_pairs, harmonize, run_checks
from simulation_v2.generator import condition_params, record, schema, training_population
from simulation_v2.run import EVALUATION_SEEDS, RESULTS, honest
from simulation_v2.summary import wilson

CONDITIONS = ('E_pi100_education', 'E_pi100_occupation', 'E_pi85_education', 'E_pi85_occupation',
              'no_difference')


def _concepts(table):
    out = set()
    for c in table['concept']:
        out.update(x.strip() for x in str(c).split('&'))
    return out


def main():
    src = RESULTS / 'evaluation' / 'per_seed.jsonl'
    stored = {}
    for line in src.read_text(encoding='utf-8').splitlines():
        if line.strip():
            d = json.loads(line)
            stored[(d['condition'], d['seed'], d['events'])] = d['checks_undeclared']
    rows, false_alerts = [], []
    start = time.time()
    tasks = [(c, ev) for c in CONDITIONS for ev in (180, 60) if (c, EVALUATION_SEEDS[0], ev) in stored]
    for n, (cond, ev) in enumerate(tasks, 1):
        p = condition_params(cond)
        gap = p['gap']
        hon = honest(gap)
        for seed in EVALUATION_SEEDS:
            pop = training_population(seed, ev, p)
            data = record(pop, p, seed)
            s = schema(p, 'all_exact')
            checks = run_checks(harmonize(data, s), pop['y'], s, data=data)
            original = _concepts(checks.alerts())
            refined = _concepts(drop_explained_pairs(checks).alerts())
            for _, a in checks.alerts().iterrows():
                for c in _concepts(pd.DataFrame([a])) & hon:
                    false_alerts.append(dict(condition=cond, events=ev, seed=seed, concept=c, check=a['check'],
                                             alert=a['concept']))
            rows.append(dict(condition=cond, events=ev, seed=seed,
                             reproduced=' '.join(sorted(original)) == stored[(cond, seed, ev)],
                             gap_original=(gap in original) if gap else np.nan,
                             gap_refined=(gap in refined) if gap else np.nan,
                             honest_original=len(original & hon), honest_refined=len(refined & hon)))
        print(f'[{n}/{len(tasks)}] {cond}, {ev} events | {time.time() - start:.0f} s', flush=True)
    df = pd.DataFrame(rows)
    out = RESULTS / 'evaluation' / 'posthoc_c4'
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'by_replicate.csv', index=False)
    pd.DataFrame(false_alerts).to_csv(out / 'false_alerts.csv', index=False)
    summary = {'reproduced_all': bool(df['reproduced'].all())}
    for (cond, ev), g in df.groupby(['condition', 'events']):
        key = f'{cond}_{ev}'
        if condition_params(cond)['gap']:
            for v in ('original', 'refined'):
                loc = g[f'gap_{v}'].astype(bool) & (g[f'honest_{v}'] == 0)
                summary[f'{key}_localization_{v}'] = wilson(int(loc.sum()), len(g))
                summary[f'{key}_detection_{v}'] = float(g[f'gap_{v}'].astype(bool).mean())
        else:
            for v in ('original', 'refined'):
                summary[f'{key}_any_false_alert_{v}'] = float((g[f'honest_{v}'] > 0).mean())
    e = df[df['condition'].str.startswith('E_pi100') & (df['events'] == 180)]
    for v in ('original', 'refined'):
        loc = e[f'gap_{v}'].astype(bool) & (e[f'honest_{v}'] == 0)
        summary[f'C4_rule_{v}'] = wilson(int(loc.sum()), len(e))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k.startswith(('C4', 'reproduced'))}, indent=1,
                     default=float))
    print(f'written {out}')


if __name__ == '__main__':
    main()
