"""Calibration of the fixed hyperparameters of each engine (simulation v2, before the evaluation).

Pre-specified rule (written before running it): with seeds 1–20 (never used in the evaluation, which uses
101–200) in the no-difference scenario (no differential measurement, nothing missing), each engine takes the
combination of its grid with the lowest mean excess log-loss over the oracle in a new registry of 10,000
events. The engines are thus tuned for the population, never for the missingness, and the same
hyperparameters are then used by every strategy of that engine. Ties (difference < 1e-4) go to the first
combination of the grid.

Usage (from the root of the rabmv repository):
    python -m simulation_v2.calibrate
Writes simulation_v2/results/calibration/grid.csv and hyperparameters.json (synthetic aggregates only).
"""
import json
import time

import numpy as np
import pandas as pd

from simulation_v2.methods import CALIBRATION_FILE, Engine, metrics

SEEDS = range(1, 21)
# 12 combinations per engine: bins × minimum leaf × smoothing (RabMV's final Whittaker smoothing κ; EBM's
# smoothing rounds, 75 by default in InterpretML 0.7.8).
GRID = {'rabmv': [dict(learning_rate=0.05, max_leaves=3, min_samples_leaf=m, max_bins=b, smoothing=k)
                  for b in (32, 128) for m in (10, 60) for k in (0.0, 2.0, 10.0)],
        'ebm': [dict(learning_rate=0.05, min_samples_leaf=m, max_bins=b, smoothing_rounds=k)
                for b in (32, 1024) for m in (2, 30) for k in (0, 75, 200)]}
TIE = 1e-4


def main():
    from rabmv import evaluate_eligibility, harmonize
    from simulation_v2.generator import condition_params, record, schema, target_population, training_population

    p = condition_params('no_difference')
    rows, start = [], time.time()
    for i, seed in enumerate(SEEDS, 1):
        pop = training_population(seed, 180, p)
        y = pop['y'].to_numpy()
        d = record(pop, p, seed)
        s = schema(p)
        h = harmonize(d, s)
        m = evaluate_eligibility(h, pd.Series(y), s.catalog).matrix(h)
        tp = target_population(seed, p)
        mt = evaluate_eligibility(h, pd.Series(y), s.catalog).matrix(harmonize(record(tp, p, seed + 7_000), s))
        yt, p_or = tp['y'].to_numpy(), 1 / (1 + np.exp(-tp['logit'].to_numpy()))
        for eng, grid in GRID.items():
            for k, hyper in enumerate(grid):
                model = Engine(eng, m.X.columns, m.types, hyper).fit(m.X, y)
                rows.append(dict(engine=eng, combination=k, seed=seed, **hyper,
                                 **metrics(yt, model.predict_proba(mt.X), p_or)))
        print(f'[{i}/{len(SEEDS)}] seed {seed} | elapsed {time.time() - start:.0f} s', flush=True)
    df = pd.DataFrame(rows)
    out = CALIBRATION_FILE.parent
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'grid.csv', index=False)
    chosen, table = {}, {}
    for eng, grid in GRID.items():
        g = df[df['engine'] == eng].groupby('combination')[['excess_logloss', 'aucpr', 'cal_slope']].mean()
        best = g['excess_logloss'].min()
        k = int(g.index[g['excess_logloss'] <= best + TIE][0])
        chosen[eng] = grid[k]
        table[eng] = g.round(5).reset_index().to_dict(orient='records')
        print(eng, '\n', g.round(4).to_string(), '\nchosen:', grid[k], flush=True)
    CALIBRATION_FILE.write_text(json.dumps({'rule': __doc__.split('\n\n')[1].replace('\n', ' '), 'seeds': [1, 20],
                                            'chosen': chosen, 'grid_means': table}, indent=1) + '\n',
                                encoding='utf-8')
    print(f'written {CALIBRATION_FILE}')


if __name__ == '__main__':
    main()
