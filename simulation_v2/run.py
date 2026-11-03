"""Simulation v2 of RabMV: runner, checkpoints, freezing and summary.

Usage (from the root of the rabmv repository):

    python -m simulation_v2.run test --processes 4          # 3 seeds: finds errors and measures times
    python -m simulation_v2.run freeze                      # writes simulation_v2/FROZEN_PROTOCOL.json
    python -m simulation_v2.run evaluation --processes 4    # every pre-specified task (refuses if not frozen)
    python -m simulation_v2.run summary --level evaluation  # summary, criteria and figures

Every finished task (condition × seed × events) is appended at once to ``results/<level>/per_seed.jsonl``;
a run that stops resumes where it was. Progress (done, total, elapsed time, estimated time left) is printed
after every task. Only synthetic data are used.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RESULTS = ROOT / 'results'
FROZEN = ROOT / 'FROZEN_PROTOCOL.json'
TEST_SEEDS = (101, 102, 103)
EVALUATION_SEEDS = tuple(range(101, 201))            # 100 replicates (see PLAN.md, Monte Carlo)
EVENTS = 180
EVENTS_LOW = 60
LOW_CONDITIONS = ('E_pi100_education', 'E_pi100_occupation')
SAVE_CURVES_SEEDS = 10
FROZEN_FILES = ('simulation_v2/__init__.py', 'simulation_v2/generator.py', 'simulation_v2/methods.py',
                'simulation_v2/audits.py', 'simulation_v2/run.py', 'simulation_v2/summary.py',
                'simulation_v2/calibrate.py', 'simulation_v2/results/calibration/hyperparameters.json',
                'simulation_v2/PLAN.md', 'simulation/generator.py', 'simulation/models.py',
                'src/rabmv/estimator.py', 'src/rabmv/checks.py', 'src/rabmv/eligibility.py',
                'src/rabmv/availability.py', 'src/rabmv/schema/mapping.py', 'src/rabmv/schema/catalog.py')
HONEST_BASE = ('age', 'sex', 'area', 'insurance', 'stratum', 'marital_status', 'f7', 'f8')


def fingerprint(path):
    return hashlib.sha256((REPO / path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()[:16].upper()


def tasks(level):
    from simulation_v2.generator import CONDITIONS
    seeds = TEST_SEEDS if level == 'test' else EVALUATION_SEEDS
    out = [(c, s, EVENTS) for s in seeds for c in CONDITIONS]
    out += [(c, s, EVENTS_LOW) for s in seeds for c in LOW_CONDITIONS]
    return out


def honest(gap):
    other = {'education': 'occupation', 'occupation': 'education'}.get(gap)
    return set(HONEST_BASE) | ({other} if other else {'education', 'occupation'})


def one_task(args):
    """Every method, audit and target registry for one condition, seed and number of events."""
    condition, seed, events = args
    import pandas as pd

    from rabmv import evaluate_eligibility, harmonize
    from simulation.models import naive_matrix
    from simulation_v2 import audits
    from simulation_v2.generator import (
        TARGETS,
        condition_params,
        record,
        schema,
        target_params,
        target_population,
        training_population,
    )
    from simulation_v2.methods import METHODS, Engine, Fitted, metrics

    t0 = time.time()
    p = condition_params(condition)
    gap = p['gap']
    pop = training_population(seed, events, p)
    y = pop['y'].to_numpy()
    data = record(pop, p, seed)
    s_ok, s_ex = schema(p), schema(p, 'all_exact')
    h = harmonize(data, s_ok)
    elig = evaluate_eligibility(h, pd.Series(y), s_ok.catalog)
    m = elig.matrix(h)
    X_naive = naive_matrix(harmonize(data, s_ex))
    r = {'condition': condition, 'seed': seed, 'events': events, 'n_train': int(len(y)), 'events_train': int(y.sum()),
         'eligible': ' '.join(m.X.columns), 'gap_eligible': bool(gap in m.X.columns) if gap else False,
         'train_missing_gap': float(m.X[gap].isna().mean()) if gap in m.X.columns else np.nan}

    # audits on the training registry
    r.update(audits.shukla(m, y, data, seed))
    flagged = audits.missingness_test(m, y)
    und = audits.rabmv_checks(data, y, s_ex)
    dec = audits.rabmv_checks(data, y, s_ok)
    hon = honest(gap)
    r['missing_test_flagged'] = ' '.join(flagged)
    r['checks_undeclared'] = ' '.join(und)
    r['checks_declared'] = ' '.join(dec)
    for name, fl in (('missing_test', flagged), ('checks_undeclared', und), ('checks_declared', dec)):
        r[f'{name}_gap'] = bool(gap in fl) if gap else np.nan
        r[f'{name}_honest_alerts'] = int(len(set(fl) & hon))
    r['time_audits'] = time.time() - t0

    t1 = time.time()
    fitted = Fitted(m, X_naive, y, gap)
    # descriptive: the 'sensitivity' eligibility mode (admits partially comparable concepts), marginalized
    elig_s = evaluate_eligibility(h, pd.Series(y), s_ok.catalog, mode='sensitivity')
    m_s = elig_s.matrix(h)
    sens = Engine('rabmv', m_s.X.columns, m_s.types).fit(m_s.X, y)
    r['eligible_sensitivity'] = ' '.join(m_s.X.columns)
    r['time_fit'] = time.time() - t1

    t2 = time.time()
    curves = {}
    targets = TARGETS if gap else ('T0',)
    for t in targets:
        tp = target_params(p, t)
        tpop = target_population(seed, tp)
        yt = tpop['y'].to_numpy()
        td = record(tpop, tp, seed + 7_000)
        h_t = harmonize(td, schema(tp))
        mt = elig.matrix(h_t)
        Xn = naive_matrix(harmonize(td, schema(tp, 'all_exact')))
        for c in X_naive.columns:
            if c not in Xn.columns:
                Xn[c] = -1.0
        p_or = 1 / (1 + np.exp(-tpop['logit'].to_numpy()))
        r[f'{t}_prevalence'] = float(yt.mean())
        r[f'{t}_missing_gap'] = float(mt.X[gap].isna().mean()) if gap in mt.X.columns else np.nan
        for k, v in metrics(yt, sens.predict_proba(elig_s.matrix(h_t).X), p_or).items():
            r[f'{t}_rabmv_sensitivity_{k}'] = v
        for meth in METHODS:
            pr = fitted.predict(meth, mt.X, Xn, tpop['logit'].to_numpy())
            if pr is None:
                continue
            for k, v in metrics(yt, pr, p_or).items():
                r[f'{t}_{meth}_{k}'] = v
            if t == 'T0':
                a, b = fitted.shortcut(meth, mt.X, yt, seed)
                r[f'T0_{meth}_shortcut_abs'], r[f'T0_{meth}_shortcut_aucpr'] = a, b
            if t == 'T1' and gap in mt.X.columns:
                if gap == 'education':
                    true = -0.04 * tpop['education'].to_numpy()
                else:
                    true = 0.8 * tpop['occupation'].to_numpy()
                code = mt.X[gap].to_numpy()
                ok = ~np.isnan(code)
                effect = pd.Series(true[ok]).groupby(code[ok]).mean().to_dict()
                rmse, codes, est, tru = fitted.shape(meth, mt.X, effect, seed)
                r[f'{meth}_shape_rmse'] = rmse
                if seed < TEST_SEEDS[0] + SAVE_CURVES_SEEDS and codes is not None:
                    curves[meth] = {'codes': codes, 'estimate': est, 'truth': tru}
    r['time_targets'] = time.time() - t2
    r['time_total'] = time.time() - t0
    return r, curves


def _done(path):
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            d = json.loads(line)
            out.add((d['condition'], d['seed'], d['events']))
    return out


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return [_clean(x) for x in v.tolist()]
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v


def check_frozen():
    if not FROZEN.exists():
        raise SystemExit('simulation_v2 is not frozen: run «python -m simulation_v2.run freeze» first')
    frozen = json.loads(FROZEN.read_text(encoding='utf-8'))['fingerprints']
    changed = [f for f, h in frozen.items() if fingerprint(f) != h]
    if changed:
        raise SystemExit(f'frozen files changed, the evaluation does not run: {changed}')


def run(level, processes):
    if level == 'evaluation':
        check_frozen()
    out = RESULTS / level
    out.mkdir(parents=True, exist_ok=True)
    path, curves_path = out / 'per_seed.jsonl', out / 'curves.jsonl'
    done = _done(path)
    todo = [t for t in tasks(level) if t not in done]
    total = len(tasks(level))
    print(f'simulation v2, level {level}: {total} tasks, {len(done)} already done, {len(todo)} to run, '
          f'{processes} processes', flush=True)
    start = time.time()
    finished = 0
    with Pool(processes) as pool, open(path, 'a', encoding='utf-8') as f, open(curves_path, 'a',
                                                                               encoding='utf-8') as fc:
        for r, curves in pool.imap_unordered(one_task, todo):
            f.write(json.dumps({k: _clean(v) for k, v in r.items()}) + '\n')
            f.flush()
            if curves:
                fc.write(json.dumps({'condition': r['condition'], 'seed': r['seed'], 'events': r['events'],
                                     'curves': {k: {kk: _clean(vv) for kk, vv in v.items()}
                                                for k, v in curves.items()}}) + '\n')
                fc.flush()
            finished += 1
            el = time.time() - start
            left = el / finished * (len(todo) - finished)
            print(f'[{len(done) + finished}/{total}] {r["condition"]} seed {r["seed"]} events {r["events"]}: '
                  f'{r["time_total"]:.0f} s | elapsed {el / 3600:.2f} h, left about {left / 3600:.2f} h', flush=True)
    print('done', flush=True)


def freeze():
    from datetime import date
    data = {'frozen': date.today().isoformat(), 'evaluation_seeds': [EVALUATION_SEEDS[0], EVALUATION_SEEDS[-1]],
            'events': [EVENTS, EVENTS_LOW], 'fingerprints': {f: fingerprint(f) for f in FROZEN_FILES}}
    FROZEN.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'frozen: {FROZEN}')


def main():
    ap = argparse.ArgumentParser(prog='python -m simulation_v2.run')
    ap.add_argument('command', choices=['test', 'evaluation', 'freeze', 'summary'])
    ap.add_argument('--processes', type=int, default=2)
    ap.add_argument('--level', default='evaluation', choices=['test', 'evaluation'])
    a = ap.parse_args()
    if a.command == 'freeze':
        freeze()
    elif a.command == 'summary':
        from simulation_v2.summary import summarize
        summarize(a.level)
    else:
        run(a.command, a.processes)


if __name__ == '__main__':
    main()
