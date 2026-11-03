"""Development of RabMV with real data (phase 5): execution by levels and summary.

Levels (same code and same design; only how much is run changes):

* ``test``: every combination on fold 0 with 3 evaluations per optimizer. Detects code errors and measures
  times; the summary estimates the duration of the development level on this machine.
* ``development``: complete repetition 1 of the 5×5 (folds 0–4), inner CV of 5 and 24 evaluations per
  optimizer, like the publication profile of Competencia 1. A descriptive development result: it decides
  nothing about the study; repetitions 2–5 are reserved for confirmation.

Phase 6 (confirmation, ``PROTOCOL_PHASE6.md``) adds the levels ``confirmation`` (repetitions 2–5, folds 5–24),
``temporal`` (fit 2016–2020, test 2021–2022) and ``final`` (final models on every case), each with no balancing
and the Bayesian search for the five variants, and the command ``analyze`` (the pre-specified analysis).

Usage (from the root of the rabmv repository, with its environment active)::

    python -m development.run execute --level test --data <datos_modelado.csv>
        --schema <schema folder> --workdir <private folder> --processes 4 [--balancing none weights]
    python -m development.run summary --level test --workdir <private folder>
    python -m development.run report --level test --workdir <private folder> [--schema <schema folder>]
    python -m development.run analyze --workdir <private folder> --data <datos_modelado.csv> --schema <schema folder>
    python -m development.run explore --workdir <private folder> --data <datos_modelado.csv> --schema <schema folder>

The work folder stores checkpoints with per-case probabilities: it must be OUTSIDE any repository. The
summary writes only aggregates to ``development/results/<level>/``.
"""
import argparse
import datetime as dt
import glob
import json
import os
import pickle
import platform
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import execution as X
from . import progress as P

ROOT = Path(__file__).resolve().parent
METRICS = ['auc_pr', 'auc_roc', 'brier', 'ece', 'calib_intercept', 'calib_slope', 'precision', 'recall',
           'f1', 'specificity', 'tpr_gap']
# Paired comparisons (a − b): the method against EBM; the registry against the same matrix; the diagnostic
# splits that difference into education (common − RabMV) and the rest of the registry (registry − common).
COMPARISONS = [('RabMV', 'EBM'), ('RabMV_registry', 'RabMV'), ('RabMV_common_education', 'RabMV'),
               ('RabMV_registry', 'RabMV_common_education'), ('RabMV_registry', 'EBM_registry'),
               ('EBM_registry', 'EBM')]
DIFFERENCES = ['auc_pr', 'auc_roc', 'brier', 'calib_slope', 'precision', 'recall', 'f1']


def _execute(args):
    """Run every task of the level in parallel (each task on one core) and report progress."""
    workdir = Path(args.workdir).resolve()
    if (workdir / '.git').exists() or (ROOT.parent in workdir.parents) or workdir == ROOT.parent:
        sys.exit('ERROR: the work folder cannot be inside the RabMV repository (it stores per-case data)')
    pending = X.tasks(args.level, workdir, args.data, args.schema, variants=args.variants, balancing=args.balancing)
    X.prepare(args.data, args.schema)                         # fails here, before launching processes
    total, t0, errors = len(pending), time.perf_counter(), 0
    print(f'{total} tasks of level «{args.level}» with {args.processes} process(es)', flush=True)
    folder = os.path.join(workdir, args.level, 'progress')
    n, last_status = 0, time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.processes) as ex:
        futures = {ex.submit(X.run_task, t): t for t in pending}
        waiting = set(futures)
        while waiting:
            done, waiting = wait(waiting, timeout=30, return_when=FIRST_COMPLETED)
            for f in done:
                n += 1
                t = futures[f]
                try:
                    r = f.result()
                    status = ('ERROR ' + str(r['error'])[:120] if r.get('error')
                              else f'AUC-PR {r["metrics"]["original"]["auc_pr"]:.4f}')
                    errors += bool(r.get('error'))
                except Exception as e:                       # reported, and the others continue
                    status, errors = f'ERROR {type(e).__name__}: {str(e)[:160]}', errors + 1
                print(f'[{n}/{total}] {(time.perf_counter() - t0) / 60:6.1f} min  {t["variant"]} · {t["balancing"]} · '
                      f'{t["optimizer"]} · fold {t["fold"]}: {status}', flush=True)
            if waiting and time.perf_counter() - last_status >= args.status_every:
                last_status = time.perf_counter()
                print(P.status_line(folder, n, total, (last_status - t0) / 60), flush=True)
    print(f'\nFinished in {(time.perf_counter() - t0) / 60:.1f} min; tasks with errors: {errors}')
    if errors:
        sys.exit(1)


def load_tasks(workdir, level, missing_ok=True):
    """Results of every saved task of a level (an empty list if the level has none and ``missing_ok``)."""
    paths = sorted(glob.glob(os.path.join(workdir, level, 'tasks', '*', 'fold_*.pkl')))
    out = []
    for p in paths:
        with open(p, 'rb') as f:
            r = pickle.load(f)
        if r.get('design_version') != C.DESIGN_VERSION:
            sys.exit(f'{p} belongs to design version {r.get("design_version", 1)}, not {C.DESIGN_VERSION}: '
                     'use the work folder of this version')
        out.append(r)
    if not out and not missing_ok:
        sys.exit(f'There are no saved tasks of level «{level}» in {workdir}')
    return out


def fold_table(res):
    """One row per task with the metrics of the original probabilities (no per-case data)."""
    rows = []
    for r in res:
        base = {k: r.get(k) for k in ('variant', 'balancing', 'optimizer', 'fold', 'evaluations',
                                      'failed_evaluations', 'seconds_optimization', 'seconds_refit',
                                      'best_inner_score', 'threshold', 'error')}
        base['best_params'] = json.dumps(r['best_params']) if r.get('best_params') else None
        base['seconds_per_evaluation'] = (r['seconds_optimization'] / r['evaluations']
                                          if r.get('evaluations') else np.nan)
        m = (r.get('metrics') or {}).get('original') or {}
        base.update({k: m.get(k, np.nan) for k in METRICS})
        rows.append(base)
    order = {v: i for i, v in enumerate(C.combinations())}
    t = pd.DataFrame(rows)
    t['_order'] = [order[(a, b, c)] for a, b, c in zip(t.variant, t.balancing, t.optimizer)]
    return t.sort_values(['_order', 'fold']).drop(columns='_order').reset_index(drop=True)


def combination_summary(folds):
    """Mean and standard deviation across folds per combination."""
    g = folds.groupby(['variant', 'balancing', 'optimizer'], sort=False)
    out = g[METRICS].agg(['mean', 'std'])
    out.columns = [f'{m}_{s}' for m, s in out.columns]
    out['folds'] = g.size()
    out['errors'] = g['error'].apply(lambda s: int(s.notna().sum()))
    out['minutes_per_task'] = g['seconds_optimization'].mean() / 60
    return out.reset_index()


def paired_differences(folds, a, b):
    """Paired differences a − b by (balancing, optimizer, fold), summarized per combination."""
    keys = ['balancing', 'optimizer', 'fold']
    A = folds[folds.variant == a].set_index(keys)[DIFFERENCES]
    B = folds[folds.variant == b].set_index(keys)[DIFFERENCES]
    common = A.index.intersection(B.index)
    if common.empty:
        return pd.DataFrame()
    diff = (A.loc[common] - B.loc[common]).reset_index()
    rows = []
    for (bal, opt), s in diff.groupby(['balancing', 'optimizer'], sort=False):
        row = {'comparison': f'{a} − {b}', 'balancing': bal, 'optimizer': opt, 'folds': len(s)}
        for m in DIFFERENCES:
            row[f'd_{m}_mean'] = s[m].mean()
            row[f'd_{m}_std'] = s[m].std()
        row['folds_a_wins_auc_pr'] = int((s['auc_pr'] > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def estimate_development(folds, processes):
    """Estimated hours of the development level from the times of the test level.

    Per combination: seconds per evaluation × 24 evaluations + refit, times 5 folds; the total is spread over
    the processes. It is an estimate: the genetic and Bayesian searches vary by configuration.
    """
    budget = C.LEVELS['development']['budget']
    n_folds = len(C.LEVELS['development']['folds'])
    per_task = folds['seconds_per_evaluation'] * budget + folds['seconds_refit'].fillna(0)
    return float(per_task.sum() * n_folds / processes / 3600)


def _summary(args):
    """Read the checkpoints of the level and write aggregates only."""
    res = load_tasks(args.workdir, args.level)
    if not res:
        sys.exit(f'There are no saved tasks of level «{args.level}» in {args.workdir}')
    folds = fold_table(res)
    expected = len(X.tasks(args.level, args.workdir, '', ''))
    out = Path(args.results or ROOT / 'results' / args.level)
    out.mkdir(parents=True, exist_ok=True)
    comb = combination_summary(folds)
    diff = pd.concat([paired_differences(folds, a, b) for a, b in COMPARISONS], ignore_index=True)
    folds.to_csv(out / 'metrics_by_fold.csv', index=False, lineterminator='\n')
    comb.to_csv(out / 'combination_summary.csv', index=False, lineterminator='\n')
    diff.to_csv(out / 'paired_differences.csv', index=False, lineterminator='\n')
    fingerprints = sorted({r['data_fingerprint'] for r in res})
    manifest = {'level': args.level, 'date': dt.date.today().isoformat(), 'tasks': len(res),
                'expected_tasks': expected, 'tasks_with_errors': int(folds['error'].notna().sum()),
                'data_fingerprint': fingerprints, 'seed': C.SEED, 'window': list(C.WINDOW),
                'configuration': dict(C.LEVELS[args.level]),
                'python': platform.python_version(), 'single_machine': len({r['machine'] for r in res}) == 1}
    try:
        import imblearn
        import interpret
        import optuna
        import sklearn

        import rabmv
        manifest['versions'] = {'rabmv': rabmv.__version__, 'interpret': interpret.__version__,
                                'scikit-learn': sklearn.__version__, 'imbalanced-learn': imblearn.__version__,
                                'optuna': optuna.__version__, 'numpy': np.__version__, 'pandas': pd.__version__}
    except Exception as e:                                      # the summary does not depend on this
        manifest['versions'] = f'not available: {e}'
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    pd.set_option('display.width', 200)
    print(f'Tasks: {len(res)} of {expected}; with errors: {manifest["tasks_with_errors"]}')
    print(f'Data fingerprint: {", ".join(h[:16].upper() for h in fingerprints)}\n')
    cols = ['variant', 'balancing', 'optimizer', 'auc_pr_mean', 'brier_mean', 'calib_slope_mean',
            'recall_mean', 'f1_mean', 'minutes_per_task', 'errors']
    print(comb[cols].round(4).to_string(index=False))
    if not diff.empty:
        print('\n== Paired differences (mean across folds)')
        print(diff[['comparison', 'balancing', 'optimizer', 'folds', 'd_auc_pr_mean', 'd_brier_mean',
                    'd_recall_mean', 'd_f1_mean', 'folds_a_wins_auc_pr']].round(4).to_string(index=False))
    if args.level == 'test':
        blocks = {'none + weights': folds.balancing.isin(['none', 'weights']),
                  'smote + adasyn': folds.balancing.isin(['smote', 'adasyn'])}
        for p in (1, 2, 4, 8):
            parts = '; '.join(f'{k} {estimate_development(folds[m], p):.1f} h' for k, m in blocks.items() if m.any())
            print(f'\nEstimated duration of the development level with {p} process(es): '
                  f'{estimate_development(folds, p):.1f} h ({parts})')
    print(f'\nAggregates in: {out}')


def _report(args):
    """Figures, equations and tables from the saved checkpoints (no training)."""
    from . import report
    res = load_tasks(args.workdir, args.level)
    if not res:
        sys.exit(f'There are no saved tasks of level «{args.level}» in {args.workdir}')
    out = Path(args.results or ROOT / 'results' / args.level) / 'report'
    labels = report.Labels(args.schema) if args.schema else None
    print(f'Report of {len(res)} task(s) of level «{args.level}» into {out}', flush=True)
    n = report.build(res, fold_table(res), out, labels, progress=True)
    print(f'Report of {len(res)} task(s), {n} model(s) described, in: {out}')


def _analyze(args):
    """Pre-specified analysis of phase 6 (PROTOCOL_PHASE6.md), from the checkpoints, without training."""
    from . import analysis
    out = Path(args.results or ROOT / 'results' / 'phase6')
    analysis.run(args.workdir, args.data, args.schema, out, load_tasks)


def _explore(args):
    """Exploratory analyses of phase 6 (PLAN_PHASE6_v2.md), after «analyze», without training."""
    from . import exploratory
    out = Path(args.results or ROOT / 'results' / 'phase6')
    exploratory.run(args.workdir, args.data, args.schema, out, load_tasks)


def main(argv=None):
    """Entry point: subcommands ``execute``, ``summary`` and ``report``."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    e = sub.add_parser('execute', help='run (or resume) the tasks of the level')
    e.add_argument('--level', choices=list(C.LEVELS), required=True)
    e.add_argument('--data', required=True, help='datos_modelado.csv of etl.py version 2')
    e.add_argument('--schema', required=True, help='folder with catalog.toml and mappings/')
    e.add_argument('--workdir', required=True, help='private folder for the checkpoints (outside the repository)')
    e.add_argument('--processes', type=int, default=1)
    e.add_argument('--variants', nargs='*', choices=C.VARIANTS, default=None)
    e.add_argument('--balancing', nargs='*', choices=C.BALANCING, default=None,
                   help='only these balancing methods (to run the level in blocks)')
    e.add_argument('--status-every', type=int, default=P.STATUS_EVERY,
                   help='seconds between status lines (fits done by each running task); default 600')
    s = sub.add_parser('summary', help='write the aggregates of the level')
    s.add_argument('--level', choices=list(C.LEVELS), required=True)
    s.add_argument('--workdir', required=True)
    s.add_argument('--results', default=None, help='by default development/results/<level>')
    r = sub.add_parser('report', help='figures, equations and tables from the saved checkpoints (no training)')
    r.add_argument('--level', choices=list(C.LEVELS), required=True)
    r.add_argument('--workdir', required=True)
    r.add_argument('--results', default=None, help='by default development/results/<level> (writes report/)')
    r.add_argument('--schema', default=None,
                   help='study schema folder with labels_en.toml: English names in figures, equations and tables')
    a = sub.add_parser('analyze', help='pre-specified analysis of phase 6 (no training)')
    a.add_argument('--workdir', required=True)
    a.add_argument('--data', required=True, help='datos_modelado.csv of etl.py version 2 (fingerprint checked)')
    a.add_argument('--schema', required=True, help='study schema folder (catalog, mappings and labels_en.toml)')
    a.add_argument('--results', default=None, help='by default development/results/phase6')
    x = sub.add_parser('explore', help='exploratory analyses of phase 6 decided after its results (no training)')
    x.add_argument('--workdir', required=True)
    x.add_argument('--data', required=True, help='datos_modelado.csv of etl.py version 2 (fingerprint checked)')
    x.add_argument('--schema', required=True, help='study schema folder (catalog, mappings and labels_en.toml)')
    x.add_argument('--results', default=None, help='folder of the analysis of phase 6; by default development/results/phase6')
    args = ap.parse_args(argv)
    {'execute': _execute, 'summary': _summary, 'report': _report, 'analyze': _analyze,
     'explore': _explore}[args.command](args)


if __name__ == '__main__':
    main()
