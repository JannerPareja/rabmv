"""Evaluation of the simulation with seeds 101–300 (phase 3, second stage).

It only runs if the code matches the fingerprints of ``simulation/FROZEN_PROTOCOL.json``: the generator, the
scenarios, the method and the criteria cannot be touched after freezing them. Each seed is saved when it
finishes, so an interrupted run resumes where it stopped.

Pre-specified criteria (design document, section «Simulation design»), with 180 events:

* Detection A: lower Wilson limit of the sensitivity ≥ 0.95. B, C and E: ≥ 0.90.
* False alarms (S4): per-check specificity with lower Wilson limit ≥ 0.99.
* S5a, main concept (occupation) absent in the whole secretariat: 90 % CI of the mean AUC-PR difference
  between marginalizing and retraining within ±0.01 (equivalence).
* S5b, occupation absent in 2016–2017: lower limit of the 95 % CI of the AUC-PR difference > −0.01 against
  retraining and against imputing the mode (non-inferiority).
* The global calibration (mean predicted − prevalence) of S5 is reported.
* S2 (transport) and the 120- and 60-event levels: descriptive.
All results are reported with their Monte Carlo error.

Usage (from the root of the rabmv repository):
    python -m simulation.evaluate audit
    python -m simulation.evaluate models --processes 2
    python -m simulation.evaluate summary
"""
import argparse
import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).parent / 'results'
PROTOCOL = Path(__file__).parent / 'FROZEN_PROTOCOL.json'
SEEDS = range(101, 301)
EVENTS = (180, 120, 60)
ALPHA = 0.01
FROZEN_FILES = ['simulation/generator.py', 'simulation/models.py', 'simulation/calibrate.py',
                'simulation/evaluate.py', 'src/rabmv/estimator.py', 'src/rabmv/checks.py',
                'src/rabmv/eligibility.py', 'src/rabmv/availability.py', 'src/rabmv/schema/mapping.py',
                'src/rabmv/schema/catalog.py', 'src/rabmv/core/segmentation.py', 'src/rabmv/core/binning.py',
                'src/rabmv/core/numeric.py', 'src/rabmv/core/smoothing.py']


def fingerprints():
    """SHA-256 fingerprint (16 characters) of each frozen file."""
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()[:16].upper() for f in FROZEN_FILES}


def check_frozen():
    """Stop if any file changed since the protocol was frozen."""
    if not PROTOCOL.exists():
        raise SystemExit('simulation/FROZEN_PROTOCOL.json does not exist: freeze the protocol before evaluating.')
    expected = json.loads(PROTOCOL.read_text(encoding='utf-8'))['fingerprints']
    current = fingerprints()
    different = [f for f in expected if expected[f] != current.get(f)]
    if different:
        raise SystemExit(f'The code changed since it was frozen; not evaluating. Files: {different}')


def _done(path):
    """(seed, events) pairs already saved in a per-seed results CSV."""
    if not path.exists():
        return set()
    t = pd.read_csv(path, usecols=['seed', 'events'])
    return set(zip(t.seed, t.events))


def _append(path, rows):
    """Append rows to a CSV (with header only if the file is new)."""
    pd.DataFrame(rows).to_csv(path, mode='a', header=not path.exists(), index=False, lineterminator='\n')


def _sort_by_seed(path):
    """Rewrite a per-seed CSV sorted by (events, seed): the file no longer depends on the number of processes.

    The values are read and written back as text, so no digit changes (parsing and re-printing floats could
    alter the last digit).
    """
    t = pd.read_csv(path, dtype=str, keep_default_na=False)
    order = sorted(range(len(t)), key=lambda i: (-int(t['events'].iat[i]), int(t['seed'].iat[i])))
    t.iloc[order].to_csv(path, index=False, lineterminator='\n')


def run_audit():
    """S1, S3 and S4 per seed and event level (fast)."""
    from simulation.calibrate import B_PAIRS, EXPECTED_ELIGIBLE, HONEST, alert, one_seed
    path = OUTPUT / 'evaluation_audit_by_seed.csv'
    done = _done(path)
    for ev in EVENTS:
        for s in SEEDS:
            if (s, ev) in done:
                continue
            eligible, c3, c4, n1 = one_seed(s, ev)
            al3 = {(r.check, r.concept) for r in c3[c3.apply(alert, axis=1, alpha=ALPHA)].itertuples()}
            ev4 = c4[(c4.result != 'not evaluable') & c4.check.isin(['A', 'B', 'C'])]
            false4 = int(ev4.apply(alert, axis=1, alpha=ALPHA).sum())
            hon = c3[(c3.result != 'not evaluable') & c3.apply(
                lambda r: r.concept in HONEST or (r.check == 'B' and set(r.concept.split(' & ')) <= HONEST),
                axis=1)]
            _append(path, [dict(
                seed=s, events=ev, completed=n1, s1_correct=int(eligible == EXPECTED_ELIGIBLE),
                det_A_f1=int(('A', 'f1') in al3), det_A_f2=int(('A', 'f2') in al3),
                det_B=int(any(('B', p) in al3 for p in B_PAIRS)), det_C=int(('C', 'f6') in al3),
                det_E=int(('E', 'education') in al3), det_F=int(any('age' in c for _, c in al3)),
                s4_checks=len(ev4), s4_false=false4,
                s3_honest=len(hon), s3_honest_false=int(hon.apply(alert, axis=1, alpha=ALPHA).sum()))])


def _model(args):
    """One seed of S2 and S5 (in a separate process)."""
    from simulation.models import one_seed
    return one_seed(*args)


def run_models(processes=2):
    """S2 and S5 per seed and event level (slow: about 30 s per seed and level)."""
    path = OUTPUT / 'evaluation_models_by_seed.csv'
    done = _done(path)
    tasks = [(s, ev) for ev in EVENTS for s in SEEDS if (s, ev) not in done]
    with Pool(processes) as pool:
        for r in pool.imap_unordered(_model, tasks):
            _append(path, [r])


def wilson(successes, n, z=1.959963984540054):
    """95 % Wilson interval."""
    p = successes / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    m = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, c - m, c + m


def mean_ci(x, level):
    """Mean, Monte Carlo error and t interval of a difference between independent replicates."""
    x = np.asarray(x, float)
    n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    q = student.ppf(1 - (1 - level) / 2, n - 1)
    return float(x.mean()), float(se), float(x.mean() - q * se), float(x.mean() + q * se)


def summary():
    """Design criteria and descriptives, from the per-seed results."""
    for name in ('evaluation_audit_by_seed.csv', 'evaluation_models_by_seed.csv'):
        if (OUTPUT / name).exists():
            _sort_by_seed(OUTPUT / name)
    aud = pd.read_csv(OUTPUT / 'evaluation_audit_by_seed.csv')
    mod = pd.read_csv(OUTPUT / 'evaluation_models_by_seed.csv')
    out = {'alpha': ALPHA, 'seeds': [min(SEEDS), max(SEEDS)], 'by_events': {}}
    for ev in EVENTS:
        a = aud[aud.events == ev]
        m = mod[mod.events == ev]
        r = {'audit_replicates': len(a), 'model_replicates': len(m)}
        det = {'A': (int(a.det_A_f1.sum() + a.det_A_f2.sum()), 2 * len(a)), 'B': (int(a.det_B.sum()), len(a)),
               'C': (int(a.det_C.sum()), len(a)), 'E': (int(a.det_E.sum()), len(a)), 'F': (int(a.det_F.sum()), len(a))}
        for k, (e, n) in det.items():
            p, lo, hi = wilson(e, n)
            r[f'sensitivity_{k}'] = dict(estimate=float(p), wilson_low=float(lo), wilson_high=float(hi), n=n)
        n4, f4 = int(a.s4_checks.sum()), int(a.s4_false.sum())
        p, lo, hi = wilson(n4 - f4, n4)
        r['specificity_S4'] = dict(estimate=float(p), wilson_low=float(lo), wilson_high=float(hi), n=n4, false=f4)
        nh, fh = int(a.s3_honest.sum()), int(a.s3_honest_false.sum())
        r['specificity_S3_honest'] = dict(estimate=(nh - fh) / nh, n=nh, false=fh)
        r['s1_correct_eligibility'] = float(a.s1_correct.mean())
        if len(m) > 1:
            for mo in ('strict', 'naive', 'trees'):
                mean, se, lo, hi = mean_ci(m[f'drop_{mo}'], 0.95)
                r[f's2_drop_{mo}'] = dict(mean=mean, mc_se=se, ci95=[lo, hi],
                                          aucpr_valledupar=float(m[f'aucpr_{mo}_valledupar'].mean()),
                                          aucpr_transport=float(m[f'aucpr_{mo}_transport'].mean()))
            for c in ('occupation', 'education'):
                mean, se, lo, hi = mean_ci(m[f'diff_s5a_{c}_marg_vs_retrain'], 0.90)
                r[f's5a_{c}_marg_vs_retrain'] = dict(mean=mean, mc_se=se, ci90=[lo, hi],
                                                     equivalent=bool(lo > -0.01 and hi < 0.01))
                for other in ('retrain', 'mode'):
                    mean, se, lo, hi = mean_ci(m[f'diff_s5b_{c}_marg_vs_{other}'], 0.95)
                    r[f's5b_{c}_marg_vs_{other}'] = dict(mean=mean, mc_se=se, ci95=[lo, hi],
                                                         non_inferior=bool(lo > -0.01))
                for v in ('a', 'b'):
                    r[f's5{v}_{c}_global_calibration'] = {
                        n: float(m[f'calib_s5{v}_{c}_{n}'].mean()) for n in ('marginalize', 'retrain', 'mode')}
        out['by_events'][str(ev)] = r
    r180 = out['by_events']['180']
    criteria = {
        'detection_A': bool(r180['sensitivity_A']['wilson_low'] >= 0.95),
        'detection_B': bool(r180['sensitivity_B']['wilson_low'] >= 0.90),
        'detection_C': bool(r180['sensitivity_C']['wilson_low'] >= 0.90),
        'detection_E': bool(r180['sensitivity_E']['wilson_low'] >= 0.90),
        'specificity_S4': bool(r180['specificity_S4']['wilson_low'] >= 0.99),
    }
    if 's5a_occupation_marg_vs_retrain' in r180:
        criteria['s5a_equivalence_occupation'] = r180['s5a_occupation_marg_vs_retrain']['equivalent']
        criteria['s5b_non_inferior_retrain_occupation'] = r180['s5b_occupation_marg_vs_retrain']['non_inferior']
        criteria['s5b_non_inferior_mode_occupation'] = r180['s5b_occupation_marg_vs_mode']['non_inferior']
    out['criteria_180_events'] = criteria
    out['method_validity'] = all(criteria.values())
    (OUTPUT / 'evaluation_summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('part', choices=['audit', 'models', 'summary', 'fingerprints'])
    ap.add_argument('--processes', type=int, default=2)
    a = ap.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    if a.part == 'fingerprints':
        print(json.dumps(fingerprints(), indent=2))
    else:
        check_frozen()
        OUTPUT.mkdir(exist_ok=True)
        if a.part == 'audit':
            run_audit()
        elif a.part == 'models':
            run_models(a.processes)
        print(json.dumps(summary() if a.part == 'summary' else {'done': a.part}, ensure_ascii=False, indent=2))
