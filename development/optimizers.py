"""The four hyperparameter optimization methods, with a common objective function.

English translation of c1/optimizadores.py of Competencia 1. The only change: the space and the pipeline
come from the development «variant» (development.models). The algorithms, the score, the genetic cache and
their parameters are the same (the unused first definition of the genetic fitness function, which was
redefined before use, was removed).

The four use the SAME score (``SCORE``: AUC-PR computed with ``predict_proba``) and the SAME inner CV. The
complete pipeline (ColumnTransformer → balancing → model) is fitted from scratch in each inner fold, so
preprocessing and oversampling only see the training data of that fold (no leakage). The four have the same
evaluation budget, run on a single core (n_jobs=1) and record every evaluation (configuration, score per
fold, time).

- grid    : ``sklearn.model_selection.GridSearchCV`` over the grid of the space (refit=False; the refit is
            done by execution.py on X_fit).
- random  : ``sklearn.model_selection.RandomizedSearchCV`` over the continuous space (log-uniform on the
            scales), with the study seed.
- bayes   : Optuna with TPE (Tree-structured Parzen Estimator) as surrogate. TPE models p(x | y good) = l(x)
            and p(x | y bad) = g(x) and chooses the candidate that maximizes l(x)/g(x), proportional to the
            Expected Improvement (EI). The first evaluations are random (exploration).
- genetic : DEAP. Genes in [0, 1] (one per hyperparameter), tournament selection, BLX-α crossover, Gaussian
            mutation, elitism; the population diversity per generation is recorded.
"""
import json
import random
import time
import traceback
import warnings

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV

from . import config as C
from . import models as M
from . import progress as P

BAYES_INITIAL = 8            # minimum initial TPE evaluations for budgets > 24
BAYES_EI_CANDIDATES = 24     # candidates sampled from l(x) at each TPE step

# Common score of the four optimizers: AUC-PR on predict_proba[:, 1].
_AUC_PR = make_scorer(average_precision_score, response_method='predict_proba')


def SCORE(estimator, X, y):
    """AUC-PR of a fitted inner-CV model (the scorer above); each call also counts one fit for the visible
    progress of the run (``development.progress``). The value is exactly that of ``_AUC_PR``."""
    value = _AUC_PR(estimator, X, y)
    P.tick()
    return value


# ----------------------------------------------------------------------------- common objective
def evaluate(base, params, X, y, inner_cv):
    """Mean inner-CV AUC-PR of a configuration.

    Returns
    -------
    dict
        ``score`` (mean; NaN on failure), ``scores`` (per fold), ``seconds``, ``convergence_warnings`` and
        ``error`` (text or None).
    """
    t0 = time.perf_counter()
    scores, warn = [], 0
    try:
        for tr, va in inner_cv.split(X, y):
            model = clone(base).set_params(**params)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(X.iloc[tr], y[tr])
            warn += sum(issubclass(a.category, ConvergenceWarning) for a in w)
            scores.append(float(SCORE(model, X.iloc[va], y[va])))
        return {'score': float(np.mean(scores)), 'scores': scores,
                'seconds': time.perf_counter() - t0, 'convergence_warnings': warn, 'error': None}
    except Exception as e:                                          # recorded, not hidden
        return {'score': np.nan, 'scores': scores, 'seconds': time.perf_counter() - t0,
                'convergence_warnings': warn,
                'error': f'{type(e).__name__}: {e}'[:500] + ' | ' + traceback.format_exc(limit=2)[-300:]}


class _Log:
    """Accumulates the evaluation history of an optimizer."""

    def __init__(self, model, balancing, budget):
        """Store the identifiers and the budget (for the record only)."""
        self.model, self.balancing, self.budget = model, balancing, budget
        self.rows = []
        self.best = (-np.inf, None)

    def add(self, params_without_prefix, r, extra=None):
        """Record an evaluation and update the best configuration.

        Returns
        -------
        float
            The score, or −inf if the evaluation failed (so the optimizer discards it).
        """
        n = len(self.rows) + 1
        valid = np.isfinite(r['score'])
        if valid and r['score'] > self.best[0]:
            self.best = (r['score'], dict(params_without_prefix))
        row = {'evaluation': n, 'params': json.dumps(_serializable(params_without_prefix)),
               'score': r['score'], 'inner_scores': json.dumps(r['scores']),
               'seconds': r['seconds'], 'convergence_warnings': r['convergence_warnings'],
               'error': r['error'], 'best_so_far': self.best[0] if np.isfinite(self.best[0]) else np.nan}
        if extra:
            row.update(extra)
        self.rows.append(row)
        return r['score'] if valid else -np.inf


def _serializable(d):
    """Convert tuples and numpy scalars to native types to store them as JSON."""
    return {k: (list(v) if isinstance(v, tuple) else (v.item() if hasattr(v, 'item') else v))
            for k, v in d.items()}


# ----------------------------------------------------------------------------- spaces
class _IntLog:
    """Log-uniform integer distribution for ParameterSampler."""

    def __init__(self, a, b):
        """Range limits (included)."""
        self.a, self.b = a, b

    def rvs(self, random_state=None):
        """Sample a log-uniform integer in [a, b]."""
        from sklearn.utils import check_random_state
        u = check_random_state(random_state).uniform(np.log(self.a), np.log(self.b + 1))
        return int(min(self.b, max(self.a, np.floor(np.exp(u)))))


def _distribution(p):
    """Sampling distribution of RandomizedSearchCV for a hyperparameter."""
    if p.kind == 'cat':
        return list(p.options)
    if p.kind == 'float':
        return stats.loguniform(p.low, p.high) if p.log else stats.uniform(p.low, p.high - p.low)
    return _IntLog(int(p.low), int(p.high)) if p.log else stats.randint(int(p.low), int(p.high) + 1)


def _decode(genes, space):
    """Genes in [0, 1] → hyperparameter values."""
    out = {}
    for g, p in zip(genes, space):
        g = float(min(1.0, max(0.0, g)))
        if p.kind == 'cat':
            out[p.name] = p.options[min(int(g * len(p.options)), len(p.options) - 1)]
            continue
        lo, hi = float(p.low), float(p.high) + (1 if p.kind == 'int' else 0)
        v = np.exp(np.log(lo) + g * (np.log(hi) - np.log(lo))) if p.log else lo + g * (hi - lo)
        out[p.name] = int(min(p.high, max(p.low, np.floor(v)))) if p.kind == 'int' else float(min(p.high, v))
    return out


# ----------------------------------------------------------------------------- methods
def _sklearn_search(searcher, X, y, model, balancing, budget):
    """Run GridSearchCV / RandomizedSearchCV and pass cv_results_ to the common history.

    The order of cv_results_ is the evaluation order (ParameterGrid / ParameterSampler with n_jobs=1), so the
    "best so far" curve is the same as with the other methods.
    """
    log = _Log(model, balancing, budget)
    pre = M.prefix(balancing)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        searcher.fit(X, y)
    warn = sum(issubclass(a.category, ConvergenceWarning) for a in w)
    failures = [str(a.message)[:500] for a in w if a.category.__name__ == 'FitFailedWarning']
    r = searcher.cv_results_
    k = searcher.n_splits_
    for n, cfg in enumerate(r['params']):
        scores = [float(r[f'split{f}_test_score'][n]) for f in range(k)]
        mean = float(r['mean_test_score'][n])
        log.add({c[len(pre):]: v for c, v in cfg.items()},
                {'score': mean if np.all(np.isfinite(scores)) else np.nan,
                 'scores': scores,
                 'seconds': float((r['mean_fit_time'][n] + r['mean_score_time'][n]) * k),
                 'convergence_warnings': np.nan,
                 'error': (failures[0] if failures else 'FitFailed') if not np.all(np.isfinite(scores)) else None})
    return log, {'implementation': type(searcher).__name__, 'n_jobs': 1,
                 'total_convergence_warnings': warn, 'failed_fits': len(failures)}


def grid(base, model, balancing, X, y, cv, budget, seed=C.SEED):
    """Grid Search over the grid of the variant with GridSearchCV.

    If the budget is smaller than the grid (test level), its first points are evaluated. ``refit=False``: the
    refit with the best configuration is done in ``execution.py``.
    """
    space = {M.prefix(balancing) + p.name: p.grid for p in M.SPACES[model]}
    if budget < len(ParameterGrid(space)):          # test level: first points of the grid
        space = [{k: [v] for k, v in cfg.items()} for cfg in list(ParameterGrid(space))[:budget]]
    gs = GridSearchCV(base, space, scoring=SCORE, cv=cv, refit=False, n_jobs=1,
                      error_score=np.nan, return_train_score=False)
    return _sklearn_search(gs, X, y, model, balancing, budget)


def random_search(base, model, balancing, X, y, cv, budget, seed=C.SEED):
    """Random Search: sample ``budget`` configurations with RandomizedSearchCV (log-uniform scales)."""
    dist = {M.prefix(balancing) + p.name: _distribution(p) for p in M.SPACES[model]}
    rs = RandomizedSearchCV(base, dist, n_iter=budget, scoring=SCORE, cv=cv, refit=False,
                            n_jobs=1, random_state=seed, error_score=np.nan, return_train_score=False)
    return _sklearn_search(rs, X, y, model, balancing, budget)


def bayesian(base, model, balancing, X, y, cv, budget, seed=C.SEED):
    """Bayesian optimization with Optuna (TPE surrogate, Expected Improvement acquisition).

    Initial random evaluations: a third of the budget up to 24 (24 → 8, 12 → 4) and 10 % for larger budgets
    (minimum ``BAYES_INITIAL``).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = M.SPACES[model]
    log = _Log(model, balancing, budget)
    initial = max(1, round(budget / 3)) if budget <= 24 else max(BAYES_INITIAL, budget // 10)

    def objective(trial):
        """Optuna objective: propose a configuration and return its inner AUC-PR."""
        cfg = {}
        for p in space:
            if p.kind == 'cat':
                keys = [str(o) for o in p.options]
                cfg[p.name] = p.options[keys.index(trial.suggest_categorical(p.name, keys))]
            elif p.kind == 'int':
                cfg[p.name] = trial.suggest_int(p.name, int(p.low), int(p.high), log=p.log)
            else:
                cfg[p.name] = trial.suggest_float(p.name, p.low, p.high, log=p.log)
        return log.add(cfg, evaluate(base, M.with_prefix(cfg, balancing), X, y, cv),
                       extra={'phase': 'initial' if trial.number < initial else 'TPE'})

    tpe = optuna.samplers.TPESampler(seed=seed, n_startup_trials=initial, n_ei_candidates=BAYES_EI_CANDIDATES)
    study = optuna.create_study(direction='maximize', sampler=tpe)
    study.optimize(objective, n_trials=budget, catch=())
    return log, {'surrogate': 'TPE (l(x)/g(x))', 'acquisition': 'Expected Improvement (EI)',
                 'random_initial': initial, 'seed': seed, 'ei_candidates': BAYES_EI_CANDIDATES}


def genetic(base, model, balancing, X, y, cv, budget, seed=C.SEED, population=None):
    """Genetic algorithm with DEAP over genes in [0, 1] (one per hyperparameter).

    Tournament, BLX-α crossover, Gaussian mutation and elite (``GA_*`` parameters of config). Fitness cache: a
    configuration already evaluated does not consume budget; a repeated child is mutated again or replaced by a
    random immigrant. Records the population diversity per generation.
    """
    from deap import base as dbase
    from deap import creator, tools
    if not hasattr(creator, 'FitnessMaxC1'):
        creator.create('FitnessMaxC1', dbase.Fitness, weights=(1.0,))
    if not hasattr(creator, 'IndividualC1'):
        creator.create('IndividualC1', list, fitness=creator.FitnessMaxC1)
    space = M.SPACES[model]
    n_genes = len(space)
    if population is None:
        population = budget // C.GA_GENERATIONS if budget >= 2 * C.GA_GENERATIONS else budget
    population_n = min(population, budget)
    rnd = random.Random(seed)
    random.seed(seed)                              # DEAP uses the global random module
    log = _Log(model, balancing, budget)

    def diversity(pop, gen):
        """Mean distance between individuals, unique configurations and fitness of the generation."""
        G = np.array([list(i) for i in pop])
        d = [np.linalg.norm(G[i] - G[j]) / np.sqrt(n_genes)
             for i in range(len(G)) for j in range(i + 1, len(G))]
        unique = len({json.dumps(_serializable(_decode(i, space)), sort_keys=True) for i in pop})
        fitness = [i.fitness.values[0] for i in pop if np.isfinite(i.fitness.values[0])]
        return {'generation': gen, 'mean_distance': float(np.mean(d)) if d else 0.0,
                'unique_configurations': unique, 'population': len(pop),
                'best': max(fitness) if fitness else np.nan,
                'mean': float(np.mean(fitness)) if fitness else np.nan}

    # Fitness cache: an already-evaluated configuration is not evaluated again nor consumes budget. If a child
    # repeats a configuration, it is mutated again (up to 20 times) and, if still repeated, replaced by a random
    # immigrant; so every evaluation of the budget is a new configuration.
    cache = {}
    counts = {'duplicates_avoided': 0, 'immigrants': 0}

    def key(ind):
        """JSON key of the configuration, with reals to 4 significant digits (detects repeats)."""
        cfg = _serializable(_decode(ind, space))
        cfg = {k_: (float(f'{v:.4g}') if isinstance(v, float) else v) for k_, v in cfg.items()}
        return json.dumps(cfg, sort_keys=True)

    def make_unique(ind):
        """Mutate again (up to 20 times) or replace by an immigrant an already-evaluated individual."""
        attempts = 0
        while key(ind) in cache and attempts < 20:
            counts['duplicates_avoided'] += 1
            tools.mutGaussian(ind, mu=0.0, sigma=C.GA_SIGMA, indpb=1.0)
            ind[:] = [min(1.0, max(0.0, g)) for g in ind]
            attempts += 1
        while key(ind) in cache and attempts < 120:
            counts['immigrants'] += 1
            ind[:] = [rnd.random() for _ in range(n_genes)]
            attempts += 1
        return ind

    def evaluate_ind(ind, gen):
        """Fitness of an individual with cache: an already-evaluated configuration is not repeated."""
        k = key(ind)
        if k in cache:
            return cache[k]
        cfg = _decode(ind, space)
        cache[k] = (log.add(cfg, evaluate(base, M.with_prefix(cfg, balancing), X, y, cv), extra={'generation': gen}),)
        return cache[k]

    pop = []
    for _ in range(population_n):
        ind = make_unique(creator.IndividualC1([rnd.random() for _ in range(n_genes)]))
        ind.fitness.values = evaluate_ind(ind, 0)
        pop.append(ind)
    history = [diversity(pop, 0)]
    gen = 0
    while len(log.rows) < budget:
        gen += 1
        elite = [creator.IndividualC1(list(i)) for i in tools.selBest(pop, C.GA_ELITE)]
        for e, o in zip(elite, tools.selBest(pop, C.GA_ELITE)):
            e.fitness.values = o.fitness.values
        children = [creator.IndividualC1(list(i)) for i in tools.selTournament(pop, population_n,
                                                                                tournsize=C.GA_TOURNAMENT)]
        for a, b in zip(children[::2], children[1::2]):
            if random.random() < C.GA_P_CROSSOVER:
                tools.cxBlend(a, b, alpha=0.5)
        for h in children:
            if random.random() < C.GA_P_MUTATION:
                tools.mutGaussian(h, mu=0.0, sigma=C.GA_SIGMA, indpb=C.GA_INDPB)
            h[:] = [min(1.0, max(0.0, g)) for g in h]
        evaluated = []
        for h in children:
            if len(log.rows) >= budget:           # exact budget of unique evaluations
                break
            make_unique(h)
            h.fitness.values = evaluate_ind(h, gen)
            evaluated.append(h)
        evaluated.sort(key=lambda i: i.fitness.values[0], reverse=True)
        rest = sorted(pop, key=lambda i: i.fitness.values[0], reverse=True)
        pop = (elite + evaluated + rest)[:population_n]   # elite + best children (+ best previous if needed)
        history.append(diversity(pop, gen))
    return log, {'population': population_n, 'generations': gen + 1, 'seed': seed,
                 'selection': f'tournament k={C.GA_TOURNAMENT}', 'crossover': f'BLX-α (α=0.5), p={C.GA_P_CROSSOVER}',
                 'mutation': f'Gaussian σ={C.GA_SIGMA}, p={C.GA_P_MUTATION}, per gene {C.GA_INDPB}',
                 'elitism': C.GA_ELITE, 'unique_evaluations': len(log.rows), **counts, 'diversity': history}


METHODS = {'grid': grid, 'random': random_search, 'bayes': bayesian, 'genetic': genetic}


def optimize(method, model, balancing, X, y, cv, budget, seed=C.SEED, categorical=None, **options):
    """Run an optimizer and return the best configuration and its history.

    Returns
    -------
    best : dict or None
        Winning hyperparameters (without prefix); None if every evaluation failed.
    history : list of dict
    info : dict
        Details of the method (surrogate, operators, diversity...).
    """
    base = M.pipeline(model, balancing, categorical)
    log, info = METHODS[method](base, model, balancing, X, y, cv, budget, seed=seed, **options)
    return log.best[1], log.rows, info
