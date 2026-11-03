"""Data-generating mechanism of simulation v2 (the «D» of ADEMP).

The true population is the one of simulation v1 (``simulation.generator.population``, frozen file, same
declared assumptions and the same random draws). Version 2 adds a new way of **recording** it, with the
parameters that phase 6 showed to matter:

* ``pi``: coupling between form and class. P(form B | completed) = pi and P(form B | attempt) = 1 − pi.
  ``pi = 1`` is Valledupar (the form is the class); ``pi = 0.5`` makes the form independent of the class.
  ``None``: every event with form A (the complete instrument).
* ``gap``: the concept with structural gaps (``'education'``, weak effect, as in the real data, or
  ``'occupation'``, strong effect, OR ≈ 2.2).
* ``gap_mode``: where the form does not measure it. ``'A2016'``: form A in 2016 (as the real 2016 gap of
  form 356); ``'B_all'``: form B in every year (another municipality with another instrument for completed
  suicides); ``'none'``.
* ``blanks``: blanks of the gap concept made by people, not by the form, in every form and registry.
  ``'MAR'``: about 10 % that depend on age; ``'MNAR'``: they depend on the true value (low education or not
  employed are left blank more often), stable in every registry, so the blank is real information;
  ``'none'``.
* ``convergence``: year from which completed suicides are recorded with an instrument that measures what
  form 356 measures (form A); ``None``.

Form B keeps the other differences of simulation v1 (questions not asked, single choice, blanks recorded
as "No", education in 5 levels) because they are part of the real registry; the eligibility rule of
RabMV handles them and the checks of v1 already validated their detection.

Domain facts behind the targets (user, 1 Nov 2026): completed suicides come from vital statistics (RUAF-ND,
Legal Medicine) and each municipality codes them with its own instrument; the Valledupar forms are becoming
more similar to SIVIGILA over time; a field written once in the paper form becomes 1 for the option written
and 2 for the others when it is moved into the SIVIGILA structure.
"""
import numpy as np
import pandas as pd

from simulation.generator import _BIN, _CATALOG, _EDU_105, EDUCATION_105, FACTORS, SCHEMES, _sec, population, subsample

GAPS = ('education', 'occupation')
TARGETS = ('T0', 'T1', 'T2', 'T3', 'T4')
TARGET_SIZE = 10_000                     # events per target registry (about 790 completed suicides)
CONVERGENCE_YEAR = 2020
YEAR_TREND = (1.0, 0.6)                  # no_difference_year: keep a completed suicide, logit 1.0 + 0.6 (year − 2019)


def condition_params(condition):
    """Recording parameters of a condition (see ``CONDITIONS``)."""
    return dict(CONDITIONS[condition])


# Conditions with 180 events (pre-specified); no-difference scenarios have no gap and every event with form A.
CONDITIONS = {}
for _gap in GAPS:
    for _pi in (1.0, 0.85, 0.5):
        CONDITIONS[f'E_pi{int(_pi * 100)}_{_gap}'] = dict(pi=_pi, gap=_gap, gap_mode='A2016', blanks='none')
    CONDITIONS[f'MAR_{_gap}'] = dict(pi=0.5, gap=_gap, gap_mode='none', blanks='MAR')
    CONDITIONS[f'MNAR_{_gap}'] = dict(pi=0.5, gap=_gap, gap_mode='none', blanks='MNAR')
CONDITIONS['no_difference'] = dict(pi=None, gap=None, gap_mode='none', blanks='none')
CONDITIONS['no_difference_year'] = dict(pi=None, gap=None, gap_mode='none', blanks='none', year_trend=YEAR_TREND)


def target_params(params, target):
    """Recording parameters of a target registry for a model trained under ``params``.

    T0 the same registry; T1 the complete instrument for both classes (people still leave blanks); T2 the
    form independent of the class; T3 another municipality whose completed-suicide instrument does not
    measure the gap concept; T4 convergence from 2020 on.
    """
    p = dict(params)
    if target == 'T0':
        return p
    if target == 'T1':
        return dict(p, pi=None, gap_mode='none', convergence=None)
    if target == 'T2':
        return dict(p, pi=0.5 if p['pi'] is not None else None)
    if target == 'T3':
        return dict(p, gap_mode='B_all' if p['gap'] else 'none', pi=p['pi'] if p['pi'] is not None else None)
    if target == 'T4':
        return dict(p, convergence=CONVERGENCE_YEAR)
    raise ValueError(target)


def training_population(seed, events, params):
    """Population of the training registry (v1 population, optionally with a year trend of the class)."""
    pop = subsample(population(seed), events, seed)
    if params.get('year_trend'):
        pop = _year_trend(pop, params['year_trend'], seed)
    return pop


def target_population(seed, params):
    """Population of a target registry: new events (another seed) with the natural prevalence."""
    pop = population(seed + 7_000, n=TARGET_SIZE)
    if params.get('year_trend'):
        pop = _year_trend(pop, params['year_trend'], seed + 7_000)
    return pop


def _year_trend(pop, trend, seed):
    """Keep completed suicides with a probability that grows with the year (label–era association only)."""
    rng = np.random.default_rng(seed + 40_000)
    a, b = trend
    keep_p = 1 / (1 + np.exp(-(a + b * (pop['year'].to_numpy() - 2019))))
    drop = (pop['y'].to_numpy() == 1) & (rng.random(len(pop)) > keep_p)
    return pop.loc[~drop].reset_index(drop=True)


def record(pop, params, seed):
    """Source data as recorded by the forms (columns of the source plus ``form`` and ``date``)."""
    rng = np.random.default_rng(seed + 20_000)
    rng_form = np.random.default_rng(seed + 30_000)
    n = len(pop)
    y = pop['y'].to_numpy() == 1
    year = pop['year'].to_numpy()
    pi, gap, gap_mode, blanks = params['pi'], params['gap'], params['gap_mode'], params['blanks']
    if pi is None:
        form_b = np.zeros(n, bool)
    else:
        u = rng_form.random(n)
        form_b = np.where(y, u < pi, u < 1 - pi)
    if params.get('convergence'):
        form_b &= year < params['convergence']
    d = pd.DataFrame(index=pop.index)
    d['form'] = np.where(form_b, 'B', 'A')
    d['date'] = [f'{a}-{m:02d}-15' for a, m in zip(year, rng.integers(1, 13, n))]
    d['age'] = pop['age']
    d['sex'] = np.where(form_b, pop['sex'].map({'F': 'Female', 'M': 'Male'}), pop['sex'])
    d['area'] = pop['area']
    d['insurance'] = pop['insurance']
    d['occupation'] = np.where(pop['occupation'] == 1, 1, 2).astype(object)
    d['stratum'] = pop['stratum']
    d['marital_status'] = pop['marital_status']
    d['education'] = pd.Series(np.where(form_b, pop['education'].map(EDUCATION_105), pop['education'].astype(object)),
                               index=pop.index, dtype=object)
    for f in FACTORS:
        d[f] = np.where(pop[f] == 1, 1.0, 2.0)
    if form_b.any():
        d.loc[form_b, ['f1', 'f2']] = 2.0
        for i in np.flatnonzero(form_b):
            marked = [f for f in ('f3', 'f4', 'f5') if pop.at[pop.index[i], f] == 1]
            if len(marked) > 1:
                chosen = marked[rng.integers(len(marked))]
                for f in marked:
                    d.at[d.index[i], f] = 1.0 if f == chosen else 2.0
        blank6 = form_b & (rng.random(n) < 0.9)
        d.loc[blank6, 'f6'] = np.nan
    if gap is not None:
        rng_gap = np.random.default_rng(seed + 50_000)
        if gap_mode == 'A2016':
            not_measured = (~form_b) & (year == 2016)
        elif gap_mode == 'B_all':
            not_measured = form_b.copy()
        else:
            not_measured = np.zeros(n, bool)
        if blanks == 'MAR':
            p = np.clip(0.10 * pop['age'].to_numpy() / 32.0, 0, 0.5)
            not_measured |= rng_gap.random(n) < p
        elif blanks == 'MNAR':
            if gap == 'education':
                p = np.where(pop['education'].to_numpy() <= 2, 0.5, 0.03)
            else:
                p = np.where(pop['occupation'].to_numpy() == 0, 0.3, 0.03)
            not_measured |= rng_gap.random(n) < p
        d.loc[not_measured, gap] = None
    return d


# ----------------------------------------------------------------------------- schemas (TOML)
def _common(form_b, occupation=True, occupation_blank='no'):
    sex = {'female': '"F"', 'male': '"M"'} if form_b else {'f': '"F"', 'm': '"M"'}
    out = [_sec('age'), _sec('sex', recoding=sex), _sec('area', recoding={str(k): k for k in (1, 2, 3)}),
           _sec('insurance', recoding={r.lower(): f'"{r}"' for r in SCHEMES})]
    if occupation:
        out.append(_sec('occupation', blank=occupation_blank, recoding=_BIN, response='multiple'))
    out += [_sec('stratum', recoding={str(k): k for k in range(7)}),
            _sec('marital_status', recoding={str(k): k for k in range(1, 6)})]
    return out


def schema_texts(params, declared='correct'):
    """Catalog and mappings (TOML) of a registry recorded with ``params``.

    ``declared='correct'`` tells the truth; ``'all_exact'`` declares every concept measured and exact in
    both forms, in a single version (what an ETL that does not know the forms would declare): the checks
    must detect the rest.
    """
    gap, gap_mode, blanks = params['gap'], params['gap_mode'], params['blanks']
    occ_blank = 'missing' if (gap == 'occupation' and (blanks != 'none' or gap_mode != 'none')) else 'no'
    head = 'form = "{f}"\nversion = "{v}"\nvalid_from = {d}\n{h}\n'
    fac_a = [_sec(f, blank='no', recoding=_BIN, response='multiple') for f in FACTORS]
    edu_a = _sec('education', recoding={str(k): k for k in range(13)})
    edu_b = _sec('education', 'aggregated', recoding=_EDU_105, resolution='five_levels')
    fac_b = ([_sec(f, 'partial', blank='no', recoding=_BIN, response='single') for f in ('f3', 'f4', 'f5')]
             + [_sec('f6', 'partial', blank='no', recoding=_BIN, response='multiple')]
             + [_sec(f, blank='no', recoding=_BIN, response='multiple') for f in ('f7', 'f8')])
    uses_b = params['pi'] is not None

    def form_a(with_gap=True):
        occ = not (gap == 'occupation' and not with_gap)
        edu = [] if (gap == 'education' and not with_gap) else [edu_a]
        return '\n'.join(_common(False, occ, occ_blank) + edu + fac_a)

    def form_b(with_gap=True):
        occ = not (gap == 'occupation' and not with_gap)
        edu = [] if (gap == 'education' and not with_gap) else [edu_b]
        return '\n'.join(_common(True, occ, occ_blank) + edu + fac_b)

    if declared == 'all_exact':
        maps = {'A': head.format(f='A', v='1', d='2016-01-01', h='') + form_a()}
        if uses_b:
            maps['B'] = (head.format(f='B', v='1', d='2016-01-01', h='') + '\n'.join(
                _common(True, True, occ_blank) + [edu_b] + fac_a))
        return _CATALOG, maps
    if declared != 'correct':
        raise ValueError(declared)
    maps = {}
    if gap_mode == 'A2016':
        maps['A_2016'] = head.format(f='A', v='2016', d='2016-01-01', h='valid_to = 2016-12-31') + form_a(False)
        maps['A_2017'] = head.format(f='A', v='2017', d='2017-01-01', h='') + form_a()
    else:
        maps['A'] = head.format(f='A', v='1', d='2016-01-01', h='') + form_a()
    if uses_b:
        maps['B'] = head.format(f='B', v='1', d='2016-01-01', h='') + form_b(gap_mode != 'B_all')
    return _CATALOG, maps


def schema(params, declared='correct'):
    """Schema object from ``schema_texts``."""
    import tomllib

    from rabmv.schema import Catalog, Mapping, Schema
    catalog_text, mappings = schema_texts(params, declared)
    cat = Catalog.from_dict(tomllib.loads(catalog_text), source='simulation v2 catalog')
    return Schema(cat, [Mapping.from_dict(tomllib.loads(t), cat, source=n) for n, t in mappings.items()])
