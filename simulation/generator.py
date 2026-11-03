"""Data-generating mechanism (the «D» of ADEMP): people with true effects, recorded by different forms.

True population (declared assumptions, not estimates):

* ``sex``: 36 % men among the events; log-OR of completed suicide of 2.3 for men (prevalence of completed
  suicide close to 16 % in men versus 3 % in women; in the Valledupar study, 19 % versus 4 %).
* ``age``: log-OR of 0.075 per year: from about 2 % completed suicides under 20 to about 29 % at 60+ (in
  the study, from 3 % to 41 % in 2013–2022; the 2016–2022 window has fewer older completed suicides).
* ``area`` (3 categories), ``insurance`` (6), ``marital_status`` (5), ``occupation`` (binary), ``stratum``
  (0–6) and ``education`` (0–12): small or moderate effects, assumed.
* Eight psychosocial factors ``f1``–``f8``: prevalences of 15 % to 40 % among the events and OR between
  1.5 and 3 (assumed), independent of each other.
* Years 2016–2022 uniform. The intercept is set for an expected prevalence of 7.7 % (about 180 completed
  suicides in 2,350 events); the versions with 120 and 60 events subsample the completed suicides.

Forms (the form coincides with the class, as in Valledupar, except in the transport scenario):

* Form ``A`` (attempts): measures everything, multiple choice, no blanks; in 2016 it has no education (E).
* Form ``B`` (completed suicides): does not ask ``f1`` or ``f2`` (they stay at "No", A); ``f3``–``f5`` in a
  single-choice question (B); ``f6`` almost always blank and the blank is turned into "No" (C); education
  in 5 levels (aggregated); everything else as A.
* Subtle difference F (only in scenario ``'F'``): form B records age 3 years higher.

Translated to English on 27 Oct 2026 from simulacion/generador.py (fingerprint 528A212079C3E55F) without
changing any random draw: the evaluation reproduces the frozen results exactly.
"""
import numpy as np
import pandas as pd

FACTORS = [f'f{i}' for i in range(1, 9)]
FACTOR_PREVALENCE = dict(zip(FACTORS, [0.30, 0.20, 0.35, 0.25, 0.20, 0.30, 0.15, 0.40]))
FACTOR_LOGOR = dict(zip(FACTORS, np.log([2.0, 1.5, 2.5, 1.8, 3.0, 2.0, 1.6, 1.5])))
SCHEMES = ['C', 'S', 'P', 'E', 'N', 'I']
SCHEME_PROB = [0.23, 0.64, 0.03, 0.02, 0.07, 0.01]
SCHEME_LOGOR = dict(zip(SCHEMES, [0.0, 0.2, -0.3, 0.1, 0.5, 0.0]))
AREA_LOGOR = {1: 0.0, 2: 0.3, 3: 0.5}
MARITAL_LOGOR = {1: 0.0, 2: 0.3, 3: 0.2, 4: 0.6, 5: 0.5}
EDUCATION_105 = {0: 'out of school', 1: 'primary', 2: 'primary', 3: 'secondary', 4: 'secondary',
                 5: 'secondary', 6: 'secondary', 7: 'technical', 8: 'technical', 9: 'university',
                 10: 'university', 11: 'university', 12: 'university'}
INTERCEPT = -8.1            # expected prevalence close to 7.6 % (about 180 completed suicides in 2,350 events)


def population(seed, n=2350):
    """Events with their true values and their class (1 = completed suicide).

    Parameters
    ----------
    seed : int
    n : int, default=2350

    Returns
    -------
    pandas.DataFrame
        True values, ``year``, ``logit`` and ``y``.
    """
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        'age': np.clip(np.round(rng.gamma(4.0, 6.5, n) + 10), 10, 90).astype(int),
        'sex': np.where(rng.random(n) < 0.36, 'M', 'F'),
        'area': rng.choice([1, 2, 3], n, p=[0.87, 0.07, 0.06]),
        'insurance': rng.choice(SCHEMES, n, p=SCHEME_PROB),
        'occupation': (rng.random(n) < 0.30).astype(int),
        'stratum': rng.choice(np.arange(7), n, p=[0.05, 0.45, 0.30, 0.12, 0.05, 0.02, 0.01]),
        'marital_status': rng.choice([1, 2, 3, 4, 5], n, p=[0.58, 0.07, 0.25, 0.02, 0.08]),
        'education': rng.choice(np.arange(13), n, p=[0.02, 0.01, 0.20, 0.45, 0.03, 0.05, 0.01, 0.04,
                                                     0.03, 0.12, 0.02, 0.01, 0.01]),
        'year': rng.integers(2016, 2023, n),
    })
    for f in FACTORS:
        d[f] = (rng.random(n) < FACTOR_PREVALENCE[f]).astype(int)
    logit = (INTERCEPT + 2.3 * (d['sex'] == 'M') + 0.075 * (d['age'] - 10)
             + d['area'].map(AREA_LOGOR) + d['insurance'].map(SCHEME_LOGOR) + 0.8 * d['occupation']
             - 0.08 * d['stratum'] + d['marital_status'].map(MARITAL_LOGOR) - 0.04 * d['education']
             + sum(FACTOR_LOGOR[f] * d[f] for f in FACTORS))
    d['logit'] = logit
    d['y'] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return d


def subsample(pop, events, seed):
    """Keep every attempt and ``events`` random completed suicides (if there are more)."""
    rng = np.random.default_rng(seed + 10_000)
    completed = np.flatnonzero(pop['y'].to_numpy() == 1)
    if len(completed) <= events:
        return pop.reset_index(drop=True)
    drop = rng.choice(completed, len(completed) - events, replace=False)
    return pop.drop(index=pop.index[drop]).reset_index(drop=True)


def record(pop, scenario, seed):
    """Source data as recorded by the forms.

    Parameters
    ----------
    pop : pandas.DataFrame
        Output of ``population`` (or ``subsample``).
    scenario : {'valledupar', 'F', 'null', 'transport'}
        ``'valledupar'``: form A for attempts and form B, with its differences, for completed suicides.
        ``'F'``: like valledupar, plus age shifted in form B.
        ``'null'``: both classes with the complete form A (no differences and no 2016 gap).
        ``'transport'``: completed suicides also with form A (a secretariat with a complete form).
    seed : int

    Returns
    -------
    pandas.DataFrame
        Source columns plus ``form`` and ``date``.
    """
    rng = np.random.default_rng(seed + 20_000)
    n = len(pop)
    completed = pop['y'].to_numpy() == 1
    form_b = completed & (scenario in ('valledupar', 'F'))
    d = pd.DataFrame(index=pop.index)
    d['form'] = np.where(form_b, 'B', 'A')
    d['date'] = [f'{a}-{m:02d}-15' for a, m in zip(pop['year'], rng.integers(1, 13, n))]
    d['age'] = pop['age'] + np.where(form_b & (scenario == 'F'), 3, 0)
    d['sex'] = np.where(form_b, pop['sex'].map({'F': 'Female', 'M': 'Male'}), pop['sex'])
    d['area'] = pop['area']
    d['insurance'] = pop['insurance']
    d['occupation'] = np.where(pop['occupation'] == 1, 1, 2)
    d['stratum'] = pop['stratum']
    d['marital_status'] = pop['marital_status']
    edu = pop['education'].astype(object)
    edu = np.where(form_b, pop['education'].map(EDUCATION_105), edu)
    gap_2016 = (~form_b) & (pop['year'].to_numpy() == 2016) & (scenario in ('valledupar', 'F'))
    d['education'] = pd.Series(edu, index=pop.index).where(~gap_2016, None)
    for f in FACTORS:
        d[f] = np.where(pop[f] == 1, 1.0, 2.0)
    if form_b.any():
        d.loc[form_b, ['f1', 'f2']] = 2.0                                   # A: not asked → "No"
        for i in np.flatnonzero(form_b):                                    # B: single choice f3–f5
            marked = [f for f in ('f3', 'f4', 'f5') if pop.at[pop.index[i], f] == 1]
            if len(marked) > 1:
                chosen = marked[rng.integers(len(marked))]
                for f in marked:
                    d.at[d.index[i], f] = 1.0 if f == chosen else 2.0
        blank = form_b & (rng.random(n) < 0.9)                              # C: f6 almost always blank
        d.loc[blank, 'f6'] = np.nan
    return d


# ----------------------------------------------------------------------------- schemas (TOML)
_CATALOG = '''[catalog]
name = "RabMV simulation"
version = "sim-1"

[concepts.age]
definition = "Age in years"
type = "numeric"
minimum = 0
maximum = 120

[concepts.sex]
definition = "Sex"
type = "nominal"
categories = ["F", "M"]

[concepts.area]
definition = "Area"
type = "nominal"
categories = [1, 2, 3]

[concepts.insurance]
definition = "Health insurance scheme"
type = "nominal"
categories = ["C", "S", "P", "E", "N", "I"]

[concepts.occupation]
definition = "Employed"
type = "binary"

[concepts.stratum]
definition = "Socioeconomic stratum"
type = "ordinal"
levels = [0, 1, 2, 3, 4, 5, 6]

[concepts.marital_status]
definition = "Marital status"
type = "nominal"
categories = [1, 2, 3, 4, 5]

[concepts.education]
definition = "Education (0 = none)"
type = "ordinal"
levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

[concepts.education.resolutions]
five_levels = [[0], [1, 2], [3, 4, 5, 6], [7, 8], [9, 10, 11, 12]]
''' + ''.join(f'\n[concepts.{f}]\ndefinition = "Factor {f}"\ntype = "binary"\n' for f in FACTORS)


def _sec(concept, equivalence='exact', blank='missing', recoding=None, **extra):
    """TOML section of a concept in a mapping."""
    lines = [f'[concepts.{concept}]', f'equivalence = "{equivalence}"', f'blank = "{blank}"']
    lines += [f'{k} = "{v}"' for k, v in extra.items()]
    if recoding:
        lines.append('recoding = { ' + ', '.join(f'"{k}" = {v}' for k, v in recoding.items()) + ' }')
    return '\n'.join(lines) + '\n'


_BIN = {'1': 1, '2': 0}


def _common(form_b=False):
    """Sections of the concepts that both forms measure the same way (except the sex labels)."""
    sex = {'female': '"F"', 'male': '"M"'} if form_b else {'f': '"F"', 'm': '"M"'}
    return [_sec('age'), _sec('sex', recoding=sex), _sec('area', recoding={str(k): k for k in (1, 2, 3)}),
            _sec('insurance', recoding={r.lower(): f'"{r}"' for r in SCHEMES}),
            _sec('occupation', blank='no', recoding=_BIN, response='multiple'),
            _sec('stratum', recoding={str(k): k for k in range(7)}),
            _sec('marital_status', recoding={str(k): k for k in range(1, 6)})]


_EDU_105 = {'out of school': 0, 'primary': 2, 'secondary': 3, 'technical': 7, 'university': 9}


def schema_texts(declared):
    """Catalog and mappings (TOML) according to what is declared.

    Parameters
    ----------
    declared : {'correct', 'all_exact', 'null'}
        ``'correct'``: the schema tells the truth about forms A and B.
        ``'all_exact'``: the schema declares everything exact and measured (scenario S3): the checks must
        detect what was not declared.
        ``'null'``: both forms complete and equal (scenario S4 and transport).

    Returns
    -------
    (str, dict)
        Catalog text and {name: text} of the mappings.
    """
    edu_a = _sec('education', recoding={str(k): k for k in range(13)})
    fac_a = [_sec(f, blank='no', recoding=_BIN, response='multiple') for f in FACTORS]
    head = 'form = "{f}"\nversion = "{v}"\nvalid_from = {d}\n{h}\n'
    if declared == 'correct':
        a16 = head.format(f='A', v='2016', d='2016-01-01', h='valid_to = 2016-12-31') + '\n'.join(_common() + fac_a)
        a17 = head.format(f='A', v='2017', d='2017-01-01', h='') + '\n'.join(_common() + [edu_a] + fac_a)
        b = (head.format(f='B', v='1', d='2016-01-01', h='') + '\n'.join(
            _common(True)
            + [_sec('education', 'aggregated', recoding=_EDU_105, resolution='five_levels')]
            + [_sec(f, 'partial', blank='no', recoding=_BIN, response='single') for f in ('f3', 'f4', 'f5')]
            + [_sec('f6', 'partial', blank='no', recoding=_BIN, response='multiple')]
            + [_sec(f, blank='no', recoding=_BIN, response='multiple') for f in ('f7', 'f8')]))
        return _CATALOG, {'A_2016': a16, 'A_2017': a17, 'B': b}
    if declared == 'all_exact':
        a = head.format(f='A', v='1', d='2016-01-01', h='') + '\n'.join(_common() + [edu_a] + fac_a)
        b = (head.format(f='B', v='1', d='2016-01-01', h='') + '\n'.join(
            _common(True) + [_sec('education', 'aggregated', recoding=_EDU_105, resolution='five_levels')] + fac_a))
        return _CATALOG, {'A': a, 'B': b}
    if declared == 'null':
        a = head.format(f='A', v='1', d='2016-01-01', h='') + '\n'.join(_common() + [edu_a] + fac_a)
        return _CATALOG, {'A': a}
    raise ValueError(declared)


def schema(declared):
    """Schema object from ``schema_texts``."""
    import tomllib

    from rabmv.schema import Catalog, Mapping, Schema
    catalog_text, mappings = schema_texts(declared)
    cat = Catalog.from_dict(tomllib.loads(catalog_text), source='simulation catalog')
    return Schema(cat, [Mapping.from_dict(tomllib.loads(t), cat, source=n) for n, t in mappings.items()])
