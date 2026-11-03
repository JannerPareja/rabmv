"""Tests of the empirical audit checks (phase 2e).

Synthetic generator: people with four binary factors (multiple choice, ~25 % each) and age, recorded by
form «A» (class 0) or form «B» (class 1). In each scenario one type of differential measurement that the
schema does NOT declare (every mapping says «exact») is injected, and it is checked that the audit
detects it and that, with nothing injected, it gives no false alarms. Made-up data.
"""
import tomllib

import numpy as np
import pandas as pd
import pytest

from rabmv.availability import harmonize
from rabmv.checks import run_checks
from rabmv.schema import Catalog, Mapping, Schema

FACTORS = ['f1', 'f2', 'f3', 'f4']
CATALOG = '\n'.join(['[catalog]\nname = "test"\nversion = "1"\n',
                     '[concepts.age]\ndefinition = "Age"\ntype = "numeric"\n']
                    + [f'[concepts.{f}]\ndefinition = "Factor {f}"\ntype = "binary"\n' for f in FACTORS])


def mapping(form):
    """Mapping of a form that declares everything exact (the schema «does not know» about the difference)."""
    parts = [f'form = "{form}"\nversion = "1"\nvalid_from = 2016-01-01\n',
             '[concepts.age]\nequivalence = "exact"\nblank = "missing"\n']
    parts += [f'[concepts.{f}]\nequivalence = "exact"\nresponse = "multiple"\nblank = "no"\n'
              'recoding = { 1 = 1, 2 = 0 }\n' for f in FACTORS]
    return '\n'.join(parts)


@pytest.fixture(scope='module')
def schema():
    """Schema with two forms declared equivalent in everything."""
    cat = Catalog.from_dict(tomllib.loads(CATALOG))
    return Schema(cat, [Mapping.from_dict(tomllib.loads(mapping(f)), cat) for f in ('A', 'B')])


def generate(scenario, seed, n0=1500, n1=180):
    """Source data of a scenario: 'null', 'A', 'B', 'C' or 'E'."""
    rng = np.random.default_rng(seed)
    n = n0 + n1
    y = np.r_[np.zeros(n0, int), np.ones(n1, int)]
    prob = np.where(y[:, None] == 1, 0.32, 0.25) * np.ones((1, 4))
    fac = (rng.random((n, 4)) < prob).astype(float)
    if scenario == 'A':                                     # form B does not ask f1: it stays at No
        fac[y == 1, 0] = 0
    if scenario == 'B':                                     # form B only allows one option
        for i in np.flatnonzero(y == 1):
            marked = np.flatnonzero(fac[i] == 1)
            if len(marked) > 1:
                fac[i] = 0
                fac[i, rng.choice(marked)] = 1
    raw = np.where(fac == 1, 1.0, 2.0)
    if scenario == 'C':                                     # form B leaves f2 blank 90 % of the time
        raw[(y == 1) & (rng.random(n) < 0.9), 1] = np.nan
    years = rng.integers(2016, 2023, n)
    if scenario == 'E':                                     # in 2016 form A did not have f3
        raw[(y == 0) & (years == 2016), 2] = np.nan
    d = pd.DataFrame(raw, columns=FACTORS)
    d['age'] = rng.integers(10, 80, n)
    d['form'] = np.where(y == 1, 'B', 'A')
    d['date'] = [f'{a}-06-15' for a in years]
    return d, pd.Series(y)


def run(schema, scenario, seed=0, with_data=True):
    """Harmonize and check a scenario."""
    d, y = generate(scenario, seed)
    a = harmonize(d, schema)
    return run_checks(a, y, schema, data=d if with_data else None)


def alerts(v):
    """(check, concept) pairs with an alert."""
    return {(r.check, r.concept) for r in v.alerts().itertuples()}


@pytest.mark.parametrize('seed', range(10))
def test_no_differential_measurement_no_false_alarms(schema, seed):
    """Null scenario (S4): no alert."""
    assert alerts(run(schema, 'null', seed)) == set()


def test_detects_type_A(schema):
    """A factor that the form of one class does not ask (structural zero)."""
    v = run(schema, 'A')
    assert ('A', 'f1') in alerts(v)
    assert {t for t, c in alerts(v)} == {'A'}
    row = v.table[(v.table.check == 'A') & (v.table.concept == 'f1')].iloc[0]
    assert 'equals 0 in every case of class 1' in row.detail and row.p < 1e-6


def test_detects_type_B(schema):
    """The factors of a single-choice question never appear together in that class."""
    v = run(schema, 'B')
    assert {t for t, _ in alerts(v)} == {'B'}
    pairs = {c for t, c in alerts(v)}
    assert len(pairs) >= 4                                   # most of the 6 pairs of f1..f4
    row = v.alerts().iloc[0]
    assert 'never marked together' in row.detail and row.n_both_class_1 == 0


def test_type_B_is_not_hidden_by_other_questions(schema):
    """Single choice only between f1 and f2; f3 and f4 remain multiple choice: the pair is detected."""
    d, y = generate('null', 7)
    for i in np.flatnonzero(y == 1):
        if d.loc[i, 'f1'] == 1 and d.loc[i, 'f2'] == 1:
            d.loc[i, 'f2' if i % 2 else 'f1'] = 2.0
    v = run_checks(harmonize(d, schema), y, schema, data=d)
    assert ('B', 'f1 & f2') in alerts(v)
    assert all(c == 'f1 & f2' for t, c in alerts(v))


def test_detects_type_C_only_with_the_source_data(schema):
    """Blanks turned into "No": they are visible in the source, not in the harmonized data."""
    v = run(schema, 'C')
    assert ('C', 'f2') in alerts(v)
    assert all(t != 'A' or c != 'f2' for t, c in alerts(v))
    without = run(schema, 'C', with_data=False)
    assert ('C', 'f2') not in alerts(without)
    assert (without.table[without.table.check == 'C'].result == 'not evaluable').all()


def test_detects_type_E(schema):
    """A year in which the form did not have the field."""
    v = run(schema, 'E')
    assert ('E', 'f3') in alerts(v)
    row = v.table[(v.table.check == 'E') & (v.table.concept == 'f3')].iloc[0]
    assert 'form A, year 2016: completeness 0%' in row.detail


def test_subtle_type_F_is_not_detectable(schema):
    """Negative control (declared limit): a subtle measurement difference raises no alert.

    If form B records age 3 years higher, that is confounded with a real effect of age when the form
    coincides with the class. No check can see it; that is why the design requires declaring it in the
    mapping and reports it as a limit.
    """
    d, y = generate('null', 3)
    d.loc[y == 1, 'age'] += 3
    v = run_checks(harmonize(d, schema), y, schema, data=d)
    assert alerts(v) == set()


def test_report_without_rows_and_with_suppression(schema):
    """The report has counts (suppressed from 1 to 4) and no individual data."""
    v = run(schema, 'B')
    aud = v.audit()
    text = aud.to_string()
    assert 'age' in text and 'f1' in text
    for c in [c for c in aud.columns if c.startswith('n_')]:
        assert all(not isinstance(x, (int, np.integer)) or x == 0 or x >= 5 for x in aud[c].dropna())
    assert set(v.parameters) == {'alpha', 'blank_difference', 'min_cases', 'low_completeness', 'high_completeness'}


def test_invalid_inputs(schema):
    """Misaligned indices or a single class."""
    d, y = generate('null', 0)
    a = harmonize(d, schema)
    with pytest.raises(ValueError, match='same index'):
        run_checks(a, y.set_axis(range(1, len(y) + 1)), schema)
    with pytest.raises(ValueError, match='2 classes'):
        run_checks(a, pd.Series(np.zeros(len(y))), schema)
    with pytest.raises(ValueError, match='data must have the same index'):
        run_checks(a, y, schema, data=d.iloc[:-1])
