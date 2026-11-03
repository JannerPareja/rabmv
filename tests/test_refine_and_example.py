"""The optional refinement of check B and the public example. Synthetic data only."""
import runpy
from pathlib import Path

import pandas as pd
import pytest

import rabmv
from simulation_v2.generator import condition_params, record, schema, training_population

ROOT = Path(__file__).resolve().parent.parent


def _checks(seed):
    p = condition_params('E_pi100_education')
    pop = training_population(seed, 180, p)
    d = record(pop, p, seed)
    s = schema(p, 'all_exact')
    return rabmv.run_checks(rabmv.harmonize(d, s), pop['y'], s, data=d)


def _alerted(checks):
    out = set()
    for c in checks.alerts()['concept']:
        out.update(x.strip() for x in c.split('&'))
    return out


def test_drop_explained_pairs_removes_the_false_partner_and_keeps_the_rest():
    """Seed 116 of simulation v2: f8 is alerted only as the partner of f6 (blanks recorded as "No", alert C)."""
    checks = _checks(116)
    assert 'f8' in _alerted(checks)
    refined = rabmv.drop_explained_pairs(checks)
    assert 'f8' not in _alerted(refined)
    assert {'education', 'f1', 'f2', 'f6'} <= _alerted(refined)           # real differences still alerted
    a, b = checks.table, refined.table
    changed = a['result'] != b['result']
    assert (a.loc[changed, 'check'] == 'B').all() and changed.sum() >= 1
    assert b.loc[changed, 'detail'].str.contains('explained by the alert of f6').all()
    assert 'refined' in refined.parameters and 'refined' not in checks.parameters   # the original is untouched


def test_drop_explained_pairs_rejects_other_input():
    with pytest.raises(TypeError):
        rabmv.drop_explained_pairs(pd.DataFrame())


def test_quickstart_runs(capsys):
    runpy.run_path(str(ROOT / 'examples' / 'quickstart.py'), run_name='__main__')
    out = capsys.readouterr().out
    assert 'Held-out AUC-PR' in out and 'education' in out and 'meets the rule' in out


def test_example_registry_is_synthetic_and_matches_its_schema():
    here = ROOT / 'examples' / 'synthetic_registry'
    data = pd.read_csv(here / 'registry.csv', dtype=str, keep_default_na=False, na_values=[''])
    assert len(data) == 2350 and set(data['outcome']) == {'attempt', 'completed'}
    assert ((data['form'] == 'B') == (data['outcome'] == 'completed')).all()
    s = rabmv.Schema.from_folder(here / 'catalog.toml', here / 'mappings')
    rabmv.harmonize(data, s)                                             # raises if a value is not mapped
